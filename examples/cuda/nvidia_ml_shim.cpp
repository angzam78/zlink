// zlink/examples/cuda/nvidia_ml_shim.cpp
//
// Minimal NVML (NVIDIA Management Library) shim for zlink.
//
// This shim replaces libnvidia-ml.so.1 on the CPU client machine.
// PyTorch loads libnvidia-ml.so.1 via ctypes to query GPU status,
// memory usage, device names, driver version, etc. Without this
// shim, those queries fail with "libnvidia-ml.so.1: cannot open
// shared object file" and features like torch.cuda.mem_get_info()
// break.
//
// DESIGN:
//   - Returns reasonable defaults for most queries
//   - Forwards memory/device queries to the zlink server when possible
//   - Falls back to hardcoded values if the server is unreachable
//   - Only implements the subset of NVML that PyTorch actually uses
//
// USAGE:
//   This library is built as libnvidia-ml.so.1 and placed in a
//   directory that appears first in LD_LIBRARY_PATH, so the dynamic
//   linker finds it instead of the real NVML library.

#include "cuda_api_v2.hpp"

#include <zlink/transport.hpp>
#include <zlink/tcp_transport.hpp>
#include <zlink/config.hpp>

#include <zpp_bits.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

// ── NVML type definitions ──────────────────────────────────────────────
// We define our own types to avoid depending on nvml.h

typedef int nvmlReturn_t;
typedef struct { unsigned int gpu; unsigned int memory; } nvmlUtilization_t;
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } nvmlMemory_t;
typedef struct { unsigned int gpuTemp; unsigned int memoryTemp; } nvmlTemperature_t;
typedef void* nvmlDevice_t;

// NVML return codes
#define NVML_SUCCESS                    0
#define NVML_ERROR_UNINITIALIZED        1
#define NVML_ERROR_INVALID_ARGUMENT     2
#define NVML_ERROR_NOT_SUPPORTED        3
#define NVML_ERROR_NO_PERMISSION        4
#define NVML_ERROR_ALREADY_INITIALIZED  5
#define NVML_ERROR_NOT_FOUND            6
#define NVML_ERROR_INSUFFICIENT_SIZE    7
#define NVML_ERROR_INSUFFICIENT_POWER   8
#define NVML_ERROR_DRIVER_NOT_LOADED    9
#define NVML_ERROR_TIMEOUT              10
#define NVML_ERROR_UNKNOWN              999

// NVML device persistence mode
#define NVML_FEATURE_DISABLED           0
#define NVML_FEATURE_ENABLED            1

// ── Global state ───────────────────────────────────────────────────────
static bool g_nvidia_ml_initialized = false;
static std::mutex g_nvml_mutex;

// Use the same transport as the CUDA shim if available.
// The CUDA shim connects to the server; we can share that connection
// by connecting independently (same server, same port).
static std::unique_ptr<zlink::transport> g_nvml_transport;
static bool g_nvml_connected = false;

// ── NVML RPC helpers ──────────────────────────────────────────────────
// We reuse the same cuda_v2_rpc_base protocol to query device info
// from the server. This way we get accurate memory info, device count,
// etc. without duplicating the RPC code.

static void nvml_connect() {
    if (g_nvml_connected) return;

    const char* server_env = std::getenv("ZLINK_SERVER");
    if (!server_env) return;

    std::string host = server_env;
    std::uint16_t port = zlink::default_port;
    auto colon = host.find(':');
    if (colon != std::string::npos) {
        port = static_cast<std::uint16_t>(std::stoi(host.substr(colon + 1)));
        host = host.substr(0, colon);
    }

    // Don't create a second connection — the CUDA shim already has one.
    // Instead, we'll use the CUDA driver API RPC to query device info.
    // The NVML shim just needs to return plausible values; for accurate
    // info, PyTorch can call torch.cuda.mem_get_info() which goes
    // through the CUDA Driver API shim (cuMemGetInfo).
    g_nvml_connected = true;
}

// ── Query server for device count via CUDA Driver API ─────────────────
static int get_server_device_count() {
    // Try connecting and querying via the CUDA RPC
    // For now, return 1 (the common case for remote GPU)
    // TODO: Share the transport with the CUDA shim for efficiency
    const char* server_env = std::getenv("ZLINK_SERVER");
    if (!server_env) return 0;

    try {
        std::string host = server_env;
        std::uint16_t port = zlink::default_port;
        auto colon = host.find(':');
        if (colon != std::string::npos) {
            port = static_cast<std::uint16_t>(std::stoi(host.substr(colon + 1)));
            host = host.substr(0, colon);
        }

        auto transport = zlink::make_transport(zlink::transport_kind::tcp);
        auto ec = transport->connect(host, port);
        if (ec) return 1; // Default to 1 GPU

        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        cuda_v2_rpc::client client{in, out};
        client.template request<v2_func::device_get_count>().or_throw();

        zlink::frame req_frame;
        req_frame.call_id = 1;
        req_frame.type = zlink::frame_type::request;
        req_frame.payload.assign(data.begin(), data.end());
        transport->send(req_frame);

        zlink::frame resp_frame;
        transport->receive(resp_frame);

        data.clear();
        data.assign(resp_frame.payload.begin(), resp_frame.payload.end());

        auto ret = client.template response<v2_func::device_get_count>().or_throw();
        transport->close();
        return ret.count;
    } catch (...) {
        return 1; // Default to 1 GPU on any error
    }
}

// ── Query server for memory info ──────────────────────────────────────
static bool get_server_memory_info(unsigned long long* free_bytes, unsigned long long* total_bytes) {
    const char* server_env = std::getenv("ZLINK_SERVER");
    if (!server_env) return false;

    try {
        std::string host = server_env;
        std::uint16_t port = zlink::default_port;
        auto colon = host.find(':');
        if (colon != std::string::npos) {
            port = static_cast<std::uint16_t>(std::stoi(host.substr(colon + 1)));
            host = host.substr(0, colon);
        }

        auto transport = zlink::make_transport(zlink::transport_kind::tcp);
        auto ec = transport->connect(host, port);
        if (ec) return false;

        using namespace zpp::bits;
        auto [data, in, out] = data_in_out();
        cuda_v2_rpc::client client{in, out};
        client.template request<v2_func::mem_get_info>().or_throw();

        zlink::frame req_frame;
        req_frame.call_id = 1;
        req_frame.type = zlink::frame_type::request;
        req_frame.payload.assign(data.begin(), data.end());
        transport->send(req_frame);

        zlink::frame resp_frame;
        transport->receive(resp_frame);

        data.clear();
        data.assign(resp_frame.payload.begin(), resp_frame.payload.end());

        auto ret = client.template response<v2_func::mem_get_info>().or_throw();
        transport->close();

        *free_bytes = ret.free_bytes;
        *total_bytes = ret.total_bytes;
        return true;
    } catch (...) {
        return false;
    }
}

// ══════════════════════════════════════════════════════════════════════
// EXPORTED NVML API FUNCTIONS
// ══════════════════════════════════════════════════════════════════════

extern "C" {

// ── Initialization ─────────────────────────────────────────────────────

nvmlReturn_t nvmlInit() {
    std::lock_guard lock(g_nvml_mutex);
    if (g_nvidia_ml_initialized) return NVML_ERROR_ALREADY_INITIALIZED;
    g_nvidia_ml_initialized = true;
    nvml_connect();
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInit_v2() {
    return nvmlInit();
}

nvmlReturn_t nvmlShutdown() {
    std::lock_guard lock(g_nvml_mutex);
    g_nvidia_ml_initialized = false;
    return NVML_SUCCESS;
}

// ── System queries ─────────────────────────────────────────────────────

nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
    if (!version || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    // Report a plausible driver version. The actual driver is on the
    // remote GPU server, but PyTorch mainly uses this for logging.
    const char* ver = "535.129.03"; // Common recent driver
    std::strncpy(version, ver, length - 1);
    version[length - 1] = '\0';
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) {
    if (!cudaDriverVersion) return NVML_ERROR_INVALID_ARGUMENT;
    // 12000 = CUDA 12.0, encoded as major*1000 + minor*10
    *cudaDriverVersion = 12010; // CUDA 12.1
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) {
    return nvmlSystemGetCudaDriverVersion(cudaDriverVersion);
}

nvmlReturn_t nvmlSystemGetProcessCount(unsigned int* count) {
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    *count = 1;
    return NVML_SUCCESS;
}

// ── Device queries ─────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetCount(unsigned int* deviceCount) {
    if (!deviceCount) return NVML_ERROR_INVALID_ARGUMENT;
    *deviceCount = static_cast<unsigned int>(get_server_device_count());
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) {
    return nvmlDeviceGetCount(deviceCount);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
    if (!device) return NVML_ERROR_INVALID_ARGUMENT;
    // Use the index as the handle (it's opaque to the caller)
    *device = reinterpret_cast<nvmlDevice_t>(static_cast<std::uintptr_t>(index + 1));
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    return nvmlDeviceGetHandleByIndex(index, device);
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
    if (!pciBusId || !device) return NVML_ERROR_INVALID_ARGUMENT;
    *device = reinterpret_cast<nvmlDevice_t>(static_cast<std::uintptr_t>(1));
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device) {
    return nvmlDeviceGetHandleByPciBusId(pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
    if (!name || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    // Report a generic GPU name. The real name is on the server.
    // PyTorch uses this primarily for logging/display.
    const char* gpu_name = "NVIDIA GPU (zlink remote)";
    std::strncpy(name, gpu_name, length - 1);
    name[length - 1] = '\0';
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, unsigned int* type) {
    if (!type) return NVML_ERROR_INVALID_ARGUMENT;
    *type = 0; // NVML_BRAND_UNKNOWN
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUuid(nvmlDevice_t device, char* uuid, unsigned int length) {
    if (!uuid || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    const char* id = "zlink-remote-gpu-00000000-0000-0000-0000-000000000001";
    std::strncpy(uuid, id, length - 1);
    uuid[length - 1] = '\0';
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUuid_v2(nvmlDevice_t device, char* uuid, unsigned int length) {
    return nvmlDeviceGetUuid(device, uuid, length);
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) {
    if (!index) return NVML_ERROR_INVALID_ARGUMENT;
    *index = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, void* pci) {
    // nvmlPciInfo_t — just zero it out
    if (pci) std::memset(pci, 0, 256);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, void* pci) {
    return nvmlDeviceGetPciInfo(device, pci);
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, void* pci) {
    return nvmlDeviceGetPciInfo(device, pci);
}

// ── Memory queries ─────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (!memory) return NVML_ERROR_INVALID_ARGUMENT;

    // Try to get real memory info from the server
    unsigned long long free_bytes = 0, total_bytes = 0;
    if (get_server_memory_info(&free_bytes, &total_bytes)) {
        memory->total = total_bytes;
        memory->free = free_bytes;
        memory->used = total_bytes - free_bytes;
    } else {
        // Fallback: 24GB GPU with some used
        memory->total = 24ULL * 1024 * 1024 * 1024;
        memory->free = 20ULL * 1024 * 1024 * 1024;
        memory->used = 4ULL * 1024 * 1024 * 1024;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_t* memory) {
    return nvmlDeviceGetMemoryInfo(device, memory);
}

// ── Utilization queries ────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) {
    if (!utilization) return NVML_ERROR_INVALID_ARGUMENT;
    utilization->gpu = 0;
    utilization->memory = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device,
                                              unsigned int* utilization,
                                              unsigned int* samplingPeriodUs) {
    if (!utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
    *utilization = 0;
    *samplingPeriodUs = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device,
                                              unsigned int* utilization,
                                              unsigned int* samplingPeriodUs) {
    if (!utilization || !samplingPeriodUs) return NVML_ERROR_INVALID_ARGUMENT;
    *utilization = 0;
    *samplingPeriodUs = 0;
    return NVML_SUCCESS;
}

// ── Temperature queries ────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, unsigned int sensorType,
                                       unsigned int* temp) {
    if (!temp) return NVML_ERROR_INVALID_ARGUMENT;
    *temp = 45; // Reasonable idle temperature
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, unsigned int thresholdType,
                                                 unsigned int* temp) {
    if (!temp) return NVML_ERROR_INVALID_ARGUMENT;
    *temp = 95; // Common shutdown threshold
    return NVML_SUCCESS;
}

// ── Power queries ──────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) {
    if (!power) return NVML_ERROR_INVALID_ARGUMENT;
    *power = 50000; // 50W in milliwatts
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) {
    if (!limit) return NVML_ERROR_INVALID_ARGUMENT;
    *limit = 300000; // 300W in milliwatts
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) {
    return nvmlDeviceGetPowerManagementLimit(device, limit);
}

// ── Clock queries ──────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, unsigned int type,
                                     unsigned int* clock) {
    if (!clock) return NVML_ERROR_INVALID_ARGUMENT;
    *clock = 1500; // 1500 MHz — reasonable GPU clock
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, unsigned int type,
                                        unsigned int* clock) {
    if (!clock) return NVML_ERROR_INVALID_ARGUMENT;
    *clock = 2100; // 2100 MHz — reasonable max boost clock
    return NVML_SUCCESS;
}

// ── Persistence mode ──────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, unsigned int* mode) {
    if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = NVML_FEATURE_ENABLED;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, unsigned int mode) {
    return NVML_SUCCESS;
}

// ── Compute mode ──────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, unsigned int* mode) {
    if (!mode) return NVML_ERROR_INVALID_ARGUMENT;
    *mode = 0; // NVML_COMPUTEMODE_DEFAULT
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, unsigned int mode) {
    return NVML_SUCCESS;
}

// ── ECC mode ──────────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, unsigned int* current, unsigned int* pending) {
    if (current) *current = NVML_FEATURE_DISABLED;
    if (pending) *pending = NVML_FEATURE_DISABLED;
    return NVML_SUCCESS;
}

// ── Process info ──────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device,
                                                    unsigned int* infoCount,
                                                    void* infos) {
    if (!infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    *infoCount = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device,
                                                     unsigned int* infoCount,
                                                     void* infos) {
    if (!infoCount) return NVML_ERROR_INVALID_ARGUMENT;
    *infoCount = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device,
                                           unsigned int* count,
                                           unsigned int* pids) {
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    *count = 0;
    return NVML_SUCCESS;
}

// ── Performance state ─────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, unsigned int* pState) {
    if (!pState) return NVML_ERROR_INVALID_ARGUMENT;
    *pState = 0; // NVML_PSTATE_0 = maximum performance
    return NVML_SUCCESS;
}

// ── Display mode ──────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, unsigned int* isActive) {
    if (!isActive) return NVML_ERROR_INVALID_ARGUMENT;
    *isActive = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, unsigned int* display) {
    if (!display) return NVML_ERROR_INVALID_ARGUMENT;
    *display = 0;
    return NVML_SUCCESS;
}

// ── Miscellaneous ─────────────────────────────────────────────────────

nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) {
    if (!speed) return NVML_ERROR_INVALID_ARGUMENT;
    *speed = 30; // 30% fan speed
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores) {
    if (!numCores) return NVML_ERROR_INVALID_ARGUMENT;
    *numCores = 0; // Unknown
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, unsigned int errorType,
                                           unsigned int* eccCounts) {
    if (!eccCounts) return NVML_ERROR_INVALID_ARGUMENT;
    *eccCounts = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, unsigned int errorType,
                                               unsigned int counterType,
                                               unsigned int* eccCounts) {
    if (!eccCounts) return NVML_ERROR_INVALID_ARGUMENT;
    *eccCounts = 0;
    return NVML_SUCCESS;
}

// ── Event set (stub) ──────────────────────────────────────────────────

nvmlReturn_t nvmlEventSetCreate(void** set) {
    if (!set) return NVML_ERROR_INVALID_ARGUMENT;
    *set = nullptr;
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes,
                                        void* set) {
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlEventSetWait(void* set, void* data, unsigned int timeoutms) {
    return NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlEventSetFree(void* set) {
    return NVML_SUCCESS;
}

} // extern "C"

// ── Library constructor ────────────────────────────────────────────────
__attribute__((constructor))
static void nvml_shim_init() {
    std::cerr << "zlink NVML shim loaded (libnvidia-ml.so.1 replacement)\n";
}
