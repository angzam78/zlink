// zlink/examples/cuda/cudart_shim.cpp
//
// CUDA Runtime API shim (libcudart.so.12 replacement) for zlink.
//
// This shim replaces PyTorch's bundled libcudart.so.12 on the CPU client
// machine. It implements the CUDA Runtime API functions by calling through
// to the CUDA Driver API (libcuda.so.1), which is our zlink driver shim
// that forwards calls over the network to the remote GPU server.
//
// ARCHITECTURE:
//
//   PyTorch
//     │
//     │  cudaMalloc(), cudaMemcpy(), cudaLaunchKernel(), ...
//     ▼
//   libcudart.so.12  ← THIS FILE
//     │
//     │  dlopen("libcuda.so.1") + dlsym("cuMemAlloc", ...)
//     ▼
//   libcuda.so.1  ← zlink driver shim (cuda_shim.cpp)
//     │
//     │  zlink RPC over TCP
//     ▼
//   zlink server on remote GPU machine
//     │
//     │  real cuMemAlloc(), cuLaunchKernel(), ...
//     ▼
//   NVIDIA GPU
//
// KEY DESIGN DECISIONS:
//   1. The shim loads libcuda.so.1 via dlopen and resolves cu* function
//      pointers via dlsym. This is lazy — dlopen happens on first use.
//   2. Each cuda* function calls the corresponding cu* driver function
//      through the resolved pointers.
//   3. The shim handles the mapping between Runtime API types and Driver
//      API types (e.g., cudaStream_t ↔ CUstream, cudaEvent_t ↔ CUevent).
//   4. cudaGetDeviceCount() does NOT check /dev/nvidia or do stub
//      detection — it calls cuInit + cuDeviceGetCount directly.
//   5. The shim is built as libcudart.so.12 with the correct SONAME.
//
// BUILD:
//   g++ -std=c++20 -shared -fPIC -Wl,-soname,libcudart.so.12 \
//       -o libcudart.so.12 cudart_shim.cpp -ldl
//
// USAGE (on the CPU client machine):
//   export ZLINK_SERVER=hostname:port
//   export LD_LIBRARY_PATH=/path/to/zlink-libs:$LD_LIBRARY_PATH
//   python my_script.py
//
// The directory /path/to/zlink-libs should contain:
//   - libcudart.so.12  (this file)
//   - libcuda.so.1     (the zlink driver shim)

// ══════════════════════════════════════════════════════════════════════════
// INCLUDES
// ══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <vector>

#include <dlfcn.h>
#include <pthread.h>

// ══════════════════════════════════════════════════════════════════════════
// TYPE DEFINITIONS
// ══════════════════════════════════════════════════════════════════════════
// We define our own CUDA types to avoid depending on CUDA headers.
// This shim runs on CPU-only machines that don't have the CUDA SDK.

// ── Runtime API types ──────────────────────────────────────────────────
typedef int cudaError_t;

typedef void* cudaStream_t;
typedef void* cudaEvent_t;
typedef void* cudaGraph_t;
typedef void* cudaGraphExec_t;
typedef void* cudaMemPool_t;
typedef void* cudaIpcMemHandle_t;   // opaque, 64 bytes
typedef void* cudaIpcEventHandle_t; // opaque, 64 bytes

// CUDA vector types for kernel registration
struct uint3 { unsigned int x, y, z; };
struct dim3 {
    unsigned int x, y, z;
    dim3() : x(1), y(1), z(1) {}
    dim3(unsigned int vx) : x(vx), y(1), z(1) {}
    dim3(unsigned int vx, unsigned int vy) : x(vx), y(vy), z(1) {}
    dim3(unsigned int vx, unsigned int vy, unsigned int vz) : x(vx), y(vy), z(vz) {}
};

// Texture/surface reference stubs
struct textureReference { int dummy; };
struct surfaceReference { int dummy; };

// cudaMemcpyKind
enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
};

// cudaDeviceAttr
enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxRegistersPerBlock = 11,
    cudaDevAttrClockRate = 13,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrPCIBusId = 33,
    cudaDevAttrPCIDeviceId = 34,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrComputePreemptionSupported = 43,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 49,
    cudaDevAttrManagedMemory = 46,
    cudaDevAttrPageableMemoryAccess = 50,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 51,
    cudaDevAttrCooperativeLaunch = 56,
    cudaDevAttrCooperativeMultiDeviceLaunch = 57,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 97
};

// cudaFuncAttr
enum cudaFuncAttr {
    cudaFuncAttrMaxDynamicSharedMemorySize = 6,
    cudaFuncAttrPreferredSharedMemoryCarveout = 7
};

// cudaStreamCaptureStatus
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2
};

// cudaLimit
enum cudaLimit {
    cudaLimitPrintfFifoSize = 0,
    cudaLimitMallocHeapSize = 1,
    cudaLimitStackSize = 4,
    cudaLimitDevRuntimePendingLaunchCount = 5,
    cudaLimitMaxL2FetchGranularity = 8
};

// cudaChannelFormatKind
enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3
};

// cudaMemoryType
enum cudaMemoryType {
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
};

// cudaFuncCache
enum cudaFuncCache {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3
};

// cudaSharedMemConfig
enum cudaSharedMemConfig {
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2
};

// cudaComputeMode
enum cudaComputeMode {
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3
};

// cudaEventRecordFlags
enum cudaEventRecordFlags {
    cudaEventRecordDefault = 0,
    cudaEventRecordExternal = 1
};

// cudaHostAllocFlags
enum cudaHostAllocFlags {
    cudaHostAllocDefault = 0,
    cudaHostAllocPortable = 1,
    cudaHostAllocMapped = 2,
    cudaHostAllocWriteCombined = 4
};

// cudaHostRegisterFlags
enum cudaHostRegisterFlags {
    cudaHostRegisterDefault = 0,
    cudaHostRegisterPortable = 1,
    cudaHostRegisterMapped = 2,
    cudaHostRegisterIoMemory = 4
};

// cudaMemPoolAttr
enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies = 1,
    cudaMemPoolReuseAllowInternalDependencies = 2,
    cudaMemPoolReuseAllowOpportunisticDependencies = 3,
    cudaMemPoolReleaseThreshold = 4
};

// cudaLaunchAttributeID
enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeProgrammaticStreamSerialization = 3
};

// cudaGraphInstantiateFlags
enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
    cudaGraphInstantiateFlagUpload = 2,
    cudaGraphInstantiateFlagDeviceLaunch = 4,
    cudaGraphInstantiateFlagUseNodePriority = 8
};

// ── cudaDeviceProp ─────────────────────────────────────────────────────
struct cudaDeviceProp {
    char   name[256];
    char   uuid[16];
    char   luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    regsPerBlock;
    int    warpSize;
    size_t memPitch;
    int    maxThreadsPerBlock;
    int    maxThreadsDim[3];
    int    maxGridSize[3];
    int    clockRate;
    size_t totalConstMem;
    int    major;
    int    minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int    deviceOverlap;
    int    multiProcessorCount;
    int    kernelExecTimeoutEnabled;
    int    integrated;
    int    canMapHostMemory;
    int    computeMode;
    int    maxTexture1D;
    int    maxTexture1DMipmap;
    int    maxTexture1DLinear;
    int    maxTexture2D[2];
    int    maxTexture2DMipmap[2];
    int    maxTexture2DLinear[3];
    int    maxTexture2DGather[2];
    int    maxTexture3D[3];
    int    maxTexture3DAlt[3];
    int    maxTextureCubemap;
    int    maxTexture1DLayered[2];
    int    maxTexture2DLayered[3];
    int    maxTextureCubemapLayered[2];
    int    maxSurface1D;
    int    maxSurface2D[2];
    int    maxSurface3D[3];
    int    maxSurface1DLayered[2];
    int    maxSurface2DLayered[3];
    int    maxSurfaceCubemap;
    int    maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int    concurrentKernels;
    int    ECCEnabled;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
    int    tccDriver;
    int    asyncEngineCount;
    int    unifiedAddressing;
    int    memoryClockRate;
    int    memoryBusWidth;
    int    l2CacheSize;
    int    persistingL2CacheMaxSize;
    int    maxThreadsPerMultiProcessor;
    size_t streamPrioritiesSupported;
    int    globalL1CacheSupported;
    int    localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int    regsPerMultiprocessor;
    int    managedMemory;
    int    isMultiGpuBoard;
    int    multiGpuBoardGroupID;
    int    hostNativeAtomicSupported;
    int    singleToDoublePrecisionPerfRatio;
    int    pageableMemoryAccess;
    int    concurrentManagedAccess;
    int    computePreemptionSupported;
    int    canUseHostPointerForRegisteredMem;
    int    cooperativeLaunch;
    int    cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int    pageableMemoryAccessUsesHostPageTables;
    int    directManagedMemAccessFromHost;
    int    maxBlocksPerMultiProcessor;
    int    accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
};

// ── cudaPointerAttributes ──────────────────────────────────────────────
struct cudaPointerAttributes {
    int    type;               // cudaMemoryType
    int    device;
    void*  devicePointer;
    void*  hostPointer;
    int    isManaged;
};

// ── cudaLaunchConfig ───────────────────────────────────────────────────
struct cudaLaunchAttr {
    unsigned int id;           // cudaLaunchAttributeID
    unsigned int pad;
    union {
        unsigned long long val;
        struct { unsigned long long v0, v1; } pair;
    } value;
};

struct cudaLaunchConfig {
    unsigned int numAttrs;
    unsigned int pad;
    cudaLaunchAttr* attrs;
};

// ── cudaFuncAttributes ─────────────────────────────────────────────────
struct cudaFuncAttributes {
    size_t constSizeBytes;
    size_t localSizeBytes;
    int    maxThreadsPerBlock;
    int    numRegs;
    int    ptxVersion;
    int    binaryVersion;
    size_t cachePolicyCA;
    int    maxDynamicSharedSizeBytes;
    int    preferredShmemCarveout;
};

// ── cudaIpcMemHandle / cudaIpcEventHandle ──────────────────────────────
struct cudaIpcMemHandle_st {
    char reserved[64];
};
struct cudaIpcEventHandle_st {
    char reserved[64];
};

// ── CUDA error codes ───────────────────────────────────────────────────
#define cudaSuccess                         0
#define cudaErrorMissingConfiguration       1
#define cudaErrorMemoryAllocation           2
#define cudaErrorInitializationError        3
#define cudaErrorLaunchFailure              4
#define cudaErrorPriorLaunchFailure         5
#define cudaErrorLaunchTimeout              6
#define cudaErrorLaunchOutOfResources       7
#define cudaErrorInvalidDeviceFunction      8
#define cudaErrorInvalidConfiguration       9
#define cudaErrorInvalidDevice              10
#define cudaErrorInvalidValue               11
#define cudaErrorInvalidPitchValue          12
#define cudaErrorInvalidDevicePointer       17
#define cudaErrorInvalidMemcpyDirection     21
#define cudaErrorInsufficientDriver         35
#define cudaErrorHardwareStackError         716
#define cudaErrorIllegalInstruction         717
#define cudaErrorMisalignedAddress          718
#define cudaErrorInvalidAddressSpace        719
#define cudaErrorInvalidPc                  720
#define cudaErrorIllegalAddress             721
#define cudaErrorInvalidKernelImage         47
#define cudaErrorContextAlreadyCurrent      56
#define cudaErrorMapBufferObjectFailed      59
#define cudaErrorUnmapBufferObjectFailed    60
#define cudaErrorNotSupported               80
#define cudaErrorNotReady                   34
#define cudaErrorCudartUnloading            29
#define cudaErrorUnknown                    30
#define cudaErrorDeviceUninitialized        100
#define cudaErrorPeerAccessUnsupported      214
#define cudaErrorInvalidResourceHandle      420
#define cudaErrorNotPermitted               800
#define cudaErrorInvalidGraphicsContext     901
#define cudaErrorStreamCaptureUnsupported   900
#define cudaErrorStreamCaptureInvalidated   901
#define cudaErrorIpcNotSupported            801
#define cudaErrorCooperativeLaunchTooLarge  904

// ══════════════════════════════════════════════════════════════════════════
// DRIVER API TYPE DEFINITIONS (for dlsym resolution)
// ══════════════════════════════════════════════════════════════════════════

typedef int  CUresult;
typedef int  CUdevice;
typedef void* CUcontext;
typedef void* CUstream;
typedef void* CUevent;
typedef void* CUmodule;
typedef void* CUfunction;
typedef std::uint64_t CUdeviceptr;

// CUdevice_attribute enum values we need
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK                1
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                      2
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                      3
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                      4
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                       5
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                       6
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                       7
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK          8
#define CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY                9
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE                            10
#define CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK              11
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE                           13
#define CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                    14
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                 16
#define CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT                  18
#define CU_DEVICE_ATTRIBUTE_INTEGRATED                           19
#define CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY                  20
#define CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                         22
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                   31
#define CU_DEVICE_ATTRIBUTE_ECC_ENABLED                          32
#define CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                           33
#define CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                        34
#define CU_DEVICE_ATTRIBUTE_TCC_DRIVER                           35
#define CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                    36
#define CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH              37
#define CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                        38
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR       39
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR             75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR             76
#define CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED         43
#define CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY                       46
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD                      47
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID             48
#define CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM 49
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS               50
#define CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES 51
#define CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH                   56
#define CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH      57
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR 97
#define CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED         102
#define CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO 103
#define CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS            104
#define CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST  105
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR        106
#define CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE         82
#define CU_DEVICE_ATTRIBUTE_ACCESS_POLICY_MAX_WINDOW_SIZE        83
#define CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK     111
#define CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED          68
#define CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED            69
#define CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED             70
#define CU_DEVICE_ATTRIBUTE_MAX_SURFACE1D                        57
#define CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                   40
#define CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                   41

#define CU_EVENT_DISABLE_TIMING         0x2
#define CU_EVENT_INTERPROCESS           0x4
#define CU_STREAM_DEFAULT               0x0
#define CU_STREAM_NON_BLOCKING          0x1

#define CU_MEMHOSTALLOC_PORTABLE        0x1
#define CU_MEMHOSTALLOC_DEVICEMAP       0x2
#define CU_MEMHOSTALLOC_WRITECOMBINED   0x4

#define CU_MEMHOSTREGISTER_PORTABLE     0x1
#define CU_MEMHOSTREGISTER_DEVICEMAP    0x2

#define CUDA_SUCCESS 0

// ══════════════════════════════════════════════════════════════════════════
// DRIVER API FUNCTION POINTER TYPES
// ══════════════════════════════════════════════════════════════════════════

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_t)(std::size_t*, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_t)(int*, int, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext*, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRelease_t)(CUdevice);
typedef CUresult (*cuDevicePrimaryCtxSetFlags_t)(CUdevice, unsigned int);
typedef CUresult (*cuDevicePrimaryCtxGetState_t)(CUcontext*, unsigned int*, int*);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuCtxGetCurrent_t)(CUcontext*);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuCtxPushCurrent_t)(CUcontext);
typedef CUresult (*cuCtxPopCurrent_t)(CUcontext*);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, std::size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemFreeHost_t)(void*);
typedef CUresult (*cuMemHostAlloc_t)(void**, std::size_t, unsigned int);
typedef CUresult (*cuMemHostRegister_t)(void*, std::size_t, unsigned int);
typedef CUresult (*cuMemHostUnregister_t)(void*);
typedef CUresult (*cuMemHostGetDevicePointer_t)(CUdeviceptr*, void*, unsigned int);
typedef CUresult (*cuMemGetInfo_t)(std::size_t*, std::size_t*);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, std::size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, std::size_t);
typedef CUresult (*cuMemcpyDtoD_t)(CUdeviceptr, CUdeviceptr, std::size_t);
typedef CUresult (*cuMemcpyHtoDAsync_t)(CUdeviceptr, const void*, std::size_t, CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_t)(void*, CUdeviceptr, std::size_t, CUstream);
typedef CUresult (*cuStreamCreate_t)(CUstream*, unsigned int);
typedef CUresult (*cuStreamCreateWithPriority_t)(CUstream*, unsigned int, int);
typedef CUresult (*cuStreamDestroy_t)(CUstream);
typedef CUresult (*cuStreamSynchronize_t)(CUstream);
typedef CUresult (*cuEventCreate_t)(CUevent*, unsigned int);
typedef CUresult (*cuEventDestroy_t)(CUevent);
typedef CUresult (*cuEventRecord_t)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_t)(CUevent);
typedef CUresult (*cuEventElapsedTime_t)(float*, CUevent, CUevent);
typedef CUresult (*cuLaunchKernel_t)(CUfunction,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void**, void**);
typedef CUresult (*cuDriverGetVersion_t)(int*);
typedef CUresult (*cuMemsetD8_t)(CUdeviceptr, unsigned char, std::size_t);
typedef CUresult (*cuMemsetD8Async_t)(CUdeviceptr, unsigned char, std::size_t, CUstream);
typedef CUresult (*cuMemsetD16_t)(CUdeviceptr, unsigned short, std::size_t);
typedef CUresult (*cuMemsetD32_t)(CUdeviceptr, unsigned int, std::size_t);
typedef CUresult (*cuMemsetD32Async_t)(CUdeviceptr, unsigned int, std::size_t, CUstream);
typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
typedef CUresult (*cuGetProcAddress_t)(const char*, void**, int, unsigned long long, void*);

// ══════════════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ══════════════════════════════════════════════════════════════════════════

static void* g_driver_lib = nullptr;
static std::mutex g_init_mutex;
static std::atomic<bool> g_driver_loaded{false};
static std::atomic<bool> g_init_called{false};

// ── Resolved driver function pointers ──────────────────────────────────
static cuInit_t                     p_cuInit = nullptr;
static cuDeviceGet_t                p_cuDeviceGet = nullptr;
static cuDeviceGetCount_t           p_cuDeviceGetCount = nullptr;
static cuDeviceGetName_t            p_cuDeviceGetName = nullptr;
static cuDeviceTotalMem_t           p_cuDeviceTotalMem = nullptr;
static cuDeviceGetAttribute_t       p_cuDeviceGetAttribute = nullptr;
static cuDevicePrimaryCtxRetain_t   p_cuDevicePrimaryCtxRetain = nullptr;
static cuDevicePrimaryCtxRelease_t  p_cuDevicePrimaryCtxRelease = nullptr;
static cuDevicePrimaryCtxSetFlags_t p_cuDevicePrimaryCtxSetFlags = nullptr;
static cuDevicePrimaryCtxGetState_t p_cuDevicePrimaryCtxGetState = nullptr;
static cuCtxSetCurrent_t            p_cuCtxSetCurrent = nullptr;
static cuCtxGetCurrent_t            p_cuCtxGetCurrent = nullptr;
static cuCtxSynchronize_t           p_cuCtxSynchronize = nullptr;
static cuCtxPushCurrent_t           p_cuCtxPushCurrent = nullptr;
static cuCtxPopCurrent_t            p_cuCtxPopCurrent = nullptr;
static cuMemAlloc_t                 p_cuMemAlloc = nullptr;
static cuMemFree_t                  p_cuMemFree = nullptr;
static cuMemFreeHost_t              p_cuMemFreeHost = nullptr;
static cuMemHostAlloc_t             p_cuMemHostAlloc = nullptr;
static cuMemHostRegister_t          p_cuMemHostRegister = nullptr;
static cuMemHostUnregister_t        p_cuMemHostUnregister = nullptr;
static cuMemHostGetDevicePointer_t  p_cuMemHostGetDevicePointer = nullptr;
static cuMemGetInfo_t               p_cuMemGetInfo = nullptr;
static cuMemcpyHtoD_t               p_cuMemcpyHtoD = nullptr;
static cuMemcpyDtoH_t               p_cuMemcpyDtoH = nullptr;
static cuMemcpyDtoD_t               p_cuMemcpyDtoD = nullptr;
static cuMemcpyHtoDAsync_t          p_cuMemcpyHtoDAsync = nullptr;
static cuMemcpyDtoHAsync_t          p_cuMemcpyDtoHAsync = nullptr;
static cuStreamCreate_t             p_cuStreamCreate = nullptr;
static cuStreamCreateWithPriority_t p_cuStreamCreateWithPriority = nullptr;
static cuStreamDestroy_t            p_cuStreamDestroy = nullptr;
static cuStreamSynchronize_t        p_cuStreamSynchronize = nullptr;
static cuEventCreate_t              p_cuEventCreate = nullptr;
static cuEventDestroy_t             p_cuEventDestroy = nullptr;
static cuEventRecord_t              p_cuEventRecord = nullptr;
static cuEventSynchronize_t         p_cuEventSynchronize = nullptr;
static cuEventElapsedTime_t         p_cuEventElapsedTime = nullptr;
static cuLaunchKernel_t             p_cuLaunchKernel = nullptr;
static cuDriverGetVersion_t         p_cuDriverGetVersion = nullptr;
static cuMemsetD8_t                 p_cuMemsetD8 = nullptr;
static cuMemsetD8Async_t            p_cuMemsetD8Async = nullptr;
static cuMemsetD16_t                p_cuMemsetD16 = nullptr;
static cuMemsetD32_t                p_cuMemsetD32 = nullptr;
static cuMemsetD32Async_t           p_cuMemsetD32Async = nullptr;
static cuModuleLoadData_t           p_cuModuleLoadData = nullptr;
static cuModuleUnload_t             p_cuModuleUnload = nullptr;
static cuModuleGetFunction_t        p_cuModuleGetFunction = nullptr;
static cuGetProcAddress_t           p_cuGetProcAddress = nullptr;

// ── Current device tracking ───────────────────────────────────────────
static std::atomic<int> g_current_device{0};

// ── Primary context cache per device ───────────────────────────────────
// cudaSetDevice maps to cuCtxSetCurrent with the primary context.
// We cache the primary contexts to avoid re-retaining them.
static std::mutex g_ctx_mutex;
static std::unordered_map<int, CUcontext> g_primary_contexts; // device → CUcontext

// ── Last error per-thread ─────────────────────────────────────────────
static __thread cudaError_t g_last_error = cudaSuccess;

// ── Device pointer tracking for cudaPointerGetAttributes ──────────────
static std::mutex g_ptr_mutex;
struct ptr_info {
    int memory_type;    // cudaMemoryType
    int device;
    void* host_ptr;
};
static std::unordered_map<std::uintptr_t, ptr_info> g_tracked_ptrs;

// ── Host allocated pointer tracking (for cudaFreeHost) ────────────────
static std::mutex g_host_alloc_mutex;
static std::unordered_set<std::uintptr_t> g_host_alloced_ptrs;

// ── Dummy mem pool handle ─────────────────────────────────────────────
static int g_dummy_pool = 0xDEAD;

// ══════════════════════════════════════════════════════════════════════════
// DRIVER LOADING
// ══════════════════════════════════════════════════════════════════════════

#define RESOLVE(name) \
    p_##name = reinterpret_cast<name##_t>(dlsym(g_driver_lib, #name)); \
    if (!p_##name) { \
        std::fprintf(stderr, "[cudart_shim] WARNING: could not resolve %s\n", #name); \
    }

static bool ensure_driver_loaded() {
    if (g_driver_loaded.load(std::memory_order_acquire)) return true;

    std::lock_guard lock(g_init_mutex);
    if (g_driver_loaded.load(std::memory_order_relaxed)) return true;

    std::fprintf(stderr, "[cudart_shim] Loading libcuda.so.1 via dlopen...\n");

    g_driver_lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_driver_lib) {
        std::fprintf(stderr, "[cudart_shim] FATAL: dlopen(\"libcuda.so.1\") failed: %s\n",
                     dlerror());
        return false;
    }

    // Resolve all driver API function pointers
    RESOLVE(cuInit);
    RESOLVE(cuDeviceGet);
    RESOLVE(cuDeviceGetCount);
    RESOLVE(cuDeviceGetName);
    RESOLVE(cuDeviceTotalMem);
    RESOLVE(cuDeviceGetAttribute);
    RESOLVE(cuDevicePrimaryCtxRetain);
    RESOLVE(cuDevicePrimaryCtxRelease);
    RESOLVE(cuDevicePrimaryCtxSetFlags);
    RESOLVE(cuDevicePrimaryCtxGetState);
    RESOLVE(cuCtxSetCurrent);
    RESOLVE(cuCtxGetCurrent);
    RESOLVE(cuCtxSynchronize);
    RESOLVE(cuCtxPushCurrent);
    RESOLVE(cuCtxPopCurrent);
    RESOLVE(cuMemAlloc);
    RESOLVE(cuMemFree);
    RESOLVE(cuMemFreeHost);
    RESOLVE(cuMemHostAlloc);
    RESOLVE(cuMemHostRegister);
    RESOLVE(cuMemHostUnregister);
    RESOLVE(cuMemHostGetDevicePointer);
    RESOLVE(cuMemGetInfo);
    RESOLVE(cuMemcpyHtoD);
    RESOLVE(cuMemcpyDtoH);
    RESOLVE(cuMemcpyDtoD);
    RESOLVE(cuMemcpyHtoDAsync);
    RESOLVE(cuMemcpyDtoHAsync);
    RESOLVE(cuStreamCreate);
    RESOLVE(cuStreamCreateWithPriority);
    RESOLVE(cuStreamDestroy);
    RESOLVE(cuStreamSynchronize);
    RESOLVE(cuEventCreate);
    RESOLVE(cuEventDestroy);
    RESOLVE(cuEventRecord);
    RESOLVE(cuEventSynchronize);
    RESOLVE(cuEventElapsedTime);
    RESOLVE(cuLaunchKernel);
    RESOLVE(cuDriverGetVersion);
    RESOLVE(cuMemsetD8);
    RESOLVE(cuMemsetD8Async);
    RESOLVE(cuMemsetD16);
    RESOLVE(cuMemsetD32);
    RESOLVE(cuMemsetD32Async);
    RESOLVE(cuModuleLoadData);
    RESOLVE(cuModuleUnload);
    RESOLVE(cuModuleGetFunction);
    RESOLVE(cuGetProcAddress);

    g_driver_loaded.store(true, std::memory_order_release);
    std::fprintf(stderr, "[cudart_shim] Driver loaded successfully, all cu* pointers resolved\n");
    return true;
}

#undef RESOLVE

// ══════════════════════════════════════════════════════════════════════════
// ERROR TRANSLATION
// ══════════════════════════════════════════════════════════════════════════

static cudaError_t cu_to_cuda_error(CUresult r) {
    switch (r) {
        case 0:  return cudaSuccess;
        case 1:  return cudaErrorInvalidValue;
        case 2:  return cudaErrorMemoryAllocation;
        case 3:  return cudaErrorInitializationError;
        case 4:  return cudaErrorPriorLaunchFailure;
        case 5:  return cudaErrorLaunchFailure;
        case 6:  return cudaErrorLaunchTimeout;
        case 7:  return cudaErrorLaunchOutOfResources;
        case 8:  return cudaErrorInvalidDeviceFunction;
        case 9:  return cudaErrorInvalidConfiguration;
        case 10: return cudaErrorInvalidDevice;
        case 11: return cudaErrorInvalidValue;
        case 13: return cudaErrorInvalidMemcpyDirection;
        case 17: return cudaErrorInvalidDevicePointer;
        case 18: return cudaErrorInvalidResourceHandle;
        case 34: return cudaErrorNotReady;
        case 35: return cudaErrorInsufficientDriver;
        case 46: return cudaErrorInvalidKernelImage;
        case 701: return cudaErrorNotSupported;
        case 720: return cudaErrorIpcNotSupported;
        default:  return cudaErrorUnknown;
    }
}

static void set_last_error(cudaError_t e) {
    g_last_error = e;
}

// ══════════════════════════════════════════════════════════════════════════
// PRIMARY CONTEXT HELPER
// ══════════════════════════════════════════════════════════════════════════
// cudaSetDevice(device) maps to cuCtxSetCurrent(primary_ctx[device]).
// We retain the primary context for the given device and cache it.

static CUcontext get_or_retain_primary_context(int device) {
    {
        std::lock_guard lock(g_ctx_mutex);
        auto it = g_primary_contexts.find(device);
        if (it != g_primary_contexts.end()) {
            return it->second;
        }
    }

    // Need to retain the primary context
    if (!p_cuDevicePrimaryCtxRetain) return nullptr;
    if (!p_cuDeviceGet) return nullptr;

    CUdevice cu_dev;
    CUresult r = p_cuDeviceGet(&cu_dev, device);
    if (r != CUDA_SUCCESS) {
        std::fprintf(stderr, "[cudart_shim] cuDeviceGet(%d) failed: %d\n", device, r);
        return nullptr;
    }

    CUcontext ctx = nullptr;
    r = p_cuDevicePrimaryCtxRetain(&ctx, cu_dev);
    if (r != CUDA_SUCCESS) {
        std::fprintf(stderr, "[cudart_shim] cuDevicePrimaryCtxRetain(%d) failed: %d\n", device, r);
        return nullptr;
    }

    std::lock_guard lock(g_ctx_mutex);
    g_primary_contexts[device] = ctx;
    std::fprintf(stderr, "[cudart_shim] Retained primary context for device %d: %p\n",
                 device, (void*)ctx);
    return ctx;
}

// ══════════════════════════════════════════════════════════════════════════
// CUDA DEVICE ATTRIBUTE MAPPING
// ══════════════════════════════════════════════════════════════════════════
// Maps cudaDeviceAttr values to CUdevice_attribute values.

static int map_device_attr(cudaDeviceAttr attr) {
    switch (attr) {
        case cudaDevAttrMaxThreadsPerBlock:         return CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
        case cudaDevAttrMaxBlockDimX:               return CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
        case cudaDevAttrMaxBlockDimY:               return CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
        case cudaDevAttrMaxBlockDimZ:               return CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
        case cudaDevAttrMaxGridDimX:                return CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
        case cudaDevAttrMaxGridDimY:                return CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
        case cudaDevAttrMaxGridDimZ:                return CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
        case cudaDevAttrMaxSharedMemoryPerBlock:    return CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
        case cudaDevAttrTotalConstantMemory:        return CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY;
        case cudaDevAttrWarpSize:                   return CU_DEVICE_ATTRIBUTE_WARP_SIZE;
        case cudaDevAttrMaxRegistersPerBlock:       return CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
        case cudaDevAttrClockRate:                  return CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
        case cudaDevAttrMultiProcessorCount:        return CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
        case cudaDevAttrComputeCapabilityMajor:     return CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
        case cudaDevAttrComputeCapabilityMinor:     return CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
        case cudaDevAttrConcurrentKernels:          return CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS;
        case cudaDevAttrPCIBusId:                   return CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
        case cudaDevAttrPCIDeviceId:                return CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
        case cudaDevAttrCanMapHostMemory:           return CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY;
        case cudaDevAttrEccEnabled:                 return CU_DEVICE_ATTRIBUTE_ECC_ENABLED;
        case cudaDevAttrMemoryClockRate:            return CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
        case cudaDevAttrGlobalMemoryBusWidth:       return CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
        case cudaDevAttrL2CacheSize:                return CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
        case cudaDevAttrMaxThreadsPerMultiProcessor:return CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
        case cudaDevAttrComputePreemptionSupported: return CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED;
        case cudaDevAttrCanUseHostPointerForRegisteredMem: return CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM;
        case cudaDevAttrManagedMemory:             return CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
        case cudaDevAttrPageableMemoryAccess:       return CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS;
        case cudaDevAttrPageableMemoryAccessUsesHostPageTables: return CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES;
        case cudaDevAttrCooperativeLaunch:          return CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH;
        case cudaDevAttrCooperativeMultiDeviceLaunch: return CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH;
        case cudaDevAttrMaxSharedMemoryPerMultiprocessor: return CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR;
        default: return -1; // Unknown
    }
}

// ══════════════════════════════════════════════════════════════════════════
// EXPORTED CUDA RUNTIME API FUNCTIONS
// ══════════════════════════════════════════════════════════════════════════

extern "C" {

// ──────────────────────────────────────────────────────────────────────────
// Core initialization
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;

    if (!p_cuInit || !p_cuDeviceGetCount) return cudaErrorInitializationError;

    // Call cuInit first (idempotent after first call), then get count.
    // We do NOT check /dev/nvidia or do stub detection.
    if (!g_init_called.load(std::memory_order_acquire)) {
        CUresult r = p_cuInit(0);
        if (r != CUDA_SUCCESS) {
            // If cuInit fails, report 0 devices (no GPU available)
            *count = 0;
            return cudaSuccess;
        }
        g_init_called.store(true, std::memory_order_release);
    }

    CUresult r = p_cuDeviceGetCount(count);
    if (r != CUDA_SUCCESS) {
        set_last_error(cu_to_cuda_error(r));
        return cu_to_cuda_error(r);
    }
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    if (device < 0) return cudaErrorInvalidDevice;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;

    // Ensure cuInit has been called
    if (!g_init_called.load(std::memory_order_acquire)) {
        if (p_cuInit) {
            CUresult r = p_cuInit(0);
            if (r != CUDA_SUCCESS) {
                set_last_error(cu_to_cuda_error(r));
                return cu_to_cuda_error(r);
            }
            g_init_called.store(true, std::memory_order_release);
        }
    }

    CUcontext ctx = get_or_retain_primary_context(device);
    if (!ctx) {
        set_last_error(cudaErrorInvalidDevice);
        return cudaErrorInvalidDevice;
    }

    if (p_cuCtxSetCurrent) {
        CUresult r = p_cuCtxSetCurrent(ctx);
        if (r != CUDA_SUCCESS) {
            set_last_error(cu_to_cuda_error(r));
            return cu_to_cuda_error(r);
        }
    }

    g_current_device.store(device, std::memory_order_release);
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) {
    if (!device) return cudaErrorInvalidValue;
    *device = g_current_device.load(std::memory_order_acquire);
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuCtxSynchronize) return cudaErrorInitializationError;

    CUresult r = p_cuCtxSynchronize();
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaDriverGetVersion(int* driverVersion) {
    if (!driverVersion) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) {
        // If we can't load the driver, return a default
        *driverVersion = 12040;
        return cudaSuccess;
    }
    if (p_cuDriverGetVersion) {
        CUresult r = p_cuDriverGetVersion(driverVersion);
        return cu_to_cuda_error(r);
    }
    *driverVersion = 12040;
    return cudaSuccess;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    if (!runtimeVersion) return cudaErrorInvalidValue;
    // CUDA 12.4 runtime
    *runtimeVersion = 12040;
    return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:                      return "no error";
        case cudaErrorMissingConfiguration:    return "missing configuration";
        case cudaErrorMemoryAllocation:        return "out of memory";
        case cudaErrorInitializationError:     return "initialization error";
        case cudaErrorLaunchFailure:           return "launch failure";
        case cudaErrorPriorLaunchFailure:      return "prior launch failure";
        case cudaErrorLaunchTimeout:           return "launch timeout";
        case cudaErrorLaunchOutOfResources:    return "launch out of resources";
        case cudaErrorInvalidDeviceFunction:   return "invalid device function";
        case cudaErrorInvalidConfiguration:    return "invalid configuration";
        case cudaErrorInvalidDevice:           return "invalid device";
        case cudaErrorInvalidValue:            return "invalid argument";
        case cudaErrorInvalidPitchValue:       return "invalid pitch value";
        case cudaErrorInvalidDevicePointer:    return "invalid device pointer";
        case cudaErrorInvalidMemcpyDirection:  return "invalid memcpy direction";
        case cudaErrorInsufficientDriver:      return "insufficient CUDA driver";
        case cudaErrorNotSupported:            return "operation not supported";
        case cudaErrorNotReady:                return "device not ready";
        case cudaErrorCudartUnloading:         return "CUDA runtime unloading";
        case cudaErrorUnknown:                 return "unknown error";
        case cudaErrorDeviceUninitialized:     return "device uninitialized";
        case cudaErrorPeerAccessUnsupported:   return "peer access unsupported";
        case cudaErrorInvalidResourceHandle:   return "invalid resource handle";
        case cudaErrorIpcNotSupported:         return "IPC not supported";
        default:                               return "unknown CUDA error";
    }
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:                      return "cudaSuccess";
        case cudaErrorMissingConfiguration:    return "cudaErrorMissingConfiguration";
        case cudaErrorMemoryAllocation:        return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:     return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure:           return "cudaErrorLaunchFailure";
        case cudaErrorPriorLaunchFailure:      return "cudaErrorPriorLaunchFailure";
        case cudaErrorLaunchTimeout:           return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources:    return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction:   return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration:    return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice:           return "cudaErrorInvalidDevice";
        case cudaErrorInvalidValue:            return "cudaErrorInvalidValue";
        case cudaErrorInvalidPitchValue:       return "cudaErrorInvalidPitchValue";
        case cudaErrorInvalidDevicePointer:    return "cudaErrorInvalidDevicePointer";
        case cudaErrorInvalidMemcpyDirection:  return "cudaErrorInvalidMemcpyDirection";
        case cudaErrorInsufficientDriver:      return "cudaErrorInsufficientDriver";
        case cudaErrorNotSupported:            return "cudaErrorNotSupported";
        case cudaErrorNotReady:                return "cudaErrorNotReady";
        case cudaErrorCudartUnloading:         return "cudaErrorCudartUnloading";
        case cudaErrorUnknown:                 return "cudaErrorUnknown";
        case cudaErrorDeviceUninitialized:     return "cudaErrorDeviceUninitialized";
        case cudaErrorPeerAccessUnsupported:   return "cudaErrorPeerAccessUnsupported";
        case cudaErrorInvalidResourceHandle:   return "cudaErrorInvalidResourceHandle";
        case cudaErrorIpcNotSupported:         return "cudaErrorIpcNotSupported";
        default:                               return "cudaErrorUnknown";
    }
}

cudaError_t cudaGetLastError() {
    cudaError_t e = g_last_error;
    g_last_error = cudaSuccess;
    return e;
}

cudaError_t cudaPeekAtLastError() {
    return g_last_error;
}

// ──────────────────────────────────────────────────────────────────────────
// Memory management
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemAlloc) return cudaErrorInitializationError;

    CUdeviceptr ptr = 0;
    CUresult r = p_cuMemAlloc(&ptr, size);
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) {
        *devPtr = nullptr;
        set_last_error(e);
        return e;
    }

    *devPtr = reinterpret_cast<void*>(ptr);

    // Track this pointer for cudaPointerGetAttributes
    {
        std::lock_guard lock(g_ptr_mutex);
        g_tracked_ptrs[ptr] = {
            cudaMemoryTypeDevice,
            g_current_device.load(std::memory_order_acquire),
            nullptr
        };
    }

    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    if (!devPtr) return cudaSuccess;  // cudaFree(nullptr) is a no-op
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemFree) return cudaErrorInitializationError;

    CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(devPtr);
    CUresult r = p_cuMemFree(ptr);
    cudaError_t e = cu_to_cuda_error(r);

    // Untrack
    {
        std::lock_guard lock(g_ptr_mutex);
        g_tracked_ptrs.erase(ptr);
    }

    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream) {
    // Async memory pools not supported — fall back to synchronous allocation
    (void)stream;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream) {
    // Async memory pools not supported — fall back to synchronous free
    (void)stream;
    return cudaFree(devPtr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;

    CUresult r = CUDA_SUCCESS;

    switch (kind) {
        case cudaMemcpyHostToDevice: {
            if (!p_cuMemcpyHtoD) return cudaErrorInitializationError;
            r = p_cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(dst), src, count);
            break;
        }
        case cudaMemcpyDeviceToHost: {
            if (!p_cuMemcpyDtoH) return cudaErrorInitializationError;
            r = p_cuMemcpyDtoH(dst, reinterpret_cast<CUdeviceptr>(src), count);
            break;
        }
        case cudaMemcpyDeviceToDevice: {
            if (!p_cuMemcpyDtoD) return cudaErrorInitializationError;
            r = p_cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(dst),
                               reinterpret_cast<CUdeviceptr>(src), count);
            break;
        }
        case cudaMemcpyHostToHost: {
            // Pure host copy — just memcpy
            std::memcpy(dst, src, count);
            return cudaSuccess;
        }
        case cudaMemcpyDefault: {
            // Try to determine direction from tracked pointers
            CUdeviceptr dst_addr = reinterpret_cast<CUdeviceptr>(dst);
            CUdeviceptr src_addr = reinterpret_cast<CUdeviceptr>(src);

            bool dst_is_device = false, src_is_device = false;
            {
                std::lock_guard lock(g_ptr_mutex);
                dst_is_device = g_tracked_ptrs.count(dst_addr) > 0;
                src_is_device = g_tracked_ptrs.count(src_addr) > 0;
            }

            if (src_is_device && dst_is_device) {
                if (!p_cuMemcpyDtoD) return cudaErrorInitializationError;
                r = p_cuMemcpyDtoD(dst_addr, src_addr, count);
            } else if (src_is_device && !dst_is_device) {
                if (!p_cuMemcpyDtoH) return cudaErrorInitializationError;
                r = p_cuMemcpyDtoH(dst, src_addr, count);
            } else if (!src_is_device && dst_is_device) {
                if (!p_cuMemcpyHtoD) return cudaErrorInitializationError;
                r = p_cuMemcpyHtoD(dst_addr, src, count);
            } else {
                std::memcpy(dst, src, count);
                return cudaSuccess;
            }
            break;
        }
        default:
            set_last_error(cudaErrorInvalidMemcpyDirection);
            return cudaErrorInvalidMemcpyDirection;
    }

    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    // Synchronous fallback — the zlink driver shim serializes everything anyway.
    // We ignore the stream parameter for now.
    (void)stream;
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    if (!devPtr) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;

    CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(devPtr);

    // Use cuMemsetD8 for byte-level memset
    if (p_cuMemsetD8) {
        CUresult r = p_cuMemsetD8(ptr, static_cast<unsigned char>(value), count);
        cudaError_t e = cu_to_cuda_error(r);
        if (e != cudaSuccess) set_last_error(e);
        return e;
    }

    // Fallback: write bytes directly via HtoD (very slow, but functional)
    // This shouldn't happen if the driver is loaded correctly
    set_last_error(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    // Synchronous fallback
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    if (!free || !total) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemGetInfo) return cudaErrorInitializationError;

    CUresult r = p_cuMemGetInfo(free, total);
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    if (!pHost) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemHostAlloc) return cudaErrorInitializationError;

    // Map cudaHostAllocFlags to CU_MEMHOSTALLOC flags
    unsigned int cu_flags = 0;
    if (flags & cudaHostAllocPortable)    cu_flags |= CU_MEMHOSTALLOC_PORTABLE;
    if (flags & cudaHostAllocMapped)      cu_flags |= CU_MEMHOSTALLOC_DEVICEMAP;
    if (flags & cudaHostAllocWriteCombined) cu_flags |= CU_MEMHOSTALLOC_WRITECOMBINED;

    CUresult r = p_cuMemHostAlloc(pHost, size, cu_flags);
    cudaError_t e = cu_to_cuda_error(r);
    if (e == cudaSuccess && *pHost) {
        // Track the host allocation
        std::lock_guard lock(g_host_alloc_mutex);
        g_host_alloced_ptrs.insert(reinterpret_cast<std::uintptr_t>(*pHost));

        // If mapped, also track the device pointer
        if (flags & cudaHostAllocMapped) {
            CUdeviceptr dev_ptr = 0;
            if (p_cuMemHostGetDevicePointer) {
                p_cuMemHostGetDevicePointer(&dev_ptr, *pHost, 0);
                if (dev_ptr) {
                    std::lock_guard lock2(g_ptr_mutex);
                    g_tracked_ptrs[dev_ptr] = {
                        cudaMemoryTypeHost,
                        g_current_device.load(std::memory_order_acquire),
                        *pHost
                    };
                }
            }
        }
    }
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    if (!ptr) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemHostRegister) return cudaErrorInitializationError;

    // Map cudaHostRegisterFlags to CU_MEMHOSTREGISTER flags
    unsigned int cu_flags = 0;
    if (flags & cudaHostRegisterPortable) cu_flags |= CU_MEMHOSTREGISTER_PORTABLE;
    if (flags & cudaHostRegisterMapped)   cu_flags |= CU_MEMHOSTREGISTER_DEVICEMAP;

    CUresult r = p_cuMemHostRegister(ptr, size, cu_flags);
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaHostUnregister(void* ptr) {
    if (!ptr) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemHostUnregister) return cudaErrorInitializationError;

    CUresult r = p_cuMemHostUnregister(ptr);
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
    if (!pDevice) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemHostGetDevicePointer) return cudaErrorInitializationError;

    CUdeviceptr dev_ptr = 0;
    CUresult r = p_cuMemHostGetDevicePointer(&dev_ptr, pHost, flags);
    if (r == CUDA_SUCCESS) {
        *pDevice = reinterpret_cast<void*>(dev_ptr);
    } else {
        *pDevice = nullptr;
    }
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaFreeHost(void* ptr) {
    if (!ptr) return cudaSuccess;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuMemFreeHost) return cudaErrorInitializationError;

    CUresult r = p_cuMemFreeHost(ptr);
    cudaError_t e = cu_to_cuda_error(r);

    // Untrack
    {
        std::lock_guard lock(g_host_alloc_mutex);
        g_host_alloced_ptrs.erase(reinterpret_cast<std::uintptr_t>(ptr));
    }

    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr) {
    if (!attributes) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;

    // Initialize to defaults
    attributes->type = cudaMemoryTypeHost;
    attributes->device = -1;
    attributes->devicePointer = nullptr;
    attributes->hostPointer = const_cast<void*>(ptr);
    attributes->isManaged = 0;

    CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(ptr);

    // Check our tracked pointers first
    {
        std::lock_guard lock(g_ptr_mutex);
        auto it = g_tracked_ptrs.find(addr);
        if (it != g_tracked_ptrs.end()) {
            attributes->type = it->second.memory_type;
            attributes->device = it->second.device;
            if (it->second.memory_type == cudaMemoryTypeDevice) {
                attributes->devicePointer = const_cast<void*>(ptr);
            }
            if (it->second.host_ptr) {
                attributes->hostPointer = it->second.host_ptr;
            }
            return cudaSuccess;
        }
    }

    // Not found in our tracking — assume host memory
    return cudaSuccess;
}

// ──────────────────────────────────────────────────────────────────────────
// Device properties
// ──────────────────────────────────────────────────────────────────────────

// Helper to query a single device attribute via the driver API
static int query_attr(int cu_attr, int device_ordinal) {
    if (!p_cuDeviceGetAttribute || !p_cuDeviceGet) return 0;

    CUdevice cu_dev;
    CUresult r = p_cuDeviceGet(&cu_dev, device_ordinal);
    if (r != CUDA_SUCCESS) return 0;

    int value = 0;
    p_cuDeviceGetAttribute(&value, cu_attr, cu_dev);
    return value;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp* prop, int device) {
    if (!prop) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;

    // Zero out the struct
    std::memset(prop, 0, sizeof(struct cudaDeviceProp));

    // Ensure cuInit has been called
    if (!g_init_called.load(std::memory_order_acquire)) {
        if (p_cuInit) {
            p_cuInit(0);
            g_init_called.store(true, std::memory_order_release);
        }
    }

    // Get device name
    if (p_cuDeviceGetName && p_cuDeviceGet) {
        CUdevice cu_dev;
        if (p_cuDeviceGet(&cu_dev, device) == CUDA_SUCCESS) {
            p_cuDeviceGetName(prop->name, 256, cu_dev);
        }
    }

    // Query compute capability
    prop->major = query_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    prop->minor = query_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

    // Query total global memory
    if (p_cuDeviceTotalMem && p_cuDeviceGet) {
        CUdevice cu_dev;
        if (p_cuDeviceGet(&cu_dev, device) == CUDA_SUCCESS) {
            p_cuDeviceTotalMem(&prop->totalGlobalMem, cu_dev);
        }
    }

    // Fill in all the properties by querying driver attributes
    prop->maxThreadsPerBlock       = query_attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    prop->maxThreadsDim[0]         = query_attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
    prop->maxThreadsDim[1]         = query_attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
    prop->maxThreadsDim[2]         = query_attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device);
    prop->maxGridSize[0]           = query_attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
    prop->maxGridSize[1]           = query_attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
    prop->maxGridSize[2]           = query_attr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
    prop->sharedMemPerBlock        = query_attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
    prop->totalConstMem            = query_attr(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
    prop->warpSize                 = query_attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    prop->regsPerBlock             = query_attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);
    prop->clockRate                = query_attr(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
    prop->multiProcessorCount      = query_attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    prop->kernelExecTimeoutEnabled = query_attr(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device);
    prop->integrated               = query_attr(CU_DEVICE_ATTRIBUTE_INTEGRATED, device);
    prop->canMapHostMemory         = query_attr(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device);
    prop->computeMode              = query_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device);
    prop->concurrentKernels        = query_attr(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device);
    prop->ECCEnabled               = query_attr(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device);
    prop->pciBusID                 = query_attr(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device);
    prop->pciDeviceID              = query_attr(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device);
    prop->tccDriver                = query_attr(CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device);
    prop->memoryClockRate          = query_attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
    prop->memoryBusWidth           = query_attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
    prop->l2CacheSize              = query_attr(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device);
    prop->maxThreadsPerMultiProcessor = query_attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
    prop->computePreemptionSupported = query_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, device);
    prop->managedMemory            = query_attr(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device);
    prop->isMultiGpuBoard          = query_attr(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, device);
    prop->multiGpuBoardGroupID     = query_attr(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, device);
    prop->canUseHostPointerForRegisteredMem = query_attr(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, device);
    prop->pageableMemoryAccess     = query_attr(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device);
    prop->pageableMemoryAccessUsesHostPageTables = query_attr(CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device);
    prop->cooperativeLaunch        = query_attr(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device);
    prop->cooperativeMultiDeviceLaunch = query_attr(CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, device);
    prop->sharedMemPerMultiprocessor = query_attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device);
    prop->streamPrioritiesSupported = query_attr(CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, device);
    prop->globalL1CacheSupported   = query_attr(CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, device);
    prop->localL1CacheSupported    = query_attr(CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, device);
    prop->asyncEngineCount         = query_attr(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device);
    prop->unifiedAddressing        = query_attr(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device);
    prop->persistingL2CacheMaxSize = query_attr(CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, device);
    prop->accessPolicyMaxWindowSize = query_attr(CU_DEVICE_ATTRIBUTE_ACCESS_POLICY_MAX_WINDOW_SIZE, device);
    prop->reservedSharedMemPerBlock = query_attr(CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, device);
    prop->hostNativeAtomicSupported = query_attr(CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device);
    prop->singleToDoublePrecisionPerfRatio = query_attr(CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, device);
    prop->concurrentManagedAccess  = query_attr(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device);
    prop->directManagedMemAccessFromHost = query_attr(CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device);
    prop->maxBlocksPerMultiProcessor = query_attr(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, device);

    // Set defaults for fields not directly queryable
    prop->deviceOverlap             = 1;  // All modern GPUs support overlap
    prop->textureAlignment          = 512;
    prop->texturePitchAlignment     = 512;
    prop->memPitch                  = ~0UL;
    prop->maxTexture1D              = 131072;
    prop->maxTexture1DMipmap        = 16384;
    prop->maxTexture1DLinear        = 134217728;
    prop->maxTexture2D[0]           = 131072;
    prop->maxTexture2D[1]           = 65536;
    prop->maxTexture2DMipmap[0]     = 16384;
    prop->maxTexture2DMipmap[1]     = 16384;
    prop->maxTexture2DLinear[0]     = 131072;
    prop->maxTexture2DLinear[1]     = 65000;
    prop->maxTexture2DLinear[2]     = 65536;
    prop->maxTexture2DGather[0]     = 32768;
    prop->maxTexture2DGather[1]     = 32768;
    prop->maxTexture3D[0]           = 16384;
    prop->maxTexture3D[1]           = 16384;
    prop->maxTexture3D[2]           = 16384;
    prop->maxTexture3DAlt[0]        = 8192;
    prop->maxTexture3DAlt[1]        = 8192;
    prop->maxTexture3DAlt[2]        = 32768;
    prop->maxTextureCubemap         = 32768;
    prop->maxTexture1DLayered[0]    = 16384;
    prop->maxTexture1DLayered[1]    = 2048;
    prop->maxTexture2DLayered[0]    = 16384;
    prop->maxTexture2DLayered[1]    = 16384;
    prop->maxTexture2DLayered[2]    = 2048;
    prop->maxTextureCubemapLayered[0] = 16384;
    prop->maxTextureCubemapLayered[1] = 2048;
    prop->maxSurface1D              = 32768;
    prop->maxSurface2D[0]           = 131072;
    prop->maxSurface2D[1]           = 65536;
    prop->maxSurface3D[0]           = 16384;
    prop->maxSurface3D[1]           = 16384;
    prop->maxSurface3D[2]           = 16384;
    prop->maxSurface1DLayered[0]    = 16384;
    prop->maxSurface1DLayered[1]    = 2048;
    prop->maxSurface2DLayered[0]    = 16384;
    prop->maxSurface2DLayered[1]    = 16384;
    prop->maxSurface2DLayered[2]    = 2048;
    prop->maxSurfaceCubemap         = 32768;
    prop->maxSurfaceCubemapLayered[0] = 16384;
    prop->maxSurfaceCubemapLayered[1] = 2048;
    prop->surfaceAlignment          = 512;
    prop->regsPerMultiprocessor     = 65536;

    return cudaSuccess;
}

// Alias for the default version
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {
    return cudaGetDeviceProperties_v2(prop, device);
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device) {
    if (!value) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;

    int cu_attr = map_device_attr(attr);
    if (cu_attr < 0) {
        // Unknown attribute — return a reasonable default
        *value = 0;
        return cudaSuccess;
    }

    if (!p_cuDeviceGetAttribute || !p_cuDeviceGet) return cudaErrorInitializationError;

    CUdevice cu_dev;
    CUresult r = p_cuDeviceGet(&cu_dev, device);
    if (r != CUDA_SUCCESS) {
        set_last_error(cu_to_cuda_error(r));
        return cu_to_cuda_error(r);
    }

    r = p_cuDeviceGetAttribute(value, cu_attr, cu_dev);
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (!leastPriority || !greatestPriority) return cudaErrorInvalidValue;
    // Return defaults: 0 = default priority, no range
    *leastPriority = 0;
    *greatestPriority = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device) {
    if (!memPool) return cudaErrorInvalidValue;
    (void)device;
    // Return a dummy pool handle
    *memPool = reinterpret_cast<cudaMemPool_t>(&g_dummy_pool);
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    (void)peerDevice;
    (void)flags;
    // No-op for remote GPU — return success
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    if (!canAccessPeer) return cudaErrorInvalidValue;
    (void)device;
    (void)peerDevice;
    // No peer access in zlink
    *canAccessPeer = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    if (!pciBusId || len <= 0) return cudaErrorInvalidValue;
    (void)device;
    // Return empty string — PCI bus ID is not meaningful for remote GPU
    pciBusId[0] = '\0';
    return cudaSuccess;
}

// ──────────────────────────────────────────────────────────────────────────
// Streams
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    if (!pStream) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuStreamCreate) return cudaErrorInitializationError;

    CUstream stream = nullptr;
    CUresult r = p_cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    if (r == CUDA_SUCCESS) {
        *pStream = static_cast<cudaStream_t>(stream);
    } else {
        *pStream = nullptr;
    }
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) {
    if (!pStream) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuStreamCreateWithPriority) return cudaErrorInitializationError;

    unsigned int cu_flags = (flags & 1) ? CU_STREAM_NON_BLOCKING : CU_STREAM_DEFAULT;
    CUstream stream = nullptr;
    CUresult r = p_cuStreamCreateWithPriority(&stream, cu_flags, priority);
    if (r == CUDA_SUCCESS) {
        *pStream = static_cast<cudaStream_t>(stream);
    } else {
        *pStream = nullptr;
    }
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!stream) return cudaSuccess;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuStreamDestroy) return cudaErrorInitializationError;

    CUresult r = p_cuStreamDestroy(static_cast<CUstream>(stream));
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuStreamSynchronize) return cudaErrorInitializationError;

    CUresult r = p_cuStreamSynchronize(static_cast<CUstream>(stream));
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    (void)stream;
    // Always report success — in zlink everything is synchronous at the RPC level
    return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus) {
    (void)stream;
    if (pCaptureStatus) *pCaptureStatus = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, void* callback, void* userData, int flags) {
    (void)stream; (void)callback; (void)userData; (void)flags;
    // Not supported — stream callbacks cannot be forwarded over the network
    set_last_error(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode) {
    (void)stream; (void)mode;
    // Graph capture not supported
    set_last_error(cudaErrorStreamCaptureUnsupported);
    return cudaErrorStreamCaptureUnsupported;
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    (void)stream;
    if (pGraph) *pGraph = nullptr;
    set_last_error(cudaErrorStreamCaptureUnsupported);
    return cudaErrorStreamCaptureUnsupported;
}

// ──────────────────────────────────────────────────────────────────────────
// Events
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaEventCreate(cudaEvent_t* pEvent) {
    if (!pEvent) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuEventCreate) return cudaErrorInitializationError;

    CUevent event = nullptr;
    CUresult r = p_cuEventCreate(&event, 0);  // Default: timing enabled
    if (r == CUDA_SUCCESS) {
        *pEvent = static_cast<cudaEvent_t>(event);
    } else {
        *pEvent = nullptr;
    }
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* pEvent, unsigned int flags) {
    if (!pEvent) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuEventCreate) return cudaErrorInitializationError;

    unsigned int cu_flags = 0;
    if (flags & 0x2)  cu_flags |= CU_EVENT_DISABLE_TIMING;  // cudaEventDisableTiming
    if (flags & 0x4)  cu_flags |= CU_EVENT_INTERPROCESS;     // cudaEventInterprocess

    CUevent event = nullptr;
    CUresult r = p_cuEventCreate(&event, cu_flags);
    if (r == CUDA_SUCCESS) {
        *pEvent = static_cast<cudaEvent_t>(event);
    } else {
        *pEvent = nullptr;
    }
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (!event) return cudaSuccess;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuEventDestroy) return cudaErrorInitializationError;

    CUresult r = p_cuEventDestroy(static_cast<CUevent>(event));
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuEventRecord) return cudaErrorInitializationError;

    CUresult r = p_cuEventRecord(static_cast<CUevent>(event), static_cast<CUstream>(stream));
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuEventSynchronize) return cudaErrorInitializationError;

    CUresult r = p_cuEventSynchronize(static_cast<CUevent>(event));
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    (void)event;
    // Always report not ready — we can't reliably query remote events
    return cudaErrorNotReady;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) return cudaErrorInvalidValue;
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuEventElapsedTime) return cudaErrorInitializationError;

    CUresult r = p_cuEventElapsedTime(ms, static_cast<CUevent>(start), static_cast<CUevent>(end));
    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

// ──────────────────────────────────────────────────────────────────────────
// Kernel launch
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaLaunchKernel(const void* func,
                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                              unsigned int sharedMemBytes, cudaStream_t stream,
                              void** kernelParams, void** extra) {
    if (!ensure_driver_loaded()) return cudaErrorInitializationError;
    if (!p_cuLaunchKernel) return cudaErrorInitializationError;

    CUresult r = p_cuLaunchKernel(
        reinterpret_cast<CUfunction>(const_cast<void*>(func)),
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        static_cast<CUstream>(stream),
        kernelParams, extra);

    cudaError_t e = cu_to_cuda_error(r);
    if (e != cudaSuccess) set_last_error(e);
    return e;
}

cudaError_t cudaLaunchKernelExC(const struct cudaLaunchConfig* config,
                                 const void* func,
                                 unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                 unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, cudaStream_t stream,
                                 void** kernelParams, void** extra) {
    // Ignore extra config — just forward to cuLaunchKernel
    (void)config;
    return cudaLaunchKernel(func, gridDimX, gridDimY, gridDimZ,
                            blockDimX, blockDimY, blockDimZ,
                            sharedMemBytes, stream, kernelParams, extra);
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func) {
    if (!attr) return cudaErrorInvalidValue;
    (void)func;

    // Return reasonable defaults
    attr->constSizeBytes = 0;
    attr->localSizeBytes = 0;
    attr->maxThreadsPerBlock = 1024;
    attr->numRegs = 32;
    attr->ptxVersion = 70;
    attr->binaryVersion = 70;
    attr->cachePolicyCA = 0;
    attr->maxDynamicSharedSizeBytes = 0;
    attr->preferredShmemCarveout = 0;
    return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttr attr, int value) {
    (void)func; (void)attr; (void)value;
    // No-op — return success
    return cudaSuccess;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    if (!numBlocks) return cudaErrorInvalidValue;
    (void)func; (void)dynamicSMemSize; (void)flags;

    // Return a default estimate based on SM resources
    // Assume 2048 max threads per SM, 32 warps
    if (blockSize <= 0) {
        *numBlocks = 0;
        return cudaErrorInvalidValue;
    }

    int warp_size = 32;
    int warps_per_block = (blockSize + warp_size - 1) / warp_size;
    int max_warps = 64; // Typical for modern GPUs
    *numBlocks = max_warps / warps_per_block;
    if (*numBlocks < 1) *numBlocks = 1;
    return cudaSuccess;
}

// ──────────────────────────────────────────────────────────────────────────
// IPC (not supported)
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaIpcGetMemHandle(struct cudaIpcMemHandle_st* handle, void* devPtr) {
    (void)handle; (void)devPtr;
    set_last_error(cudaErrorIpcNotSupported);
    return cudaErrorIpcNotSupported;
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, struct cudaIpcMemHandle_st handle, unsigned int flags) {
    (void)handle; (void)flags;
    if (devPtr) *devPtr = nullptr;
    set_last_error(cudaErrorIpcNotSupported);
    return cudaErrorIpcNotSupported;
}

cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    (void)devPtr;
    set_last_error(cudaErrorIpcNotSupported);
    return cudaErrorIpcNotSupported;
}

cudaError_t cudaIpcGetEventHandle(struct cudaIpcEventHandle_st* handle, cudaEvent_t event) {
    (void)handle; (void)event;
    set_last_error(cudaErrorIpcNotSupported);
    return cudaErrorIpcNotSupported;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, struct cudaIpcEventHandle_st handle) {
    (void)handle;
    if (event) *event = nullptr;
    set_last_error(cudaErrorIpcNotSupported);
    return cudaErrorIpcNotSupported;
}

// ──────────────────────────────────────────────────────────────────────────
// Memory pools (stubs)
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value) {
    (void)memPool; (void)attr;
    if (value) std::memset(value, 0, sizeof(void*));
    return cudaSuccess;
}

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool, const void* desc, size_t count) {
    (void)memPool; (void)desc; (void)count;
    return cudaSuccess;
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value) {
    (void)memPool; (void)attr; (void)value;
    return cudaSuccess;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
    (void)memPool; (void)minBytesToKeep;
    return cudaSuccess;
}

// ──────────────────────────────────────────────────────────────────────────
// Graphs (not supported)
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    (void)graphExec; (void)stream;
    set_last_error(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    (void)graph;
    return cudaSuccess;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, void* nodes, size_t* numNodes) {
    (void)graph; (void)nodes;
    if (numNodes) *numNodes = 0;
    return cudaSuccess;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                  void* pErrorNode, char* pLogBuffer, size_t bufferSize) {
    (void)graph; (void)pErrorNode; (void)pLogBuffer; (void)bufferSize;
    if (pGraphExec) *pGraphExec = nullptr;
    set_last_error(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                           unsigned long long flags) {
    (void)graph; (void)flags;
    if (pGraphExec) *pGraphExec = nullptr;
    set_last_error(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    (void)graphExec;
    return cudaSuccess;
}

cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags) {
    (void)graph; (void)path; (void)flags;
    set_last_error(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

// ──────────────────────────────────────────────────────────────────────────
// Peer (not supported / stubs)
// ──────────────────────────────────────────────────────────────────────────

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                                 size_t count, cudaStream_t stream) {
    (void)dst; (void)dstDevice; (void)src; (void)srcDevice; (void)count; (void)stream;
    set_last_error(cudaErrorNotSupported);
    return cudaErrorNotSupported;
}

} // extern "C"

// cudaGetErrorString is provided above with the correct const char* return type.

// ══════════════════════════════════════════════════════════════════════════
// LIBRARY CONSTRUCTOR / DESTRUCTOR
// ══════════════════════════════════════════════════════════════════════════

__attribute__((constructor))
static void cudart_shim_init() {
    std::fprintf(stderr, "[cudart_shim] libcudart.so.12 (zlink runtime shim) loaded\n");
    std::fprintf(stderr, "[cudart_shim] Will forward cuda* calls to libcuda.so.1 (zlink driver shim)\n");
}

__attribute__((destructor))
static void cudart_shim_fini() {
    // Release all retained primary contexts
    {
        std::lock_guard lock(g_ctx_mutex);
        for (auto& [dev, ctx] : g_primary_contexts) {
            if (p_cuDevicePrimaryCtxRelease) {
                CUdevice cu_dev;
                if (p_cuDeviceGet && p_cuDeviceGet(&cu_dev, dev) == CUDA_SUCCESS) {
                    p_cuDevicePrimaryCtxRelease(cu_dev);
                }
            }
        }
        g_primary_contexts.clear();
    }

    // Close the driver library
    if (g_driver_lib) {
        dlclose(g_driver_lib);
        g_driver_lib = nullptr;
    }

    std::fprintf(stderr, "[cudart_shim] Unloaded\n");
}

// ── Additional stub functions required by PyTorch ─────────────────────

extern "C" {

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* userData) {
    return static_cast<cudaError_t>(999); // cudaErrorNotSupported
}

cudaError_t cudaProfilerStart() {
    return cudaSuccess;
}

cudaError_t cudaProfilerStop() {
    return cudaSuccess;
}

cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream, int* captureStatus_out,
                                                    unsigned int* driverGroupId_out,
                                                    unsigned int* numCapturedGraphs_out,
                                                    void** capturedGraphs_out) {
    if (captureStatus_out) *captureStatus_out = 0; // cudaStreamCaptureStatusNone
    if (driverGroupId_out) *driverGroupId_out = 0;
    if (numCapturedGraphs_out) *numCapturedGraphs_out = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority) {
    if (priority) *priority = 0;
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    ensure_driver_loaded();
    if (!p_cuEventSynchronize) return static_cast<cudaError_t>(46); // cudaErrorNotReady
    return static_cast<cudaError_t>(p_cuEventSynchronize(reinterpret_cast<CUevent>(event)));
}

cudaError_t cudaThreadExchangeStreamCaptureMode(int* mode) {
    if (mode) *mode = 0; // cudaStreamCaptureModeGlobal
    return cudaSuccess;
}

// ── CUDA fatbin registration stubs ────────────────────────────────────
// PyTorch kernels are compiled into fatbinaries that are registered at
// load time via __cudaRegister* functions. Since our shim doesn't actually
// load fatbins (the server does), these are no-op stubs.

void** __cudaRegisterFatBinary(void* fatCubin) {
    static void* handle = reinterpret_cast<void*>(0x1);
    return &handle;
}

void __cudaRegisterFatBinaryEnd(void** fatCubinHandle) {
    // No-op: marks the end of fatbin registration
}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
    // No-op
}

void __cudaRegisterVar(void** fatCubinHandle, char* hostVar, char* deviceAddress,
                       const char* deviceName, int ext, size_t size, int constant, int global) {
    // No-op: variable registration happens on the server side
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun,
                             const char* deviceName, int thread_limit, uint3* tid, uint3* bid,
                             dim3* bDim, dim3* gDim, int* wSize) {
    // No-op: function registration happens on the server side
}

void __cudaRegisterShared(void** fatCubinHandle, void** devicePtr) {
    // No-op
}

void __cudaRegisterSharedVar(void** fatCubinHandle, void** devicePtr, size_t size, size_t alignment, int access) {
    // No-op
}

void __cudaRegisterTexture(void** fatCubinHandle, const struct textureReference* hostVar, void** deviceAddress, const char* deviceName, int dim, int norm, int ext) {
    // No-op
}

void __cudaRegisterSurface(void** fatCubinHandle, const struct surfaceReference* hostVar, void** deviceAddress, const char* deviceName, int dim, int ext) {
    // No-op
}

// ── CUDA launch configuration stubs ──────────────────────────────────
// PyTorch uses __cudaPushCallConfiguration / __cudaPopCallConfiguration
// to pass kernel launch parameters (grid size, block size, shared mem, stream)
// before calling cudaLaunchKernel. These are used by the <<<>>> syntax.
// Since we intercept cudaLaunchKernel directly, these just store/retrieve
// the configuration for the next launch.

static thread_local struct {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    bool valid;
} g_pending_launch = {{1,1,1}, {1,1,1}, 0, nullptr, false};

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
    g_pending_launch.gridDim = gridDim;
    g_pending_launch.blockDim = blockDim;
    g_pending_launch.sharedMem = sharedMem;
    g_pending_launch.stream = stream;
    g_pending_launch.valid = true;
    return 0; // cudaSuccess
}

unsigned __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, cudaStream_t* stream) {
    if (g_pending_launch.valid) {
        *gridDim = g_pending_launch.gridDim;
        *blockDim = g_pending_launch.blockDim;
        *sharedMem = g_pending_launch.sharedMem;
        *stream = g_pending_launch.stream;
        g_pending_launch.valid = false;
    } else {
        *gridDim = dim3(1);
        *blockDim = dim3(1);
        *sharedMem = 0;
        *stream = nullptr;
    }
    return 0; // cudaSuccess
}

// CUDA fatbin management (for module loading)
void* __cudaRegisterFatBinaryEx(void* fatCubin, void* hostBinary, size_t hostBinarySize, void* p) {
    static void* handle = reinterpret_cast<void*>(0x1);
    return &handle;
}

void** __cudaRegisterManagedVar(void** fatCubinHandle, void* hostVar, void** deviceAddress, const char* deviceName, size_t size, unsigned int alignment) {
    static void* handle = reinterpret_cast<void*>(0x1);
    return &handle;
}

} // extern "C"
