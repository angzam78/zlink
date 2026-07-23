#pragma once
// zlink/cuda_dep_spec.hpp — CUDA API dependency categorization for pipelining
//
// THE CORE DESIGN PRINCIPLE:
//   Simple pipelining that always works > complex pipelining that pipelines
//   everything but is fragile or inefficient. Correctness and reliability
//   come first. Performance is the goal, but not at the cost of correctness.
//
// THE THREE CATEGORIES:
//
//   barrier   — Must flush pipeline and get return value before next call.
//               Used ONLY when client code branches on the return value
//               in a way that affects subsequent GPU operations.
//               Examples: cuInit, cuDeviceGetCount, cuDeviceGetAttribute
//
//   enqueued  — Can batch into pipeline. Don't need return value immediately.
//               This is the DEFAULT for most CUDA functions.
//               Examples: cuMemcpyHtoD, cuLaunchKernel, cuCtxSynchronize,
//                         cuStreamSynchronize, cuEventRecord, cuMemFree
//
//   readback  — Must flush pipeline + get data back from server.
//               Used when copying GPU data back to host.
//               Examples: cuMemcpyDtoH, cuMemcpyDtoHAsync
//
// VIRTUAL HANDLES eliminate most barriers:
//   Before VH:  cuMemAlloc → BARRIER (must wait for dev_ptr)
//   After VH:   cuMemAlloc → enqueue_produces_handle → VH(0) → ENQUEUED!
//
//   With virtual handles, the only barriers are functions where the client
//   needs a return value to make a decision:
//   - cuInit (need to check if CUDA is available before proceeding)
//   - cuDriverGetVersion (need version string)
//   - cuDeviceGetCount (need count to decide which GPU to use)
//   - cuDeviceGetName (need name string)
//   - cuDeviceTotalMem (need value for alloc planning)
//   - cuDeviceGetAttribute (need value for occupancy calculations, etc.)
//   - cuMemGetInfo (need free/total for alloc strategy)
//   - cuEventElapsedTime (need the float value for profiling)
//
//   Everything else pipelines. Even error codes are deferred.
//
// HANDLE PRODUCERS (the only functions needing enqueue_produces_handle):
//   ~30 functions out of ~2000 CUDA API functions.
//   Pattern: any function whose return struct has a uint64_t handle field.
//
//   cuMemAlloc, cuMemAllocManaged, cuMemHostAlloc → dev_ptr/host_ptr
//   cuCtxCreate → ctx_handle
//   cuModuleLoadData, cuModuleLoad → module_handle
//   cuModuleGetFunction → func_handle
//   cuModuleGetGlobal → global_ptr
//   cuStreamCreate → stream_handle
//   cuEventCreate → event_handle
//
// HANDLE CONSUMERS (most of the API):
//   Just take handles as input → g_vhandles.translate() on the server.
//   These are simple enqueue<Index>(handle_vh, ...) calls.
//   The server auto-translates VH → real handle.
//
// ─────────────────────────────────────────────────────────────────────────
// FULL CATEGORIZATION (PyTorch-relevant subset of CUDA Driver API)
// ─────────────────────────────────────────────────────────────────────────
//
// Function                          | Category  | Handle?  | Notes
// ──────────────────────────────────┼───────────┼──────────┼───────────────
// cuInit                            | barrier   | -        | First call, check availability
// cuDeviceGet                       | barrier   | -        | Need device ordinal
// cuDeviceGetCount                  | barrier   | -        | Need count for iteration
// cuDeviceGetName                   | barrier   | -        | Need string back
// cuDeviceTotalMem                  | barrier   | -        | Need value for alloc planning
// cuDeviceGetAttribute              | barrier   | -        | Need value for occupancy calc
// cuCtxCreate                       | enqueued  | PRODUCES | Returns VH for context
// cuCtxDestroy                      | enqueued  | consumes | Takes VH context
// cuCtxSetCurrent                   | enqueued  | consumes | Takes VH context
// cuCtxGetCurrent                   | barrier   | PRODUCES | Need current ctx
// cuCtxSynchronize                  | enqueued  | -        | Just a sync point
// cuMemAlloc                        | enqueued  | PRODUCES | Returns VH for dev_ptr ★
// cuMemAllocManaged                 | enqueued  | PRODUCES | Returns VH for dev_ptr
// cuMemFree                         | enqueued  | consumes | Takes VH dev_ptr
// cuMemFreeHost                     | enqueued  | consumes | Takes VH host_ptr
// cuMemHostAlloc                    | enqueued  | PRODUCES | Returns VH for host_ptr
// cuMemHostRegister                 | enqueued  | consumes | Takes host pointer
// cuMemHostUnregister               | enqueued  | consumes | Takes host pointer
// cuMemcpyHtoD                      | enqueued  | consumes | + inline host_sync ★
// cuMemcpyDtoH                      | readback  | consumes | + inline host_read ★
// cuMemcpyDtoD                      | enqueued  | consumes | Both dev_ptrs are VHs
// cuMemcpyHtoDAsync                 | enqueued  | consumes | + inline host_sync, VH stream
// cuMemcpyDtoHAsync                 | readback  | consumes | + inline host_read, VH stream
// cuModuleLoadData                  | enqueued  | PRODUCES | + inline host_sync for image
// cuModuleLoad                      | enqueued  | PRODUCES | Filename sent as string
// cuModuleUnload                    | enqueued  | consumes | Takes VH module
// cuModuleGetFunction               | enqueued  | PROD+CON | Consumes VH module, produces VH func
// cuModuleGetGlobal                 | enqueued  | PROD+CON | Consumes VH module, produces VH global
// cuLaunchKernel                    | enqueued  | consumes | VH func, VH stream, VH args ★
// cuStreamCreate                    | enqueued  | PRODUCES | Returns VH for stream
// cuStreamDestroy                   | enqueued  | consumes | Takes VH stream
// cuStreamSynchronize               | enqueued  | consumes | Takes VH stream
// cuEventCreate                     | enqueued  | PRODUCES | Returns VH for event
// cuEventDestroy                    | enqueued  | consumes | Takes VH event
// cuEventRecord                     | enqueued  | consumes | VH event + VH stream
// cuEventSynchronize                | enqueued  | consumes | Takes VH event
// cuEventElapsedTime                | barrier   | consumes | Need float back for profiling
// cuOccupancyMaxPotentialBlockSize  | barrier   | -        | Need values for launch config
// cuMemGetInfo                      | barrier   | -        | Need values for alloc strategy
//
// KEY INSIGHT FOR SCALABILITY:
//   Adding a new CUDA function requires exactly 3 things:
//   1. Declare the RPC function signature (in the API namespace)
//   2. Add zpp::bits::bind entry (with next index number)
//   3. Categorize it: barrier/enqueued/readback + produces/consumes handles
//
//   The categorization is OBVIOUS from the function signature:
//   - Returns a handle field → enqueue_produces_handle
//   - Takes handle parameters → enqueue with g_vhandles.translate() on server
//   - Returns host data → readback
//   - Client code needs the return value to decide next steps → barrier
//   - Otherwise → enqueued (the safe default)
//
//   If in doubt, make it a barrier. That's always correct.
//   Optimize to enqueued only when you're sure it's safe.
//
// TYPICAL PYTORCH WORKLOAD PIPELINE:
//
//   cuInit(0)                    → barrier (1 RT)
//   cuDeviceGetCount()           → barrier (1 RT)
//   cuDeviceGetAttribute(...)    → barrier (1 RT)
//   cuCtxCreate(...)             → enqueued: VH(0) = ctx
//   cuMemAlloc(N)                → enqueued: VH(1) = input_buf
//   cuMemAlloc(M)                → enqueued: VH(2) = output_buf
//   cuModuleLoadData(...)        → enqueued: VH(3) = module  (+ inline sync)
//   cuModuleGetFunction(VH(3))   → enqueued: VH(4) = kernel_func
//   cuMemcpyHtoD(VH(1), ...)     → enqueued (+ inline sync)
//   cuLaunchKernel(VH(4), ...)   → enqueued
//   cuMemcpyDtoH(out, VH(2),...) → READBACK: flush! (1 RT for ALL above)
//
//   Total: 3 barrier RTs + 1 pipeline batch RT = 4 round-trips
//   Without VH: every alloc/create is a barrier → 10+ round-trips

namespace zlink {

// ── Dependency category enum (also defined in cuda_pipeline.hpp — same values) ──
// Note: call_dep is defined in cuda_pipeline.hpp. This header does not
// re-define it to avoid ODR violations.

// ── Handle production/consumption annotation ──────────────────────────
enum class handle_role : std::uint8_t {
    none,           // No handle involvement
    produces,       // This call produces a new handle (enqueue_produces_handle)
    consumes,       // This call takes handle(s) as input (server auto-translates)
    produces_and_consumes,  // Both (e.g., cuModuleGetFunction)
};

// ── Per-function dependency metadata ──────────────────────────────────
struct func_dep_info {
    int           func_index;     // RPC binding index
    zlink::call_dep dependency;   // barrier / enqueued / readback
    handle_role   handles;        // none / produces / consumes / both
    int           handle_field;   // 0-indexed field in return struct (if produces)
    bool          needs_host_sync; // True if host data must be synced before call
    bool          needs_host_read; // True if server data must be read back after call
    const char*   name;           // Human-readable function name
};

} // namespace zlink
