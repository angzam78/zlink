# RPC Framework

zlink's RPC framework is built on [zpp_bits](https://github.com/eyalz800/zpp_bits),
a C++20 header-only library that provides binary serialization and RPC with zero
code generation. All request/response framing, pipeline batch serialization, and
server dispatch in zlink use zpp_bits under the hood.

## zpp_bits Overview

zpp_bits compiles function signatures into serialization/deserialization code at
compile time. The `zpp::bits::rpc<>` template takes a list of `zpp::bits::bind<>`
entries, each mapping a function pointer to an integer index.

```cpp
using cuda_gen_rpc = zpp::bits::rpc<
    zpp::bits::bind<&init,                  func_index::init>,
    zpp::bits::bind<&device_get_count,      func_index::device_get_count>,
    zpp::bits::bind<&ctx_create,            func_index::ctx_create>,
    zpp::bits::bind<&mem_alloc,             func_index::mem_alloc>,
    // ... 30 functions total, indices 0-29
>;
```

The index is part of the serialized request so the server knows which function
to dispatch. Client and server must use the same binding indices. zlink uses
named constants in the `func_index` namespace (e.g., `func_index::mem_alloc = 12`)
rather than raw literals, so the binding is self-documenting.

## API Declaration Pattern

Both client and server share the same API namespace and binding. The pattern in
zlink is:

### 1. Declare return types and function signatures

```cpp
namespace cuda_gen {

struct MemAllocRet { int32_t result; uint64_t dptr; };
MemAllocRet mem_alloc(uint64_t bytesize);

struct MemFreeRet { int32_t result; };
MemFreeRet mem_free(uint64_t dptr);

} // namespace cuda_gen
```

### 2. Define the RPC binding

```cpp
using cuda_gen_rpc = zpp::bits::rpc<
    zpp::bits::bind<&mem_alloc, func_index::mem_alloc>,
    zpp::bits::bind<&mem_free,  func_index::mem_free>,
>;
```

### 3. Server: implement the functions

```cpp
// In the same namespace — zpp_bits calls these via the function pointer
MemAllocRet mem_alloc(uint64_t bytesize) {
    CUdeviceptr ptr = 0;
    CUresult r = cuMemAlloc(&ptr, bytesize);
    return {static_cast<int32_t>(r), ptr};
}
```

### 4. Client: call through the pipeline

```cpp
zlink::cuda_pipeline<cuda_gen_rpc> pipe(*tp);
auto r = pipe.call_barrier<cuda_gen::func_index::mem_alloc>(static_cast<uint64_t>(256));
// r.dptr contains the device pointer
```

## Key Classes

### `rpc_client<RpcDef>` (`include/zlink/rpc.hpp`)

Typed synchronous RPC client. Uses `call<FuncIndex>(args...)` for individual calls.

```cpp
zlink::rpc_client<cuda_gen_rpc> client(transport);
auto result = client.call<cuda_gen::func_index::init>(0u);  // cuInit(0)
```

Internal flow:
1. Serialize request with fresh zpp_bits context
2. Send as `frame_type::request`
3. Receive `frame_type::response`
4. Deserialize response with fresh zpp_bits context

### `rpc_server<RpcDef>` (`include/zlink/rpc.hpp`)

Typed RPC server. Dispatches incoming requests to the bound functions.

```cpp
zlink::rpc_server<cuda_gen_rpc> server(transport);
server.serve_forever();
```

### `pipeline_caller<RpcDef>` (`include/zlink/rpc.hpp`)

Simple pipeline batcher — queues calls and flushes as `pipeline_request`.
Predecessor to `cuda_pipeline`, kept for non-CUDA use cases.

```cpp
zlink::pipeline_caller<cuda_gen_rpc> pipe(transport);
pipe.push<cuda_gen::func_index::mem_alloc>(static_cast<uint64_t>(256));
pipe.push<cuda_gen::func_index::mem_free>(dev_ptr);
auto results = pipe.flush();
```

### `dynamic_rpc_server` (`include/zlink/rpc.hpp`)

Runtime function registration without compile-time API declarations. Register
handlers by function ID:

```cpp
dynamic_rpc_server server(transport);
server.register_handler(42, my_handler_fn);
```

## Pointer Map (`ptr_map.hpp`)

The `ptr_map` class manages bidirectional pointer mapping between client and
server address spaces. Used by the generic RPC layer for opaque handle remapping.

- `map(remote_ptr)` — Register a remote→local mapping, returns local pointer
- `to_remote(local_ptr)` — Look up remote pointer from local
- `to_local(remote_ptr)` — Look up local pointer from remote
- `translate_pointers(span)` — Batch translate embedded pointers

Supports shadow regions where local pointers are allocated from an mmap'd
address range, making them look like real addresses to the client application.

## Serialization Pattern

zlink uses the "fresh context per direction" pattern proven to avoid stale
archive state:

```cpp
// Serialize request
auto [req_data, req_in, req_out] = zpp::bits::data_in_out();
typename rpc_type::client req_client{req_in, req_out};
req_client.template request<FuncIndex>(args...).or_throw();

// ... network transport ...

// Deserialize response
auto [resp_data, resp_in, resp_out] = zpp::bits::data_in_out();
resp_data = std::move(response_data);
typename rpc_type::client resp_client{resp_in, resp_out};
auto result = resp_client.template response<FuncIndex>().or_throw();
```

## Important Notes

- **FuncIndex type must match exactly**: zpp_bits compares ID types via
  `std::same_as`, so `int(0) != size_t(0)`. Always use the same type as the
  binding index (integers by default).

- **Error handling**: zpp_bits uses `zpp::bits::failure(result)` to check for
  serialization/deserialization errors. Use `.or_throw()` to convert to
  exceptions.

- **Thread safety**: The base `rpc_client_base` supports concurrent async calls
  via call IDs and a pending call map with condition variables. The
  `cuda_pipeline` is NOT thread-safe — use one pipeline per thread.
