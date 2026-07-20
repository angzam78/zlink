# RPC Framework

zlink's RPC framework is built on zpp_bits, a C++20 header-only library that
provides binary serialization and RPC with zero code generation.

## zpp_bits Overview

zpp_bits compiles function signatures into serialization/deserialization code at
compile time. The `zpp::bits::rpc<>` template takes a list of `zpp::bits::bind<>`
entries, each mapping a function pointer to an integer index.

```cpp
using cuda_test_rpc = zpp::bits::rpc<
    zpp::bits::bind<&cuda_rpc_api::cuda_init,           0>,
    zpp::bits::bind<&cuda_rpc_api::get_device_count,    1>,
    zpp::bits::bind<&cuda_rpc_api::ctx_create,          4>,
    zpp::bits::bind<&cuda_rpc_api::mem_alloc,           7>,
    // ...
>;
```

The index is part of the serialized request so the server knows which function
to dispatch. Client and server must use the same binding indices.

## API Declaration Pattern

Both client and server share the same API namespace and binding. The pattern in
zlink is:

### 1. Declare return types and function signatures

```cpp
namespace cuda_rpc_api {

struct AllocRet { int32_t result; std::uint64_t dev_ptr; };
AllocRet mem_alloc(std::uint64_t bytesize);

struct FreeRet { int32_t result; };
FreeRet mem_free(std::uint64_t dev_ptr);

} // namespace cuda_rpc_api
```

### 2. Define the RPC binding

```cpp
using cuda_test_rpc = zpp::bits::rpc<
    zpp::bits::bind<&cuda_rpc_api::mem_alloc, 7>,
    zpp::bits::bind<&cuda_rpc_api::mem_free,  8>,
>;
```

### 3. Server: implement the functions

```cpp
// In the same namespace — zpp_bits calls these via the function pointer
AllocRet mem_alloc(std::uint64_t bytesize) {
    CUdeviceptr ptr = 0;
    CUresult r = cuMemAlloc(&ptr, bytesize);
    return {static_cast<int32_t>(r), ptr};
}
```

### 4. Client: call through the pipeline

```cpp
zlink::cuda_pipeline<cuda_test_rpc> pipe(*tp);
auto r = pipe.call_barrier<7>(static_cast<uint64_t>(256));
// r.dev_ptr contains the device pointer
```

## Key Classes

### `rpc_client<RpcDef>` (`include/zlink/rpc.hpp`)

Typed synchronous RPC client. Uses `call<FuncIndex>(args...)` for individual calls.

```cpp
zlink::rpc_client<cuda_test_rpc> client(transport);
auto result = client.call<0>(0u);  // cuInit(0)
```

Internal flow:
1. Serialize request with fresh zpp_bits context
2. Send as `frame_type::request`
3. Receive `frame_type::response`
4. Deserialize response with fresh zpp_bits context

### `rpc_server<RpcDef>` (`include/zlink/rpc.hpp`)

Typed RPC server. Dispatches incoming requests to the bound functions.

```cpp
zlink::rpc_server<cuda_test_rpc> server(transport);
server.serve_forever();
```

### `pipeline_caller<RpcDef>` (`include/zlink/rpc.hpp`)

Simple pipeline batcher — queues calls and flushes as `pipeline_request`.
Predecessor to `cuda_pipeline`, kept for non-CUDA use cases.

```cpp
zlink::pipeline_caller<cuda_test_rpc> pipe(transport);
pipe.push<7>(static_cast<uint64_t>(256));
pipe.push<8>(dev_ptr);
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
