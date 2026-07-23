# Wire Protocol

This document describes zlink's wire protocol as implemented in
`tcp_transport.cpp`, `config.hpp`, and `cuda_pipeline.hpp`.

## Frame Format

Every frame on the wire has a fixed 9-byte header followed by a variable-length
payload:

```
┌──────────────────┬──────────────────┬───────────────┬──────────────────────┐
│ Length (4 bytes)  │ Call ID (4 bytes)│ Type (1 byte) │ Payload (N bytes)     │
│ Big-endian uint32 │ Big-endian uint32│ uint8         │ zpp_bits / raw data  │
└──────────────────┴──────────────────┴───────────────┴──────────────────────┘
```

- **Length**: Total bytes after the length field itself (i.e., `4 + 1 + N`).
  The minimum valid length is 5 (call_id + type, zero payload).
- **Call ID**: Matches requests to responses. For pipeline frames, this is
  typically 1.
- **Type**: One of the `frame_type` enum values.
- **Payload**: Frame-type-specific data.

Maximum frame size: 64 MiB (`max_frame_size` in `config.hpp`).

## Frame Types

### 0x01 — Request

A single synchronous RPC call. The payload is a zpp_bits-serialized request
containing the function ID and arguments.

```
Payload: [zpp_bits serialized: func_id + args...]
```

### 0x02 — Response

RPC response. The payload is a zpp_bits-serialized return value.

```
Payload: [zpp_bits serialized: return value]
```

### 0x04 — Pipeline Request

Multiple RPC calls batched into one frame, plus a handle manifest. Each call
is length-prefixed. The handle manifest at the end tells the server which
calls produce virtual handles.

```
Payload:
  [4B call_count]
  For each call:
    [4B request_len]
    [request_len bytes: zpp_bits serialized request]
  [handle_manifest: 4B entry_count + N × handle_manifest_entry]
```

Handle manifest entry format (16 bytes each):
```
  [4B call_index]   (0-indexed position in the call list)
  [4B virtual_id]   (virtual handle ID to assign)
  [4B return_field] (which field in return struct is the handle)
  [4B reserved]     (alignment padding)
```

The server processes this frame by:
1. Parsing the handle manifest (at the end of the payload)
2. Processing RPC calls in order, registering handles after each
   handle-producing call using the manifest mapping

### 0x05 — Pipeline Response

Responses for a pipeline request, in the same order as the calls.

```
Payload:
  [4B response_count]
  For each response:
    [4B response_len]
    [response_len bytes: zpp_bits serialized response]
```

### 0x10 — Memory Operation

Remote memory operation request. The payload starts with a `mem_request`
header followed by optional data.

```
mem_request struct (24 bytes):
  [1B op]           (mem_op enum: read/write/alloc/free/sync/invalidate/host_sync/host_read)
  [7B padding]      (alignment)
  [8B remote_addr]  (address in server's address space)
  [8B size]         (number of bytes)

For write/host_sync operations:
  [size bytes: data to write/sync]
```

### 0x11 — Memory Reply

Memory operation response. The payload starts with a `mem_response` header
followed by optional data.

```
mem_response struct (24 bytes):
  [4B status]       (error_code enum)
  [4B padding]      (alignment)
  [8B size]         (bytes actually read/written)
  [8B remote_addr]  (for alloc: the allocated address)

For read/host_read operations:
  [size bytes: data read from remote]
```

### 0xFF — Heartbeat

Keep-alive frame. No payload.

## Memory Operation Types

Defined in `memory.hpp`:

| Op | Value | Direction | Purpose |
|----|-------|-----------|---------|
| `read` | `0x01` | Client → Server | Read from remote memory |
| `write` | `0x02` | Client → Server | Write to remote memory |
| `alloc` | `0x03` | Client → Server | Allocate remote memory |
| `free_op` | `0x04` | Client → Server | Free remote memory |
| `sync` | `0x05` | Client → Server | Flush dirty / invalidate |
| `invalidate` | `0x06` | Server → Client | Invalidate cached pages |
| `host_sync` | `0x07` | Client → Server | Sync client host page to server |
| `host_read` | `0x08` | Client → Server | Read from mirrored client memory |

## TCP Transport Details

The `tcp_transport` implementation (in `tcp_transport.cpp`) uses:

- **TCP_NODELAY** — Disabled Nagle's algorithm on all sockets for minimal
  latency. This means every `send()` call results in an immediate packet on
  the wire.

- **Blocking I/O** — Both `send()` and `receive()` are blocking. The
  `receive()` method first reads the 4-byte length prefix, then reads the
  remaining bytes.

- **Thread-safe sends** — A `send_mutex_` protects concurrent sends from
  different threads. Receives are not mutex-protected (single receiver
  assumed).

- **SO_REUSEADDR** — Set on the listening socket so the server can restart
  without waiting for TIME_WAIT to expire.

- **Big-endian wire format** — All multi-byte integers in frame headers are
  transmitted in network byte order (big-endian).

## Connection Lifecycle

1. **Server** calls `listen(bind_addr, port)` → creates listening socket
2. **Server** calls `accept()` → blocks until a client connects
3. **Client** calls `connect(host, port)` → TCP handshake
4. Both sides send/receive frames using `send()` and `receive()`
5. Either side calls `close()` to terminate the connection

Currently, the server handles one client connection at a time. The
`server.hpp` framework has the structure for multiple connections (spawning
`connection_handler` threads), but the accept loop is not yet fully
implemented.

## Error Codes

Defined in `config.hpp`:

| Code | Value | Meaning |
|------|-------|---------|
| `ok` | 0 | Success |
| `unknown_function` | -1 | Function ID not recognized |
| `serialization` | -2 | zpp_bits serialization error |
| `transport` | -3 | Network transport error |
| `call_id_mismatch` | -4 | Response call_id doesn't match request |
| `server_error` | -5 | Server-side execution error |
| `pointer_not_found` | -6 | Pointer not in ptr_map |
| `timeout` | -7 | Operation timed out |
| `shutdown` | -8 | Server is shutting down |
