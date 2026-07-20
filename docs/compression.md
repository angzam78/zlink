# LZ4 Compression in Pipeline Frames

zlink uses LZ4 compression to reduce network transfer time for large data
payloads in pipeline_mem frames. Compression is applied per-entry, with
automatic threshold and ratio checks to avoid overhead on small or
incompressible data.

## Design Rationale

On a 1 Gbps network link, transferring 256 MB of tensor data takes ~2 seconds.
LZ4 compression at 3-4 GB/s can compress this to ~100 MB in ~70 ms, saving
over 1 second of transfer time. The compression overhead is negligible compared
to the network savings.

On 10 Gbps links, the benefit is smaller but still meaningful for multi-GB
transfers (model weights, large activation tensors).

## Implementation

Defined in `include/zlink/compress.hpp`. Uses the LZ4 C library directly.

### Compression

```cpp
compress_result compress(std::span<const std::byte> input);
```

Returns:
- `data` — compressed or raw bytes
- `comp_flag` — `comp_flag_raw` (0) or `comp_flag_lz4` (1)
- `original_size` — always the uncompressed size (needed for decompression)

Logic:
1. If input < 4 KB → return raw (overhead exceeds savings)
2. Try LZ4 compression via `LZ4_compress_default()`
3. If compression ratio ≥ 0.9 (less than 10% savings) → return raw
4. Otherwise → return compressed data

### Decompression

```cpp
std::vector<std::byte> decompress(
    std::span<const std::byte> input,
    std::uint8_t comp_flag,
    std::size_t original_size);
```

If `comp_flag` is `comp_flag_raw`, returns input as-is. If `comp_flag_lz4`,
decompresses using `LZ4_decompress_safe()` with the known `original_size`.

## Wire Format Changes

### Sync Entries (Client → Server)

```
Before: [8B addr][8B size][data...]
After:  [8B addr][8B original_size][1B comp_flag][4B data_size][data...]
```

- `original_size` — uncompressed size (always present, needed for LZ4 decompression)
- `comp_flag` — 0=raw, 1=LZ4
- `data_size` — number of wire bytes in the data field (compressed or raw)
- `data` — the actual bytes (compressed or raw depending on comp_flag)

The `data_size` field is essential for the server to know how many bytes to
read before the next entry. Without it, the server couldn't distinguish
compressed bytes from the start of the next sync entry.

### Read Response Entries (Server → Client)

```
Before: [4B data_len][data...]
After:  [4B total_len][1B comp_flag][data...]
  When comp_flag=1 (LZ4): data = [8B original_size][compressed_bytes]
  When comp_flag=0 (raw):  data = raw_bytes
```

`total_len` includes the 1-byte comp_flag plus the data field.

## Compression Performance

Benchmarks from the test suite:

| Data Pattern | Size | Compressed | Ratio | Notes |
|-------------|------|-----------|-------|-------|
| Zero tensor | 256 KB | 1,038 B | 0.004 | Very common in ML |
| Repetitive activations | 1 MB | 4,219 B | 0.004 | Common inference pattern |
| FP16-like weights | 4 MB | 361 KB | 0.086 | Moderate compression |
| Sequential floats | 400 KB | 400 KB | 1.0 | High entropy, no compression |
| Random data | 100 KB | 100 KB | 1.0 | Incompressible |
| PTX module code | ~250 KB | ~15 KB | 0.06 | Text-like, compresses well |

### When Compression Helps

- **cuMemcpyHtoD with zero/initialized tensors** — 100-250x compression
- **cuMemcpyDtoH with activation outputs** — high compression on sparse/repeated patterns
- **cuModuleLoadData with PTX/CUBIN** — 5-15x compression (text-like)
- **FP16 weight tensors** — moderate compression (8-12x) due to zero low bits

### When Compression Doesn't Help

- Small transfers (< 4 KB) — overhead exceeds savings
- Fully trained FP32 weight tensors — high entropy, no compression
- Random data — incompressible
- Already compressed data (JPEG, compressed model files)

The threshold and ratio checks handle all these cases automatically — the
compression code falls back to raw mode with zero overhead.

## Asymmetric Cost

The compression cost is asymmetric in zlink's favor:

| Direction | Compressor | Decompressor | Who has spare CPU? |
|-----------|-----------|-------------|-------------------|
| cuMemcpyHtoD (client→server) | Client | Server | Both |
| cuMemcpyDtoH (server→client) | Server | Client | Both |

LZ4 decompression is ~4-7 GB/s, so the server-side overhead for cuMemcpyHtoD
is negligible compared to the GPU operation. For cuMemcpyDtoH, the client has
plenty of CPU to decompress while waiting for the network.

## Configuration

```cpp
// Minimum size to attempt compression
inline constexpr std::size_t compress_threshold = 4096;  // 4 KB

// Minimum compression ratio to use compressed version
inline constexpr double compress_ratio_threshold = 0.9;  // Must save ≥ 10%

// Compression flags
inline constexpr std::uint8_t comp_flag_raw = 0;
inline constexpr std::uint8_t comp_flag_lz4 = 1;
```
