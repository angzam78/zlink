#pragma once
// zlink/compress.hpp — LZ4 compression for pipeline_mem data transfers
//
// Compresses large sync/read data in pipeline_mem frames to reduce
// network transfer time, especially on bandwidth-limited links.
//
// Design:
//   - Per-entry compression flag (not per-frame)
//   - Threshold: only compress entries >= 4 KB
//   - Ratio check: only use compressed version if >10% smaller
//   - Uses LZ4 (fast compression, ~3-4 GB/s compress, ~4-7 GB/s decompress)
//   - Zero overhead for small entries or incompressible data
//
// Wire format change (sync entries):
//   Before: [8B addr][8B size][data...]
//   After:  [8B addr][8B original_size][1B comp_flag][data...]
//     comp_flag: 0 = raw data, 1 = LZ4 compressed
//     When compressed, data is LZ4-compressed bytes; original_size is the
//     uncompressed size (needed for decompression).
//     When raw, data is original bytes; original_size == data.length().
//
// Wire format change (read response entries):
//   Before: [4B len][data...]
//   After:  [4B len][1B comp_flag][data...]
//     Same convention: comp_flag indicates if data is LZ4-compressed.
//     len includes the 1B comp_flag byte but not the original size.
//     For compressed reads, the server also prepends 8B original_size:
//     [4B len][1B comp_flag=1][8B original_size][compressed_data...]

#include <zlink/config.hpp>

#include <cstdint>
#include <cstddef>
#include <vector>
#include <span>
#include <cstring>

// LZ4 C API
#include <lz4.h>

namespace zlink {

// ── Compression constants ─────────────────────────────────────────────
inline constexpr std::size_t compress_threshold = 4096;  // Don't bother below 4 KB
inline constexpr double compress_ratio_threshold = 0.9;   // Must save at least 10%
inline constexpr std::uint8_t comp_flag_raw = 0;
inline constexpr std::uint8_t comp_flag_lz4 = 1;

// ── Compress a data buffer ────────────────────────────────────────────
// Returns {compressed_data, comp_flag}.
// If the data is too small or doesn't compress well, returns raw data
// with comp_flag_raw.
struct compress_result {
    std::vector<std::byte> data;
    std::uint8_t comp_flag;       // comp_flag_raw or comp_flag_lz4
    std::size_t original_size;    // Always the uncompressed size
};

inline compress_result compress(std::span<const std::byte> input) {
    compress_result result;
    result.original_size = input.size();

    // Below threshold: don't compress
    if (input.size() < compress_threshold) {
        result.data.assign(input.begin(), input.end());
        result.comp_flag = comp_flag_raw;
        return result;
    }

    // LZ4 worst-case output size
    int max_compressed = LZ4_compressBound(static_cast<int>(input.size()));
    if (max_compressed <= 0) {
        // Input too large for LZ4? Just send raw.
        result.data.assign(input.begin(), input.end());
        result.comp_flag = comp_flag_raw;
        return result;
    }

    result.data.resize(static_cast<std::size_t>(max_compressed));

    int compressed_size = LZ4_compress_default(
        reinterpret_cast<const char*>(input.data()),
        reinterpret_cast<char*>(result.data.data()),
        static_cast<int>(input.size()),
        max_compressed
    );

    if (compressed_size <= 0) {
        // Compression failed — send raw
        result.data.assign(input.begin(), input.end());
        result.comp_flag = comp_flag_raw;
        return result;
    }

    // Ratio check: only use compressed version if it saves enough
    double ratio = static_cast<double>(compressed_size) / static_cast<double>(input.size());
    if (ratio >= compress_ratio_threshold) {
        // Not enough savings — send raw
        result.data.assign(input.begin(), input.end());
        result.comp_flag = comp_flag_raw;
        return result;
    }

    // Compressed version is worthwhile
    result.data.resize(static_cast<std::size_t>(compressed_size));
    result.comp_flag = comp_flag_lz4;
    return result;
}

// ── Decompress a data buffer ─────────────────────────────────────────
// If comp_flag is comp_flag_raw, just returns the input.
// If comp_flag is comp_flag_lz4, decompresses using original_size.
inline std::vector<std::byte> decompress(
    std::span<const std::byte> input,
    std::uint8_t comp_flag,
    std::size_t original_size)
{
    if (comp_flag == comp_flag_raw || comp_flag != comp_flag_lz4) {
        return std::vector<std::byte>(input.begin(), input.end());
    }

    std::vector<std::byte> output(original_size);

    int result = LZ4_decompress_safe(
        reinterpret_cast<const char*>(input.data()),
        reinterpret_cast<char*>(output.data()),
        static_cast<int>(input.size()),
        static_cast<int>(original_size)
    );

    if (result < 0) {
        // Decompression failed — this is a protocol error
        // Return empty buffer; caller should handle error
        return {};
    }

    return output;
}

// ── Convenience: compress a pending_sync's data ──────────────────────
inline compress_result compress_sync_data(const std::vector<std::byte>& data) {
    return compress(std::span<const std::byte>(data.data(), data.size()));
}

// ── Convenience: compress a pending_read's data ──────────────────────
inline compress_result compress_read_data(const std::vector<std::byte>& data) {
    return compress(std::span<const std::byte>(data.data(), data.size()));
}

} // namespace zlink
