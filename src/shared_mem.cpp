// zlink/shared_mem.cpp — Shared Memory Plane implementation
//
// Implements the r3map-inspired shared memory plane that makes client
// memory transparently accessible to the server.

#include <zlink/shared_mem.hpp>
#include <zlink/config.hpp>

#include <cstring>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <filesystem>

namespace zlink {

// ── Chunked backend implementation ─────────────────────────────────────
chunked_backend::chunked_backend(const std::string& dir, std::size_t chunk_size)
    : dir_(dir), chunk_size_(chunk_size), total_size_(0) {
    std::filesystem::create_directories(dir);
}

std::error_code chunked_backend::read_at(std::span<std::byte> buf,
                                          std::int64_t offset) {
    std::size_t remaining = buf.size();
    std::size_t buf_offset = 0;
    std::int64_t current = offset;

    while (remaining > 0) {
        std::size_t chunk_idx = static_cast<std::size_t>(current / chunk_size_);
        std::size_t in_chunk_offset = static_cast<std::size_t>(current % chunk_size_);
        std::size_t to_read = std::min(remaining, chunk_size_ - in_chunk_offset);

        std::string chunk_path = dir_ + "/chunk_" + std::to_string(chunk_idx);
        if (std::filesystem::exists(chunk_path)) {
            std::ifstream f(chunk_path, std::ios::binary);
            f.seekg(static_cast<std::streamsize>(in_chunk_offset));
            f.read(reinterpret_cast<char*>(buf.data() + buf_offset),
                   static_cast<std::streamsize>(to_read));
        } else {
            // Chunk doesn't exist — fill with zeros
            std::memset(buf.data() + buf_offset, 0, to_read);
        }

        buf_offset += to_read;
        current += static_cast<std::int64_t>(to_read);
        remaining -= to_read;
    }
    return {};
}

std::error_code chunked_backend::write_at(std::span<const std::byte> buf,
                                           std::int64_t offset) {
    std::size_t remaining = buf.size();
    std::size_t buf_offset = 0;
    std::int64_t current = offset;

    while (remaining > 0) {
        std::size_t chunk_idx = static_cast<std::size_t>(current / chunk_size_);
        std::size_t in_chunk_offset = static_cast<std::size_t>(current % chunk_size_);
        std::size_t to_write = std::min(remaining, chunk_size_ - in_chunk_offset);

        std::string chunk_path = dir_ + "/chunk_" + std::to_string(chunk_idx);

        // If writing a partial chunk, read existing data first
        if (to_write < chunk_size_ && std::filesystem::exists(chunk_path)) {
            std::vector<std::byte> chunk(chunk_size_);
            std::ifstream fin(chunk_path, std::ios::binary);
            fin.read(reinterpret_cast<char*>(chunk.data()),
                     static_cast<std::streamsize>(chunk_size_));

            std::memcpy(chunk.data() + in_chunk_offset,
                       buf.data() + buf_offset, to_write);

            std::ofstream fout(chunk_path, std::ios::binary);
            fout.write(reinterpret_cast<const char*>(chunk.data()),
                      static_cast<std::streamsize>(chunk_size_));
        } else {
            std::ofstream f(chunk_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(buf.data() + buf_offset),
                   static_cast<std::streamsize>(to_write));
        }

        buf_offset += to_write;
        current += static_cast<std::int64_t>(to_write);
        remaining -= to_write;
    }

    // Update total size
    std::int64_t end = offset + static_cast<std::int64_t>(buf.size());
    if (end > total_size_) total_size_ = end;

    return {};
}

std::int64_t chunked_backend::size() const {
    return total_size_;
}

std::error_code chunked_backend::sync() {
    // File-based backend: OS handles sync
    return {};
}

// ── Shared Memory Plane implementation ─────────────────────────────────

std::uint64_t shared_mem_plane::register_region(std::uintptr_t client_addr,
                                                  std::size_t size,
                                                  bool writable) {
    std::lock_guard lock(mutex_);

    std::uint64_t id = next_region_id_++;

    region_entry entry;
    entry.descriptor = {
        .client_addr = client_addr,
        .size        = size,
        .region_id   = id,
        .writable    = writable,
    };

    // Create a memory backend wrapping the client's memory
    // On the client side, this points to real application memory.
    // On the server side, we use an RPC backend instead (set later).
    auto* raw = reinterpret_cast<std::byte*>(client_addr);
    entry.be = std::make_shared<memory_backend>(std::span<std::byte>(raw, size));

    // Initialize server-side cache
    std::size_t n_chunks = (size + chunk_size_ - 1) / chunk_size_;
    entry.local_cache.resize(size, std::byte{0});
    entry.chunk_present.resize(n_chunks, false);
    entry.chunk_dirty.resize(n_chunks, false);
    entry.chunk_size = chunk_size_;

    regions_[id] = std::move(entry);
    return id;
}

std::uint64_t shared_mem_plane::register_region_with_backend(
        std::uintptr_t client_addr,
        std::size_t size,
        std::shared_ptr<backend> be,
        bool writable) {
    std::lock_guard lock(mutex_);

    std::uint64_t id = next_region_id_++;

    region_entry entry;
    entry.descriptor = {
        .client_addr = client_addr,
        .size        = size,
        .region_id   = id,
        .writable    = writable,
    };
    entry.be = std::move(be);

    std::size_t n_chunks = (size + chunk_size_ - 1) / chunk_size_;
    entry.local_cache.resize(size, std::byte{0});
    entry.chunk_present.resize(n_chunks, false);
    entry.chunk_dirty.resize(n_chunks, false);
    entry.chunk_size = chunk_size_;

    regions_[id] = std::move(entry);
    return id;
}

void shared_mem_plane::unregister_region(std::uint64_t region_id) {
    std::lock_guard lock(mutex_);
    regions_.erase(region_id);
}

std::shared_ptr<backend> shared_mem_plane::get_backend(std::uint64_t region_id) const {
    std::lock_guard lock(mutex_);
    auto it = regions_.find(region_id);
    if (it == regions_.end()) return nullptr;
    return it->second.be;
}

std::optional<std::pair<std::uint64_t, std::int64_t>>
shared_mem_plane::find_region(std::uintptr_t client_addr, std::size_t len) const {
    std::lock_guard lock(mutex_);

    for (const auto& [id, entry] : regions_) {
        std::uintptr_t start = entry.descriptor.client_addr;
        std::uintptr_t end   = start + entry.descriptor.size;

        if (client_addr >= start && client_addr + len <= end) {
            return {{id, static_cast<std::int64_t>(client_addr - start)}};
        }
    }
    return std::nullopt;
}

// ── Server-side pointer translation ────────────────────────────────────
// This is the MAGIC function that solves the pointer problem.
//
// When the server receives an RPC call like:
//   cuMemcpyHtoD(dst, srcHost=0x7fff1234, 1024)
//
// srcHost=0x7fff1234 is a CLIENT address. The server can't dereference it.
// But if the client registered that memory region, we can:
//   1. Find the region that contains 0x7fff1234
//   2. Calculate the offset within that region
//   3. If the chunk is already cached locally, return the cache address
//   4. If not, fetch it from the client via the backend (ReadAt)
//   5. Return a server-local pointer that the real function can dereference
//
// The real cuMemcpyHtoD then copies from our server-local cache,
// which contains the exact same data the client had at that address.

std::byte* shared_mem_plane::translate_to_server(std::uintptr_t client_addr,
                                                   std::size_t len,
                                                   std::uint64_t region_id) {
    std::lock_guard lock(mutex_);

    auto it = regions_.find(region_id);
    if (it == regions_.end()) return nullptr;

    auto& entry = it->second;
    std::int64_t offset = static_cast<std::int64_t>(client_addr - entry.descriptor.client_addr);

    if (offset < 0 ||
        static_cast<std::size_t>(offset) + len > entry.descriptor.size) {
        return nullptr;
    }

    // Determine which chunks we need
    std::size_t first_chunk = static_cast<std::size_t>(offset) / entry.chunk_size;
    std::size_t last_byte   = static_cast<std::size_t>(offset) + len - 1;
    std::size_t last_chunk  = last_byte / entry.chunk_size;

    // Fetch any missing chunks from the client's backend
    for (std::size_t c = first_chunk; c <= last_chunk; ++c) {
        if (!entry.chunk_present[c]) {
            std::size_t chunk_offset = c * entry.chunk_size;
            std::size_t to_read = std::min(entry.chunk_size,
                                           entry.descriptor.size - chunk_offset);

            std::span<std::byte> dst(entry.local_cache.data() + chunk_offset,
                                     to_read);
            auto ec = entry.be->read_at(dst, static_cast<std::int64_t>(chunk_offset));
            if (ec) return nullptr;  // Failed to fetch from client

            entry.chunk_present[c] = true;
        }
    }

    // Return a pointer into our local cache
    return entry.local_cache.data() + offset;
}

void shared_mem_plane::mark_dirty(std::uintptr_t server_addr, std::size_t len) {
    std::lock_guard lock(mutex_);

    // Find which region this server address belongs to
    for (auto& [id, entry] : regions_) {
        std::byte* cache_start = entry.local_cache.data();
        std::byte* cache_end   = cache_start + entry.local_cache.size();

        if (reinterpret_cast<std::byte*>(server_addr) >= cache_start &&
            reinterpret_cast<std::byte*>(server_addr) < cache_end) {
            std::size_t offset = reinterpret_cast<std::byte*>(server_addr) - cache_start;
            std::size_t first_chunk = offset / entry.chunk_size;
            std::size_t last_byte   = offset + len - 1;
            std::size_t last_chunk  = last_byte / entry.chunk_size;

            for (std::size_t c = first_chunk; c <= last_chunk; ++c) {
                entry.chunk_dirty[c] = true;
            }
            break;
        }
    }
}

std::error_code shared_mem_plane::flush_dirty() {
    std::lock_guard lock(mutex_);

    for (auto& [id, entry] : regions_) {
        if (!entry.descriptor.writable) continue;

        for (std::size_t c = 0; c < entry.chunk_dirty.size(); ++c) {
            if (!entry.chunk_dirty[c]) continue;

            std::size_t chunk_offset = c * entry.chunk_size;
            std::size_t to_write = std::min(entry.chunk_size,
                                            entry.descriptor.size - chunk_offset);

            std::span<const std::byte> src(entry.local_cache.data() + chunk_offset,
                                           to_write);
            auto ec = entry.be->write_at(src, static_cast<std::int64_t>(chunk_offset));
            if (ec) return ec;

            entry.chunk_dirty[c] = false;
        }
    }
    return {};
}

// ── Bulk transfer ──────────────────────────────────────────────────────
std::error_code shared_mem_plane::pull(std::uint64_t region_id,
                                        std::int64_t offset,
                                        std::span<std::byte> buf) {
    std::lock_guard lock(mutex_);

    auto it = regions_.find(region_id);
    if (it == regions_.end()) return std::make_error_code(std::errc::invalid_argument);

    return it->second.be->read_at(buf, offset);
}

std::error_code shared_mem_plane::push(std::uint64_t region_id,
                                        std::int64_t offset,
                                        std::span<const std::byte> buf) {
    std::lock_guard lock(mutex_);

    auto it = regions_.find(region_id);
    if (it == regions_.end()) return std::make_error_code(std::errc::invalid_argument);

    return it->second.be->write_at(buf, offset);
}

// ── Background operations (stub implementations) ──────────────────────
void shared_mem_plane::start_prefetch(std::uint64_t region_id,
                                       std::int64_t priority_offset) {
    // TODO: Implement background prefetch thread
    // This would use a separate thread to sequentially fetch chunks
    // from the client, prioritizing the area around priority_offset
}

void shared_mem_plane::stop_prefetch(std::uint64_t region_id) {}

void shared_mem_plane::start_background_writeback(std::uint64_t region_id,
                                                    int interval_ms) {
    // TODO: Implement periodic dirty page writeback
    // This would flush dirty chunks every interval_ms milliseconds
}

void shared_mem_plane::stop_background_writeback(std::uint64_t region_id) {}

// ── Utilities ──────────────────────────────────────────────────────────
std::size_t shared_mem_plane::region_count() const {
    std::lock_guard lock(mutex_);
    return regions_.size();
}

void shared_mem_plane::clear() {
    std::lock_guard lock(mutex_);
    regions_.clear();
}

} // namespace zlink
