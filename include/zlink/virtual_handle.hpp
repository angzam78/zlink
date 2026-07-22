#pragma once
// zlink/virtual_handle.hpp — Tagged virtual handle system for CUDA RPC
//
// THE PROBLEM:
//   CUDA workloads have sequential dependencies:
//     cuMemAlloc → dev_ptr → cuMemcpyHtoD(dev_ptr, ...) → cuLaunchKernel(dev_ptr, ...)
//   In a naive pipeline, cuMemAlloc must be a barrier (wait for dev_ptr).
//   This defeats pipelining for the entire workload.
//
// THE SOLUTION:
//   Virtual handles break the dependency chain. The client assigns virtual
//   handle IDs (0, 1, 2, ...) when enqueuing handle-producing calls.
//   Subsequent calls reference these virtual IDs. The server translates
//   virtual → real handles when processing the pipeline batch.
//
//   With virtual handles, the ENTIRE workload pipelines into ONE batch:
//     cuMemAlloc       → VH(0)     // enqueued, no barrier!
//     cuMemcpyHtoD(VH(0), ...)     // enqueued, references VH(0)
//     cuLaunchKernel(VH(0), ...)   // enqueued, references VH(0)
//     cuMemcpyDtoH(..., VH(0), ...) // readback: flush + get data
//     → 1 round-trip instead of N
//
// ENCODING:
//   Bit 63 set = virtual handle. Lower 63 bits = virtual ID.
//   Real CUDA pointers are always < 2^63 (user-space addresses), so
//   this is safe and unambiguous.
//
//   Example: VH(0) = 0x8000000000000000
//            VH(1) = 0x8000000000000001
//   Real ptr:  0x7f18c6c00000 (bit 63 clear)

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <iostream>

namespace zlink {

// ── Virtual handle encoding ───────────────────────────────────────────
constexpr std::uint64_t vhandle_flag = (1ULL << 63);
constexpr std::uint64_t vhandle_id_mask = ~vhandle_flag;

inline bool is_virtual_handle(std::uint64_t val) noexcept {
    return (val & vhandle_flag) != 0;
}

inline std::uint64_t make_virtual_handle(std::uint32_t id) noexcept {
    return vhandle_flag | static_cast<std::uint64_t>(id);
}

inline std::uint32_t virtual_handle_id(std::uint64_t val) noexcept {
    return static_cast<std::uint32_t>(val & vhandle_id_mask);
}

// ── Server-side handle table ──────────────────────────────────────────
// Maps virtual handle IDs to real CUDA handle values.
// The server populates this as it processes pipeline batches:
//   1. Process cuMemAlloc → get real dev_ptr
//   2. Register: VH(0) → 0x7f18c6c00000
//   3. Process cuMemcpyHtoD(VH(0), ...) → translate VH(0) to real ptr
class handle_table {
public:
    // Register a virtual → real mapping
    void register_handle(std::uint32_t virtual_id, std::uint64_t real_value) {
        std::lock_guard lock(mutex_);
        table_[virtual_id] = real_value;
    }

    // Look up a real value for a virtual ID
    std::optional<std::uint64_t> lookup(std::uint32_t virtual_id) const {
        std::lock_guard lock(mutex_);
        auto it = table_.find(virtual_id);
        if (it != table_.end()) return it->second;
        return std::nullopt;
    }

    // Translate a value: if it's a virtual handle, resolve to real.
    // If it's not a virtual handle, return as-is.
    std::uint64_t translate(std::uint64_t val) const {
        if (!is_virtual_handle(val)) return val;
        auto real = lookup(virtual_handle_id(val));
        if (real) return *real;
        // Virtual handle not found — this is a bug, but don't crash
        std::cerr << "  [handle_table] WARNING: unresolved VH("
                  << virtual_handle_id(val) << ")\n";
        return val;  // Return as-is (will likely fail the CUDA call)
    }

    // Check if a virtual handle is registered
    bool has_handle(std::uint32_t virtual_id) const {
        std::lock_guard lock(mutex_);
        return table_.find(virtual_id) != table_.end();
    }

    // Clear all mappings (e.g., between pipeline batches)
    void clear() {
        std::lock_guard lock(mutex_);
        table_.clear();
    }

    // Number of registered handles
    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return table_.size();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::uint32_t, std::uint64_t> table_;
};

// ── Client-side virtual handle allocator ──────────────────────────────
// Assigns sequential virtual handle IDs. Used by cuda_pipeline when
// enqueuing handle-producing calls.
class virtual_handle_allocator {
public:
    std::uint32_t allocate() {
        return next_id_++;
    }

    void reset() { next_id_ = 0; }

private:
    std::uint32_t next_id_ = 0;
};

// ── Handle manifest entry ─────────────────────────────────────────────
// Describes which pipeline calls produce handles.
// Sent alongside the pipeline_mem frame so the server knows
// which return values to register.
struct handle_manifest_entry {
    std::uint32_t call_index;     // 0-indexed position in the pipeline
    std::uint32_t virtual_id;     // The virtual handle ID to assign
    std::uint32_t return_field;   // Which field in the return struct is the handle (0-indexed)
    std::uint32_t _reserved;      // Padding for alignment
};

// Serialize handle manifest entries into a byte vector
inline void serialize_handle_manifest(
    const std::vector<handle_manifest_entry>& entries,
    std::vector<std::byte>& out)
{
    std::uint32_t count = static_cast<std::uint32_t>(entries.size());
    std::size_t start = out.size();
    out.resize(start + 4 + entries.size() * sizeof(handle_manifest_entry));
    std::memcpy(out.data() + start, &count, 4);
    if (!entries.empty()) {
        std::memcpy(out.data() + start + 4, entries.data(),
                    entries.size() * sizeof(handle_manifest_entry));
    }
}

// Parse handle manifest from byte data
inline std::vector<handle_manifest_entry> parse_handle_manifest(
    std::span<const std::byte> data, std::size_t& bytes_consumed)
{
    std::vector<handle_manifest_entry> entries;
    if (data.size() < 4) { bytes_consumed = 0; return entries; }

    std::uint32_t count = 0;
    std::memcpy(&count, data.data(), 4);
    bytes_consumed = 4;

    if (count > 0 && data.size() >= 4 + count * sizeof(handle_manifest_entry)) {
        entries.resize(count);
        std::memcpy(entries.data(), data.data() + 4,
                    count * sizeof(handle_manifest_entry));
        bytes_consumed = 4 + count * sizeof(handle_manifest_entry);
    }

    return entries;
}

} // namespace zlink
