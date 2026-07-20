#pragma once
// zlink/ptr_map.hpp — Bidirectional pointer mapping for opaque handles
//
// When a remote function returns a pointer (e.g., CUdeviceptr from cuMemAlloc),
// the client needs a "fake" local pointer that maps to the real remote pointer.
// This class manages that mapping transparently.
//
// Design inspired by Lupine's handle remapping, but with:
//   - O(1) lookup in both directions
//   - Support for mmap'd shadow regions (pointer values are real addresses)
//   - Thread-safe for concurrent RPC lanes

#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <vector>
#include <cassert>
#include <span>

namespace zlink {

class ptr_map {
public:
    // Initialize with a shadow memory region.
    // `base` is the start of the mmap'd region; `size` is its length.
    // Local pointers will be allocated from this region so they look
    // like real addresses to the client application.
    void set_shadow_region(std::uintptr_t base, std::size_t size) {
        std::lock_guard lock(mutex_);
        shadow_base_ = base;
        shadow_size_ = size;
        shadow_cursor_ = base;
    }

    // Register a mapping: remote_ptr <-> local_ptr
    // If shadow region is available, local_ptr is allocated from it.
    std::uintptr_t map(std::uintptr_t remote_ptr) {
        std::lock_guard lock(mutex_);

        // Already mapped?
        if (auto it = remote_to_local_.find(remote_ptr); it != remote_to_local_.end()) {
            return it->second;
        }

        std::uintptr_t local_ptr;
        if (shadow_base_ != 0) {
            // Allocate from shadow region
            local_ptr = shadow_cursor_;
            shadow_cursor_ += 8; // At least 8 bytes apart so pointers are distinct
            assert(shadow_cursor_ <= shadow_base_ + shadow_size_);
        } else {
            // Use sequential IDs as "pointers"
            local_ptr = static_cast<std::uintptr_t>(++next_id_);
        }

        remote_to_local_[remote_ptr] = local_ptr;
        local_to_remote_[local_ptr] = remote_ptr;
        return local_ptr;
    }

    // Look up remote pointer from local pointer
    std::optional<std::uintptr_t> to_remote(std::uintptr_t local_ptr) const {
        std::lock_guard lock(mutex_);
        if (auto it = local_to_remote_.find(local_ptr); it != local_to_remote_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Look up local pointer from remote pointer
    std::optional<std::uintptr_t> to_local(std::uintptr_t remote_ptr) const {
        std::lock_guard lock(mutex_);
        if (auto it = remote_to_local_.find(remote_ptr); it != remote_to_local_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Remove a mapping
    void unmap(std::uintptr_t local_ptr) {
        std::lock_guard lock(mutex_);
        if (auto it = local_to_remote_.find(local_ptr); it != local_to_remote_.end()) {
            remote_to_local_.erase(it->second);
            local_to_remote_.erase(it);
        }
    }

    // Batch translate: walk a span of local pointers and replace with remote pointers
    // Useful for function args that contain embedded pointers
    void translate_pointers(std::span<std::uintptr_t> ptrs) const {
        std::lock_guard lock(mutex_);
        for (auto& p : ptrs) {
            if (auto it = local_to_remote_.find(p); it != local_to_remote_.end()) {
                p = it->second;
            }
        }
    }

    // Check if a pointer is a known local (shadow) pointer
    bool is_local(std::uintptr_t ptr) const {
        std::lock_guard lock(mutex_);
        return local_to_remote_.contains(ptr);
    }

    // Check if a pointer is a known remote pointer
    bool is_remote(std::uintptr_t ptr) const {
        std::lock_guard lock(mutex_);
        return remote_to_local_.contains(ptr);
    }

    // Clear all mappings
    void clear() {
        std::lock_guard lock(mutex_);
        remote_to_local_.clear();
        local_to_remote_.clear();
        shadow_cursor_ = shadow_base_;
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return remote_to_local_.size();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::uintptr_t, std::uintptr_t> remote_to_local_;
    std::unordered_map<std::uintptr_t, std::uintptr_t> local_to_remote_;

    std::uintptr_t shadow_base_   = 0;
    std::size_t    shadow_size_   = 0;
    std::uintptr_t shadow_cursor_ = 0;
    std::uint64_t  next_id_       = 0;
};

} // namespace zlink
