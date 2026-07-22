// zlink/chunk_cache.cpp — Page-level cache implementation
//
// Port of r3map's SyncedReadWriterAt + Puller + Pusher to C++.
//
// The core insight: once a chunk is fetched from remote, it stays local.
// Subsequent reads hit the local store — zero network overhead.
// Only explicit invalidation or cache eviction forces a re-fetch.

#include <zlink/chunk_cache.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace zlink {

// ── Constructor / Destructor ──────────────────────────────────────────
chunk_cache::chunk_cache(std::shared_ptr<remote_backend> remote,
                         std::shared_ptr<local_store> local)
    : chunk_cache(std::move(remote), std::move(local), chunk_cache_config{})
{}

chunk_cache::chunk_cache(std::shared_ptr<remote_backend> remote,
                         std::shared_ptr<local_store> local,
                         const chunk_cache_config& config)
    : remote_(std::move(remote))
    , local_(std::move(local))
    , config_(config)
{
    if (remote_) {
        total_size_ = remote_->size();
    }

    // Default pull priority: all equal
    if (!config_.pull_priority) {
        config_.pull_priority = [](std::int64_t) -> std::int64_t { return 1; };
    }
}

chunk_cache::~chunk_cache() {
    stop_background();
}

// ── Core read (SyncedReadWriterAt.ReadAt equivalent) ──────────────────
std::error_code chunk_cache::read(std::uintptr_t offset, std::span<std::byte> buf) {
    // Handle reads that span multiple chunks
    std::size_t bytes_read = 0;
    while (bytes_read < buf.size()) {
        std::int64_t chunk_idx = offset_to_chunk(offset + bytes_read);
        std::uintptr_t chunk_off = chunk_to_offset(chunk_idx);
        std::size_t offset_in_chunk = (offset + bytes_read) - chunk_off;
        std::size_t bytes_in_chunk = config_.chunk_size - offset_in_chunk;
        std::size_t to_read = std::min(bytes_in_chunk, buf.size() - bytes_read);

        chunk_state& cs = get_chunk(chunk_idx);

        if (cs.local) {
            // ── FAST PATH: data is in local store, no network! ──
            std::span<std::byte> read_buf(buf.data() + bytes_read, to_read);
            auto ec = local_->read_at(offset + bytes_read, read_buf);
            if (ec) return ec;
        } else {
            // ── SLOW PATH: fetch from remote, cache locally ──
            // We need to fetch the whole chunk (even if we only need part of it)
            // because the cache operates at chunk granularity.
            std::vector<std::byte> chunk_data(config_.chunk_size);
            auto ec = remote_->read_at(chunk_off, chunk_data);
            if (ec) return ec;

            // Store in local cache
            ec = local_->write_at(chunk_off, chunk_data);
            if (ec) return ec;

            // Mark as local (r3map's chk.local = true)
            {
                std::unique_lock lock(chunks_mutex_);
                cs.local = true;
            }

            // Copy the requested portion to the caller
            std::memcpy(buf.data() + bytes_read,
                        chunk_data.data() + offset_in_chunk,
                        to_read);

            // Notify callback
            if (on_chunk_local_) {
                on_chunk_local_(chunk_idx);
            }

            // Also mark as pushable for the pusher (r3map's onChunkIsLocal)
            {
                std::lock_guard lock(pushable_mutex_);
                pushable_chunks_.insert(chunk_idx);
            }
        }

        bytes_read += to_read;
    }

    return {};
}

// ── Core write (SyncedReadWriterAt.WriteAt equivalent) ────────────────
std::error_code chunk_cache::write(std::uintptr_t offset, std::span<const std::byte> data) {
    // Handle writes that span multiple chunks
    std::size_t bytes_written = 0;
    while (bytes_written < data.size()) {
        std::int64_t chunk_idx = offset_to_chunk(offset + bytes_written);
        std::uintptr_t chunk_off = chunk_to_offset(chunk_idx);
        std::size_t offset_in_chunk = (offset + bytes_written) - chunk_off;
        std::size_t bytes_in_chunk = config_.chunk_size - offset_in_chunk;
        std::size_t to_write = std::min(bytes_in_chunk, data.size() - bytes_written);

        chunk_state& cs = get_chunk(chunk_idx);

        // If the chunk is not local yet, we need to fetch it first
        // (write-after-read coherence: can't write partial chunk without
        //  knowing the rest of the data)
        if (!cs.local && (offset_in_chunk != 0 || to_write != config_.chunk_size)) {
            // Fetch the whole chunk first so we can do a partial write
            std::vector<std::byte> chunk_data(config_.chunk_size);
            auto ec = remote_->read_at(chunk_off, chunk_data);
            if (ec) return ec;

            // Merge the write into the chunk
            std::memcpy(chunk_data.data() + offset_in_chunk,
                        data.data() + bytes_written,
                        to_write);

            // Write the full chunk to local store
            ec = local_->write_at(chunk_off, chunk_data);
            if (ec) return ec;

            cs.local = true;
        } else {
            // Write directly to local store
            std::span<const std::byte> write_buf(data.data() + bytes_written, to_write);
            auto ec = local_->write_at(offset + bytes_written, write_buf);
            if (ec) return ec;

            if (!cs.local) {
                cs.local = true;
            }
        }

        // Mark as dirty (r3map's TrackingReadWriterAt)
        cs.dirty = true;
        {
            std::lock_guard lock(dirty_mutex_);
            dirty_chunks_.insert(chunk_idx);
        }

        // Mark as pushable
        {
            std::lock_guard lock(pushable_mutex_);
            pushable_chunks_.insert(chunk_idx);
        }

        bytes_written += to_write;
    }

    return {};
}

// ── Invalidation (MarkAsRemote equivalent) ────────────────────────────
void chunk_cache::invalidate(std::span<const std::int64_t> dirty_offsets) {
    std::unique_lock lock(chunks_mutex_);
    for (auto offset : dirty_offsets) {
        auto it = chunks_.find(offset);
        if (it != chunks_.end()) {
            it->second.local = false;
            // Don't clear dirty — the pusher may still need to write it back
        }
    }
}

void chunk_cache::invalidate_all() {
    std::unique_lock lock(chunks_mutex_);
    for (auto& [idx, state] : chunks_) {
        state.local = false;
    }
}

// ── Background operations ─────────────────────────────────────────────
void chunk_cache::start_background() {
    if (running_.exchange(true)) return; // Already running

    // Build the pull queue — all chunks that aren't local yet
    {
        std::lock_guard lock(pull_queue_mutex_);
        pull_queue_.clear();
        std::size_t num_chunks = total_size_ / config_.chunk_size;
        if (total_size_ % config_.chunk_size != 0) num_chunks++;

        for (std::int64_t i = 0; i < static_cast<std::int64_t>(num_chunks); ++i) {
            pull_queue_.push_back(i);
        }

        // Sort by pull priority (higher priority first, like r3map)
        if (config_.pull_priority) {
            std::sort(pull_queue_.begin(), pull_queue_.end(),
                [this](std::int64_t a, std::int64_t b) {
                    return config_.pull_priority(a) > config_.pull_priority(b);
                });
        }

        pull_next_ = 0;
    }

    // Start puller
    puller_thread_ = std::thread([this]() { puller_loop(); });

    // Start pusher
    pusher_thread_ = std::thread([this]() { pusher_loop(); });
}

void chunk_cache::stop_background() {
    if (!running_.exchange(false)) return;

    // Flush remaining dirty chunks
    sync_dirty();

    if (puller_thread_.joinable()) puller_thread_.join();
    if (pusher_thread_.joinable()) pusher_thread_.join();
}

std::size_t chunk_cache::sync_dirty() {
    std::unordered_set<std::int64_t> to_push;
    {
        std::lock_guard lock(dirty_mutex_);
        to_push = std::move(dirty_chunks_);
        dirty_chunks_.clear();
    }

    std::size_t pushed = 0;
    for (auto chunk_idx : to_push) {
        auto ec = push_chunk(chunk_idx);
        if (!ec) pushed++;
    }

    // Sync the local store
    local_->sync();

    return pushed;
}

// ── Query ─────────────────────────────────────────────────────────────
bool chunk_cache::is_local(std::int64_t chunk_index) const {
    std::shared_lock lock(chunks_mutex_);
    auto it = chunks_.find(chunk_index);
    return it != chunks_.end() && it->second.local;
}

std::size_t chunk_cache::local_count() const {
    std::shared_lock lock(chunks_mutex_);
    std::size_t count = 0;
    for (auto& [idx, state] : chunks_) {
        if (state.local) count++;
    }
    return count;
}

std::size_t chunk_cache::total_chunks() const {
    return total_size_ / config_.chunk_size +
           (total_size_ % config_.chunk_size != 0 ? 1 : 0);
}

// ── Private helpers ───────────────────────────────────────────────────
chunk_state& chunk_cache::get_chunk(std::int64_t chunk_idx) {
    std::unique_lock lock(chunks_mutex_);
    auto it = chunks_.find(chunk_idx);
    if (it == chunks_.end()) {
        auto [inserted, _] = chunks_.emplace(chunk_idx, chunk_state{});
        return inserted->second;
    }
    return it->second;
}

std::error_code chunk_cache::fetch_chunk(std::int64_t chunk_idx) {
    std::uintptr_t chunk_off = chunk_to_offset(chunk_idx);

    std::vector<std::byte> chunk_data(config_.chunk_size);
    auto ec = remote_->read_at(chunk_off, chunk_data);
    if (ec) return ec;

    ec = local_->write_at(chunk_off, chunk_data);
    if (ec) return ec;

    {
        std::unique_lock lock(chunks_mutex_);
        chunks_[chunk_idx].local = true;
    }

    if (on_chunk_local_) {
        on_chunk_local_(chunk_idx);
    }

    return {};
}

std::error_code chunk_cache::push_chunk(std::int64_t chunk_idx) {
    std::uintptr_t chunk_off = chunk_to_offset(chunk_idx);

    // Calculate actual chunk size (last chunk may be smaller)
    std::size_t sz = config_.chunk_size;
    if (chunk_off + sz > total_size_) {
        sz = total_size_ - chunk_off;
    }

    std::vector<std::byte> chunk_data(sz);
    auto ec = local_->read_at(chunk_off, chunk_data);
    if (ec) return ec;

    ec = remote_->write_at(chunk_off, chunk_data);
    if (ec) return ec;

    // Clear dirty flag
    {
        std::unique_lock lock(chunks_mutex_);
        auto it = chunks_.find(chunk_idx);
        if (it != chunks_.end()) {
            it->second.dirty = false;
        }
    }

    return {};
}

// ── Background puller (r3map's Puller) ────────────────────────────────
void chunk_cache::puller_loop() {
    while (running_.load()) {
        std::int64_t chunk_idx = -1;

        {
            std::lock_guard lock(pull_queue_mutex_);
            if (pull_next_ < pull_queue_.size()) {
                chunk_idx = pull_queue_[pull_next_++];
            }
        }

        if (chunk_idx < 0) {
            // All chunks pulled
            break;
        }

        // Skip already-local chunks
        if (is_local(chunk_idx)) continue;

        auto ec = fetch_chunk(chunk_idx);
        if (ec && config_.verbose) {
            std::cerr << "zlink chunk_cache: pull failed for chunk "
                      << chunk_idx << ": " << ec.message() << "\n";
        }
    }
}

// ── Background pusher (r3map's Pusher) ────────────────────────────────
void chunk_cache::pusher_loop() {
    while (running_.load()) {
        // Wait for push interval
        std::this_thread::sleep_for(config_.push_interval);

        if (!running_.load()) break;

        // Collect pushable dirty chunks
        std::unordered_set<std::int64_t> to_push;
        {
            std::lock_guard lock(dirty_mutex_);
            to_push = dirty_chunks_;
            dirty_chunks_.clear();
        }

        for (auto chunk_idx : to_push) {
            if (!running_.load()) break;

            auto ec = push_chunk(chunk_idx);
            if (ec && config_.verbose) {
                std::cerr << "zlink chunk_cache: push failed for chunk "
                          << chunk_idx << ": " << ec.message() << "\n";
            }
        }

        local_->sync();
    }
}

} // namespace zlink
