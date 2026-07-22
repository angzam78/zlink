// zlink/prefetch_worker.cpp — Background page prefetch implementation

#include <zlink/prefetch_worker.hpp>

#include <algorithm>
#include <cassert>

namespace zlink {

prefetch_worker::prefetch_worker(chunk_cache& cache, prefetch_config cfg)
    : cache_(cache)
    , cfg_(std::move(cfg))
{}

prefetch_worker::~prefetch_worker() {
    stop();
}

// ── Lifecycle ────────────────────────────────────────────────────────

void prefetch_worker::start() {
    if (running_.load()) return;
    running_.store(true);
    paused_.store(false);
    worker_thread_ = std::thread(&prefetch_worker::worker_loop, this);
}

void prefetch_worker::stop() {
    running_.store(false);
    hint_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

// ── Access pattern hints ─────────────────────────────────────────────

void prefetch_worker::record_access(std::uintptr_t addr, std::size_t size, bool is_write) {
    auto chunk_size = cache_.chunk_size();
    if (chunk_size == 0) return;

    // Calculate page range for this access
    std::int64_t first_page = static_cast<std::int64_t>(addr / chunk_size);
    std::int64_t last_page  = static_cast<std::int64_t>((addr + size - 1) / chunk_size);

    // Update pattern for first page (primary access point)
    update_pattern(first_page, is_write);

    // If write: cancel any pending prefetch for these pages
    if (is_write) {
        std::vector<std::int64_t> pages;
        for (auto p = first_page; p <= last_page; p++) {
            pages.push_back(p);
        }
        cancel_prefetch(pages);
    }

    // Notify worker thread that we have new access data
    {
        std::lock_guard lock(hint_mutex_);
        recent_hints_.push_back(first_page);
    }
    hint_cv_.notify_one();
}

void prefetch_worker::cancel_prefetch(std::span<const std::int64_t> page_offsets) {
    std::lock_guard lock(in_flight_mutex_);
    for (auto p : page_offsets) {
        in_flight_.erase(p);
    }
}

void prefetch_worker::pause() {
    paused_.store(true);
}

void prefetch_worker::resume() {
    paused_.store(false);
    hint_cv_.notify_one();
}

access_pattern prefetch_worker::current_pattern() const {
    std::lock_guard lock(pattern_mutex_);
    return pattern_.type;
}

// ── Pattern detection ────────────────────────────────────────────────

void prefetch_worker::update_pattern(std::int64_t page_index, bool is_write) {
    std::lock_guard lock(pattern_mutex_);

    if (pattern_.last_page < 0) {
        pattern_.last_page = page_index;
        return;
    }

    std::int64_t delta = page_index - pattern_.last_page;

    if (delta == 1) {
        // Sequential access
        if (pattern_.type == access_pattern::sequential) {
            pattern_.confidence++;
        } else {
            pattern_.type = access_pattern::sequential;
            pattern_.confidence = 1;
            pattern_.stride = 1;
        }
    } else if (delta > 1 && delta < 256) {
        // Possible strided access
        if (pattern_.type == access_pattern::strided && delta == pattern_.stride) {
            pattern_.confidence++;
        } else {
            pattern_.type = access_pattern::strided;
            pattern_.stride = delta;
            pattern_.confidence = 1;
        }
    } else if (delta < 0 || delta >= 256) {
        // Random or reverse access
        pattern_.confidence = pattern_.confidence > 2
            ? pattern_.confidence - 2 : 0;
        if (pattern_.confidence < cfg_.min_confidence) {
            pattern_.type = access_pattern::random;
        }
    }

    pattern_.last_page = page_index;
}

std::vector<std::int64_t> prefetch_worker::predict_next_pages(std::int64_t current_page) {
    std::vector<std::int64_t> predicted;

    if (pattern_.confidence < cfg_.min_confidence) {
        return predicted;  // Not confident enough to predict
    }

    switch (pattern_.type) {
        case access_pattern::sequential:
            for (std::size_t i = 1; i <= cfg_.lookahead; i++) {
                predicted.push_back(current_page + static_cast<std::int64_t>(i));
            }
            break;

        case access_pattern::strided:
            for (std::size_t i = 1; i <= cfg_.lookahead; i++) {
                predicted.push_back(current_page + pattern_.stride * static_cast<std::int64_t>(i));
            }
            break;

        case access_pattern::random:
            // No prediction for random access
            break;
    }

    return predicted;
}

// ── Worker thread ────────────────────────────────────────────────────

void prefetch_worker::worker_loop() {
    while (running_.load()) {
        // Wait for hints or timeout
        {
            std::unique_lock lock(hint_mutex_);
            hint_cv_.wait_for(lock, cfg_.idle_timeout, [this] {
                return !running_.load() || !recent_hints_.empty();
            });
        }

        if (!running_.load()) break;
        if (paused_.load()) continue;

        // Get the latest hint
        std::int64_t current_page = -1;
        {
            std::lock_guard lock(hint_mutex_);
            if (!recent_hints_.empty()) {
                current_page = recent_hints_.back();
                recent_hints_.clear();
            }
        }

        if (current_page < 0) continue;

        // Predict next pages based on access pattern
        std::vector<std::int64_t> to_prefetch;
        {
            std::lock_guard lock(pattern_mutex_);
            to_prefetch = predict_next_pages(current_page);
        }

        if (to_prefetch.empty()) continue;

        // Fetch pages that aren't local and aren't already in flight
        std::size_t fetched = 0;
        for (auto page : to_prefetch) {
            if (fetched >= cfg_.max_outstanding) break;
            if (page < 0) continue;

            // Skip if already local
            if (cache_.is_local(page)) continue;

            // Skip if already being prefetched
            {
                std::lock_guard lock(in_flight_mutex_);
                if (in_flight_.count(page) > 0) continue;
                in_flight_.insert(page);
            }

            // Fetch the page from remote
            auto ec = cache_.fetch_chunk(page);
            if (!ec) {
                hits_.fetch_add(1);
                fetched++;
            } else {
                misses_.fetch_add(1);
            }

            // Remove from in-flight
            {
                std::lock_guard lock(in_flight_mutex_);
                in_flight_.erase(page);
            }
        }
    }
}

} // namespace zlink
