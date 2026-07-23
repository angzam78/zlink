// zlink/memory_region.cpp — Remote memory subsystem implementation
//
//   - cached_memory_client with r3map-style chunk caching
//   - host_memory_mirror for bidirectional client↔server memory sync
//   - demand paging + write tracking via memory_page_tracker

#include <zlink/memory.hpp>
#include <zlink/rpc.hpp>
#include <zlink/config.hpp>
#include <zlink/chunk_cache.hpp>
#include <zlink/memory_page_tracker.hpp>

#include <cstring>
#include <cstdlib>
#include <system_error>
#include <stdexcept>
#include <algorithm>

#include <sys/mman.h>

namespace zlink {

// ══════════════════════════════════════════════════════════════════════
//  RPC remote backend — wraps memory_client RPC as a remote_backend
//  (r3map's "RPC backend" equivalent, but using zpp_bits instead of gRPC)
// ══════════════════════════════════════════════════════════════════════
class cached_memory_client::rpc_remote_backend : public remote_backend {
public:
    explicit rpc_remote_backend(rpc_client_base* rpc) : rpc_(rpc) {}

    // Set the base address used to translate offsets → absolute addresses
    // for server communication. The server's host_memory_mirror uses
    // absolute client addresses.
    void set_base(std::uintptr_t base) { base_ = base; }

    std::error_code read_at(std::uintptr_t offset,
                             std::span<std::byte> buf) override {
        // Send a host_read memory_op frame and get the data back.
        // The server's handle_memory_op() reads from its host_memory_mirror.
        // Translate offset → absolute address for server lookup.
        mem_request req;
        req.op = mem_op::host_read;
        req.remote_addr = offset + base_;
        req.size = buf.size();

        std::vector<std::byte> req_data(sizeof(mem_request));
        std::memcpy(req_data.data(), &req, sizeof(mem_request));

        frame read_frame;
        read_frame.call_id = 0;
        read_frame.type = frame_type::memory_op;
        read_frame.payload = req_data;

        auto& tp = rpc_->get_transport();
        auto ec = tp.send(read_frame);
        if (ec) return ec;

        frame resp_frame;
        ec = tp.receive(resp_frame);
        if (ec) return ec;

        if (resp_frame.payload.size() < sizeof(mem_response)) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        mem_response resp;
        std::memcpy(&resp, resp_frame.payload.data(), sizeof(mem_response));

        if (failed(resp.status)) {
            return std::make_error_code(std::errc::io_error);
        }

        std::size_t data_offset = sizeof(mem_response);
        std::size_t data_size = resp_frame.payload.size() - data_offset;
        if (data_size > buf.size()) data_size = buf.size();
        std::memcpy(buf.data(), resp_frame.payload.data() + data_offset, data_size);

        return {};
    }

    std::error_code write_at(std::uintptr_t offset,
                              std::span<const std::byte> data) override {
        // Send as a host_sync memory_op frame (not an RPC request).
        // The server's handle_memory_op() stores the data in its
        // host_memory_mirror. Translate offset → absolute address.
        mem_request req;
        req.op = mem_op::host_sync;
        req.remote_addr = offset + base_;
        req.size = data.size();

        std::vector<std::byte> payload(sizeof(mem_request) + data.size());
        std::memcpy(payload.data(), &req, sizeof(mem_request));
        std::memcpy(payload.data() + sizeof(mem_request), data.data(), data.size());

        frame sync_frame;
        sync_frame.call_id = 0;
        sync_frame.type = frame_type::memory_op;
        sync_frame.payload = std::move(payload);

        auto& tp = rpc_->get_transport();
        auto ec = tp.send(sync_frame);
        if (ec) return ec;

        // Wait for the ack
        frame resp_frame;
        ec = tp.receive(resp_frame);
        if (ec) return ec;

        if (resp_frame.payload.size() < sizeof(mem_response)) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        mem_response resp;
        std::memcpy(&resp, resp_frame.payload.data(), sizeof(mem_response));
        if (failed(resp.status)) {
            return std::make_error_code(std::errc::io_error);
        }

        return {};
    }

    std::size_t size() const override { return region_size_; }

    void set_region_size(std::size_t s) { region_size_ = s; }

private:
    rpc_client_base* rpc_;
    std::size_t region_size_ = 0;
    std::uintptr_t base_ = 0;  // shadow region base for offset→absolute translation
};

// ══════════════════════════════════════════════════════════════════════
//  cached_memory_client implementation
// ══════════════════════════════════════════════════════════════════════
struct cached_memory_client::impl {
    rpc_client_base* rpc = nullptr;
    std::uintptr_t shadow_base = 0;  // base of the shadow region for addr↔offset translation

    std::unique_ptr<memory_page_tracker> mptracker;
};

cached_memory_client::cached_memory_client(class rpc_client_base& rpc,
                                           std::size_t region_size)
    : cached_memory_client(rpc, region_size, chunk_cache_config{})
{}

cached_memory_client::cached_memory_client(class rpc_client_base& rpc,
                                           std::size_t region_size,
                                           const chunk_cache_config& config)
    : impl_(std::make_unique<impl>())
    , local_store_(std::make_shared<memory_local_store>(region_size))
{
    impl_->rpc = &rpc;

    // Create the RPC remote backend
    rpc_backend_ = std::make_shared<rpc_remote_backend>(&rpc);
    rpc_backend_->set_region_size(region_size);

    // Create the chunk cache (r3map's SyncedReadWriterAt)
    cache_ = std::make_unique<chunk_cache>(
        rpc_backend_,      // remote backend
        local_store_,      // local store (cache)
        config
    );
}

cached_memory_client::~cached_memory_client() {
    disable_demand_paging();
    stop_background();
}

std::error_code cached_memory_client::read(std::uintptr_t remote_addr,
                                            std::span<std::byte> local_buf) {
    // Translate absolute address → offset for chunk_cache/local_store
    std::uintptr_t offset = remote_addr - impl_->shadow_base;
    return cache_->read(offset, local_buf);
}

std::error_code cached_memory_client::write(std::uintptr_t remote_addr,
                                             std::span<const std::byte> local_buf) {
    // Translate absolute address → offset for chunk_cache/local_store
    std::uintptr_t offset = remote_addr - impl_->shadow_base;
    return cache_->write(offset, local_buf);
}

std::error_code cached_memory_client::alloc(std::size_t size, std::uintptr_t& out_addr) {
    mem_request req;
    req.op = mem_op::alloc;
    req.remote_addr = 0;
    req.size = size;

    std::vector<std::byte> req_data(sizeof(mem_request));
    std::memcpy(req_data.data(), &req, sizeof(mem_request));

    std::vector<std::byte> resp_data;
    auto ec = impl_->rpc->send_request(req_data, resp_data);
    if (ec) return ec;

    mem_response resp;
    std::memcpy(&resp, resp_data.data(), sizeof(mem_response));
    if (failed(resp.status)) {
        return std::make_error_code(std::errc::io_error);
    }

    out_addr = resp.remote_addr;
    rpc_backend_->set_region_size(std::max(rpc_backend_->size(), out_addr + size));
    return {};
}

std::error_code cached_memory_client::free(std::uintptr_t remote_addr) {
    mem_request req;
    req.op = mem_op::free_op;
    req.remote_addr = remote_addr;
    req.size = 0;

    std::vector<std::byte> req_data(sizeof(mem_request));
    std::memcpy(req_data.data(), &req, sizeof(mem_request));

    std::vector<std::byte> resp_data;
    auto ec = impl_->rpc->send_request(req_data, resp_data);
    if (ec) return ec;

    mem_response resp;
    std::memcpy(&resp, resp_data.data(), sizeof(mem_response));
    if (failed(resp.status)) {
        return std::make_error_code(std::errc::io_error);
    }

    return {};
}

std::size_t cached_memory_client::flush_dirty() {
    // 1. Collect dirty pages from the memory_page_tracker (uffd WP or mprotect SIGSEGV)
    //    and write them into the chunk_cache's local store.
    //    The memory_page_tracker reports absolute addresses; translate to offsets.
    std::size_t pushed = 0;
    if (impl_->mptracker) {
        auto dirty = impl_->mptracker->collect_dirty();
        for (const auto& r : dirty) {
            std::uintptr_t offset = r.addr - impl_->shadow_base;
            cache_->write(offset, std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(r.addr), r.len));
            ++pushed;
        }
    }

    // 2. Push ALL dirty pages (from memory_page_tracker above + any explicit writes)
    //    to the server via the remote backend.
    pushed += cache_->sync_dirty();

    return pushed;
}

void cached_memory_client::invalidate(std::span<const std::int64_t> page_offsets) {
    cache_->invalidate(page_offsets);
}

void cached_memory_client::invalidate_all() {
    cache_->invalidate_all();
}

void cached_memory_client::start_background() {
    cache_->start_background();
}

void cached_memory_client::stop_background() {
    cache_->stop_background();
}

// ── Demand paging through the cache ──────────────────────────────────
// Key difference from the old implementation: page faults now go through
// the chunk_cache. If the page is already cached locally, the fault is
// resolved from local store with ZERO network traffic. Only the first
// access of each page triggers a remote fetch.

std::error_code cached_memory_client::enable_demand_paging(std::uintptr_t base,
                                                            std::size_t size) {
    // Store the shadow region base for address translation.
    impl_->shadow_base = base;
    rpc_backend_->set_base(base);

    // The memory_page_tracker is the SINGLE uffd/mprotect handler for this region.
    // It handles:
    //   MISSING/read faults  → calls read_fault_cb → fetches via chunk_cache
    //                          → UFFDIO_COPY or mprotect+memcpy (done inside tracker)
    //   WP/write faults      → marks page dirty in internal bitmap
    //
    // flush_dirty() later calls mptracker->collect_dirty() to get dirty ranges
    // and pushes them to the server.
    const std::size_t ps = cache_->chunk_size();

    // Read-fault callback: fetch page from remote via chunk_cache.
    // The memory_page_tracker does the UFFDIO_COPY / mprotect+memcpy itself.
    // Translate the absolute fault address → offset for chunk_cache.
    auto read_cb = [this, base, ps](std::uintptr_t fault_addr, std::size_t /*len*/)
        -> std::vector<std::byte>
    {
        std::uintptr_t offset = fault_addr - base;
        std::vector<std::byte> page_data(ps);
        auto ec = cache_->read(offset, page_data);
        if (ec) return {};
        return page_data;
    };

    impl_->mptracker = memory_page_tracker::create(base, size, std::move(read_cb));

    if (!impl_->mptracker) {
        return std::make_error_code(std::errc::not_supported);
    }

    return {};
}

void cached_memory_client::disable_demand_paging() {
    impl_->mptracker.reset();
}

// ══════════════════════════════════════════════════════════════════════
//  host_memory_mirror implementation (server side)
// ══════════════════════════════════════════════════════════════════════
host_memory_mirror::host_memory_mirror() = default;
host_memory_mirror::~host_memory_mirror() {
    std::lock_guard lock(mutex_);
    for (auto& r : regions_) {
        if (r.server_mmap) {
            ::munmap(r.server_mmap, r.size);
        }
    }
}

std::uintptr_t host_memory_mirror::register_region(std::uintptr_t client_base,
                                                     std::size_t size) {
    std::lock_guard lock(mutex_);

    // Check if already registered
    for (auto& r : regions_) {
        if (r.client_base == client_base && r.size == size) {
            return r.server_base;
        }
    }

    // Allocate a server-side mirror region with mmap
    void* server_ptr = ::mmap(nullptr, size,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS,
                               -1, 0);
    if (server_ptr == MAP_FAILED) {
        return 0;
    }

    region r;
    r.client_base = client_base;
    r.size = size;
    r.server_base = reinterpret_cast<std::uintptr_t>(server_ptr);
    r.server_mmap = server_ptr;
    regions_.push_back(r);

    return r.server_base;
}

void host_memory_mirror::unregister_region(std::uintptr_t client_base) {
    std::lock_guard lock(mutex_);
    auto it = std::remove_if(regions_.begin(), regions_.end(),
        [client_base](const region& r) {
            return r.client_base == client_base;
        });

    for (auto i = it; i != regions_.end(); ++i) {
        if (i->server_mmap) {
            ::munmap(i->server_mmap, i->size);
        }
    }
    regions_.erase(it, regions_.end());
}

std::error_code host_memory_mirror::sync_page(std::uintptr_t client_addr,
                                                std::span<const std::byte> data) {
    std::lock_guard lock(mutex_);

    for (auto& r : regions_) {
        if (client_addr >= r.client_base &&
            client_addr + data.size() <= r.client_base + r.size) {
            // Found the region — copy data to the mirror
            std::size_t offset = client_addr - r.client_base;
            auto* dst = reinterpret_cast<std::byte*>(r.server_base) + offset;
            std::memcpy(dst, data.data(), data.size());
            return {};
        }
    }

    // No existing region covers this address — auto-register one.
    // This is the transparent r3map behavior: the mirror lazily creates
    // regions as the client syncs data, without explicit registration.
    // We align the region to page boundaries and add a generous size
    // to cover likely nearby syncs.
    constexpr std::size_t page_size = 4096;
    std::uintptr_t aligned_base = client_addr & ~(page_size - 1);
    // Allocate enough pages to cover the sync range + some headroom
    std::size_t region_size = ((data.size() + page_size - 1) / page_size + 8) * page_size;

    void* server_ptr = ::mmap(nullptr, region_size,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS,
                               -1, 0);
    if (server_ptr == MAP_FAILED) {
        return std::make_error_code(std::errc::not_enough_memory);
    }

    region r;
    r.client_base = aligned_base;
    r.size = region_size;
    r.server_base = reinterpret_cast<std::uintptr_t>(server_ptr);
    r.server_mmap = server_ptr;

    std::size_t offset = client_addr - aligned_base;
    auto* dst = reinterpret_cast<std::byte*>(r.server_base) + offset;
    std::memcpy(dst, data.data(), data.size());

    regions_.push_back(r);
    return {};
}

std::error_code host_memory_mirror::read(std::uintptr_t client_addr,
                                          std::span<std::byte> buf) const {
    std::lock_guard lock(mutex_);

    for (const auto& r : regions_) {
        if (client_addr >= r.client_base &&
            client_addr + buf.size() <= r.client_base + r.size) {
            std::size_t offset = client_addr - r.client_base;
            auto* src = reinterpret_cast<const std::byte*>(r.server_base) + offset;
            std::memcpy(buf.data(), src, buf.size());
            return {};
        }
    }

    return std::make_error_code(std::errc::address_not_available);
}

std::optional<std::uintptr_t> host_memory_mirror::translate(std::uintptr_t client_addr) const {
    std::lock_guard lock(mutex_);

    for (const auto& r : regions_) {
        if (client_addr >= r.client_base &&
            client_addr < r.client_base + r.size) {
            std::size_t offset = client_addr - r.client_base;
            return r.server_base + offset;
        }
    }

    return std::nullopt;
}

bool host_memory_mirror::is_registered(std::uintptr_t client_addr) const {
    std::lock_guard lock(mutex_);
    for (const auto& r : regions_) {
        if (client_addr >= r.client_base &&
            client_addr < r.client_base + r.size) {
            return true;
        }
    }
    return false;
}

} // namespace zlink
