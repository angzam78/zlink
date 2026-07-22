// zlink/memory_region.cpp — Remote memory subsystem implementation
//
// Now includes:
//   - cached_memory_client with r3map-style chunk caching
//   - host_memory_mirror for bidirectional client↔server memory sync
//   - userfaultfd demand paging that goes through the cache

#include <zlink/memory.hpp>
#include <zlink/rpc.hpp>
#include <zlink/config.hpp>
#include <zlink/chunk_cache.hpp>

#include <cstring>
#include <cstdlib>
#include <system_error>
#include <stdexcept>
#include <algorithm>

#if ZLINK_HAS_USERFAULTFD
    #include <linux/userfaultfd.h>
    #include <sys/ioctl.h>
    #include <sys/mman.h>
    #include <sys/syscall.h>
    #include <unistd.h>
    #include <poll.h>
    #include <pthread.h>
    #include <fcntl.h>      // O_CLOEXEC, O_NONBLOCK
#endif

namespace zlink {

// ══════════════════════════════════════════════════════════════════════
//  RPC remote backend — wraps memory_client RPC as a remote_backend
//  (r3map's "RPC backend" equivalent, but using zpp_bits instead of gRPC)
// ══════════════════════════════════════════════════════════════════════
class cached_memory_client::rpc_remote_backend : public remote_backend {
public:
    explicit rpc_remote_backend(rpc_client_base* rpc) : rpc_(rpc) {}

    std::error_code read_at(std::uintptr_t offset,
                             std::span<std::byte> buf) override {
        // Send a memory read RPC and get the data back
        mem_request req;
        req.op = mem_op::read;
        req.remote_addr = offset;
        req.size = buf.size();

        std::vector<std::byte> req_data(sizeof(mem_request));
        std::memcpy(req_data.data(), &req, sizeof(mem_request));

        std::vector<std::byte> resp_data;
        auto ec = rpc_->send_request(req_data, resp_data);
        if (ec) return ec;

        if (resp_data.size() < sizeof(mem_response)) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        mem_response resp;
        std::memcpy(&resp, resp_data.data(), sizeof(mem_response));

        if (failed(resp.status)) {
            return std::make_error_code(std::errc::io_error);
        }

        std::size_t data_offset = sizeof(mem_response);
        std::size_t data_size = resp_data.size() - data_offset;
        if (data_size > buf.size()) data_size = buf.size();
        std::memcpy(buf.data(), resp_data.data() + data_offset, data_size);

        return {};
    }

    std::error_code write_at(std::uintptr_t offset,
                              std::span<const std::byte> data) override {
        mem_request req;
        req.op = mem_op::write;
        req.remote_addr = offset;
        req.size = data.size();

        std::vector<std::byte> req_data(sizeof(mem_request) + data.size());
        std::memcpy(req_data.data(), &req, sizeof(mem_request));
        std::memcpy(req_data.data() + sizeof(mem_request), data.data(), data.size());

        std::vector<std::byte> resp_data;
        auto ec = rpc_->send_request(req_data, resp_data);
        if (ec) return ec;

        if (resp_data.size() < sizeof(mem_response)) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        mem_response resp;
        std::memcpy(&resp, resp_data.data(), sizeof(mem_response));
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
};

// ══════════════════════════════════════════════════════════════════════
//  cached_memory_client implementation
// ══════════════════════════════════════════════════════════════════════
struct cached_memory_client::impl {
    rpc_client_base* rpc = nullptr;

#if ZLINK_HAS_USERFAULTFD
    int uffd = -1;
    std::uintptr_t demand_base = 0;
    std::size_t demand_size = 0;
    std::thread fault_thread;
    std::atomic<bool> demand_paging_enabled{false};
#endif
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
#if ZLINK_HAS_USERFAULTFD
    disable_demand_paging();
#endif
    stop_background();
}

std::error_code cached_memory_client::read(std::uintptr_t remote_addr,
                                            std::span<std::byte> local_buf) {
    // Go through the chunk cache — if page is cached, no network!
    return cache_->read(remote_addr, local_buf);
}

std::error_code cached_memory_client::write(std::uintptr_t remote_addr,
                                             std::span<const std::byte> local_buf) {
    // Write through the cache — marks pages dirty, pusher syncs later
    return cache_->write(remote_addr, local_buf);
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
    return cache_->sync_dirty();
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

#if ZLINK_HAS_USERFAULTFD
std::error_code cached_memory_client::enable_demand_paging(std::uintptr_t base,
                                                            std::size_t size) {
    impl_->uffd = static_cast<int>(syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK));
    if (impl_->uffd < 0) {
        return std::make_error_code(std::errc::invalid_argument);
    }

    struct uffdio_api api{};
    api.api = UFFD_API;
    api.features = UFFD_FEATURE_PAGEFAULT_FLAG_WP;
    if (ioctl(impl_->uffd, UFFDIO_API, &api) < 0) {
        ::close(impl_->uffd);
        impl_->uffd = -1;
        return std::make_error_code(std::errc::invalid_argument);
    }

    struct uffdio_register reg{};
    reg.range.start = base;
    reg.range.len = size;
    reg.mode = UFFDIO_REGISTER_MODE_MISSING;
    if (ioctl(impl_->uffd, UFFDIO_REGISTER, &reg) < 0) {
        ::close(impl_->uffd);
        impl_->uffd = -1;
        return std::make_error_code(std::errc::invalid_argument);
    }

    impl_->demand_base = base;
    impl_->demand_size = size;
    impl_->demand_paging_enabled.store(true);

    // Fault handler thread — resolves faults through the chunk cache
    impl_->fault_thread = std::thread([this]() {
        while (impl_->demand_paging_enabled.load()) {
            struct pollfd pfd{};
            pfd.fd = impl_->uffd;
            pfd.events = POLLIN;

            int n = poll(&pfd, 1, 100);
            if (n <= 0) continue;

            struct uffd_msg msg{};
            if (::read(impl_->uffd, &msg, sizeof(msg)) != sizeof(msg)) continue;

            if (msg.event != UFFD_EVENT_PAGEFAULT) continue;

            std::uintptr_t fault_addr = msg.arg.pagefault.address;
            std::size_t page_size = cache_->chunk_size();
            std::uintptr_t page_base = fault_addr & ~(page_size - 1);

            // ── THE KEY DIFFERENCE ──
            // Read through the chunk cache. If this page was previously
            // fetched, it's already in the local store — NO network!
            // Only the first fault for each page hits the remote.
            std::vector<std::byte> page_data(page_size);
            auto ec = cache_->read(page_base, page_data);
            if (ec) continue;

            // Resolve the page fault
            struct uffdio_copy copy{};
            copy.dst = page_base;
            copy.src = reinterpret_cast<std::uintptr_t>(page_data.data());
            copy.len = page_size;
            copy.mode = 0;
            ioctl(impl_->uffd, UFFDIO_COPY, &copy);
        }
    });

    return {};
}

void cached_memory_client::disable_demand_paging() {
    impl_->demand_paging_enabled.store(false);
    if (impl_->fault_thread.joinable()) {
        impl_->fault_thread.join();
    }
    if (impl_->uffd >= 0) {
        ::close(impl_->uffd);
        impl_->uffd = -1;
    }
}
#endif // ZLINK_HAS_USERFAULTFD

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

// ══════════════════════════════════════════════════════════════════════
//  memory_server implementation
// ══════════════════════════════════════════════════════════════════════
struct memory_server::impl {
    memory_server::read_fn  read_handler;
    memory_server::write_fn write_handler;
    memory_server::alloc_fn alloc_handler;
    memory_server::free_fn  free_handler;
};

memory_server::memory_server() : impl_(std::make_unique<impl>()) {}
memory_server::~memory_server() = default;

void memory_server::set_read_handler(read_fn h)  { impl_->read_handler = std::move(h); }
void memory_server::set_write_handler(write_fn h) { impl_->write_handler = std::move(h); }
void memory_server::set_alloc_handler(alloc_fn h) { impl_->alloc_handler = std::move(h); }
void memory_server::set_free_handler(free_fn h)   { impl_->free_handler = std::move(h); }

mem_response memory_server::handle_request(const mem_request& req,
                                            std::span<const std::byte> write_data,
                                            std::vector<std::byte>& read_data) {
    mem_response resp{};
    resp.status = error_code::ok;
    resp.remote_addr = req.remote_addr;
    resp.size = 0;

    switch (req.op) {
        case mem_op::read: {
            if (!impl_->read_handler) {
                resp.status = error_code::server_error;
                break;
            }
            read_data.resize(req.size);
            auto ec = impl_->read_handler(req.remote_addr, read_data);
            if (ec) {
                resp.status = error_code::server_error;
            } else {
                resp.size = req.size;
            }
            break;
        }

        case mem_op::write: {
            if (!impl_->write_handler) {
                resp.status = error_code::server_error;
                break;
            }
            auto ec = impl_->write_handler(req.remote_addr, write_data);
            if (ec) {
                resp.status = error_code::server_error;
            } else {
                resp.size = write_data.size();
            }
            break;
        }

        case mem_op::alloc: {
            if (!impl_->alloc_handler) {
                resp.status = error_code::server_error;
                break;
            }
            auto ec = impl_->alloc_handler(req.size, resp.remote_addr);
            if (ec) {
                resp.status = error_code::server_error;
            } else {
                resp.size = req.size;
            }
            break;
        }

        case mem_op::free_op: {
            if (!impl_->free_handler) {
                resp.status = error_code::server_error;
                break;
            }
            auto ec = impl_->free_handler(req.remote_addr);
            if (ec) {
                resp.status = error_code::server_error;
            }
            break;
        }

        case mem_op::sync: {
            // Flush any pending writes
            resp.status = error_code::ok;
            resp.size = 0;
            break;
        }

        default:
            resp.status = error_code::server_error;
            break;
    }

    return resp;
}

std::error_code memory_server::handle_host_sync(std::uintptr_t client_addr,
                                                  std::span<const std::byte> data) {
    return host_mirror_.sync_page(client_addr, data);
}

} // namespace zlink
