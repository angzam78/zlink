// zlink/write_tracker.cpp — userfaultfd write-protect tracking
//
// Tier 1: userfaultfd WP + WP_ASYNC  (kernel 6.2+)
//         Write faults are auto-resolved by the kernel; we only get
//         a notification. Zero ioctls per write fault.
//
// Tier 2: userfaultfd WP synchronous (kernel 4.11+)
//         Write faults block until the handler calls UFFDIO_WRITEPROTECT
//         to unprotect and wake. One ioctl per first-write per page.

#include <zlink/write_tracker.hpp>
#include <zlink/config.hpp>

#include <algorithm>
#include <cstring>
#include <atomic>
#include <thread>
#include <functional>
#include <mutex>

#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>

#if ZLINK_HAS_USERFAULTFD
    #include <linux/userfaultfd.h>
    #include <sys/ioctl.h>
    #include <sys/syscall.h>
    #include <poll.h>
    #include <fcntl.h>
#endif

namespace zlink {

namespace {

inline std::size_t page_size() {
    static const std::size_t ps = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    return ps;
}

inline std::uintptr_t page_align_down(std::uintptr_t addr) {
    return addr & ~(page_size() - 1);
}

// ══════════════════════════════════════════════════════════════════════
//  userfaultfd write-protect tracker (Tier 1 and Tier 2)
// ══════════════════════════════════════════════════════════════════════
//
//  A single uffd fd handles both:
//  - MISSING faults (read demand paging — caller handles via callback)
//  - WP faults (write tracking — marked dirty here)
//
//  With WP_ASYNC: the kernel auto-unprotects the page and delivers a
//  notification. We just mark dirty. No UFFDIO_WRITEPROTECT ioctl needed.
//
//  Without WP_ASYNC: the faulting thread blocks until we call
//  UFFDIO_WRITEPROTECT(WP=0) to unprotect and wake it.

#if ZLINK_HAS_USERFAULTFD

class uffd_write_tracker final : public write_tracker {
public:
    using fault_callback = std::function<std::vector<std::byte>(std::uintptr_t addr, std::size_t len)>;

    uffd_write_tracker(std::uintptr_t base, std::size_t size,
                       bool wp_async, fault_callback read_fault_cb)
        : base_(base)
        , size_(size)
        , ps_(page_size())
        , wp_async_(wp_async)
        , read_fault_cb_(std::move(read_fault_cb))
        , dirty_pages_(std::make_unique<std::atomic<std::uint8_t>[]>(
              (size + ps_ - 1) / ps_))
    {
        open_uffd();
    }

    ~uffd_write_tracker() override {
        running_.store(false, std::memory_order_release);
        if (uffd_ >= 0) {
            ::close(uffd_);
            uffd_ = -1;
        }
        if (fault_thread_.joinable())
            fault_thread_.join();
    }

    bool is_valid() const noexcept { return uffd_ >= 0; }

    const char* backend_name() const noexcept override {
        return wp_async_ ? "uffd WP_ASYNC" : "uffd WP sync";
    }

    std::error_code protect(std::uintptr_t addr, std::size_t len) override {
        if (addr < base_ || addr + len > base_ + size_)
            return std::make_error_code(std::errc::address_not_available);

        std::uintptr_t page = page_align_down(addr);
        std::uintptr_t aligned_end = page_align_down(addr + len - 1) + ps_;

        struct uffdio_writeprotect wp{};
        wp.range.start = page;
        wp.range.len   = aligned_end - page;
        wp.mode        = UFFDIO_WRITEPROTECT_MODE_WP;

        if (::ioctl(uffd_, UFFDIO_WRITEPROTECT, &wp) < 0)
            return std::make_error_code(static_cast<std::errc>(errno));
        return {};
    }

    std::vector<page_range> collect_dirty() override {
        std::vector<page_range> dirty;
        std::size_t num_pages = (size_ + ps_ - 1) / ps_;

        for (std::size_t i = 0; i < num_pages; ++i) {
            if (dirty_pages_[i].exchange(0) != 0)
                dirty.push_back({base_ + i * ps_, ps_});
        }

        // Re-protect dirty pages for the next cycle
        for (const auto& r : dirty) {
            struct uffdio_writeprotect wp{};
            wp.range.start = r.addr;
            wp.range.len   = r.len;
            wp.mode        = UFFDIO_WRITEPROTECT_MODE_WP;
            ::ioctl(uffd_, UFFDIO_WRITEPROTECT, &wp);
        }
        return dirty;
    }

private:
    void open_uffd() {
        uffd_ = static_cast<int>(
            ::syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK));
        if (uffd_ < 0) return;

        // ── Feature negotiation ──────────────────────────────────────
        // Try WP_ASYNC first (Tier 1, kernel 6.2+). If the kernel
        // doesn't know the feature bit, UFFDIO_API returns -EINVAL
        // and we retry without it (Tier 2, kernel 4.11+).
        struct uffdio_api api{};
        api.api = UFFD_API;

        if (wp_async_) {
            api.features = UFFD_FEATURE_PAGEFAULT_FLAG_WP
                         | UFFD_FEATURE_WP_ASYNC;
            if (::ioctl(uffd_, UFFDIO_API, &api) < 0) {
                wp_async_ = false;
                api.features = UFFD_FEATURE_PAGEFAULT_FLAG_WP;
                if (::ioctl(uffd_, UFFDIO_API, &api) < 0) {
                    ::close(uffd_);
                    uffd_ = -1;
                    return;
                }
            }
        } else {
            api.features = UFFD_FEATURE_PAGEFAULT_FLAG_WP;
            if (::ioctl(uffd_, UFFDIO_API, &api) < 0) {
                ::close(uffd_);
                uffd_ = -1;
                return;
            }
        }

        // ── Register the region for MISSING | WP ─────────────────────
        // MISSING: read-side demand paging (delegated to read_fault_cb_)
        // WP:      write-side dirty tracking (handled here)
        struct uffdio_register reg{};
        reg.range.start = base_;
        reg.range.len   = size_;
        reg.mode = UFFDIO_REGISTER_MODE_MISSING | UFFDIO_REGISTER_MODE_WP;

        if (::ioctl(uffd_, UFFDIO_REGISTER, &reg) < 0) {
            ::close(uffd_);
            uffd_ = -1;
            return;
        }

        // ── Start fault handler thread ────────────────────────────────
        running_.store(true, std::memory_order_release);
        fault_thread_ = std::thread([this] { fault_loop(); });
    }

    void fault_loop() {
        const std::size_t num_pages = (size_ + ps_ - 1) / ps_;

        while (running_.load(std::memory_order_acquire)) {
            struct pollfd pfd{};
            pfd.fd = uffd_;
            pfd.events = POLLIN;

            int n = ::poll(&pfd, 1, 100);
            if (n <= 0) continue;

            struct uffd_msg msg{};
            if (::read(uffd_, &msg, sizeof(msg)) != sizeof(msg)) continue;
            if (msg.event != UFFD_EVENT_PAGEFAULT) continue;

            std::uintptr_t fault_addr = msg.arg.pagefault.address;
            bool is_write = msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WRITE;
            bool is_wp    = msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP;

            if (is_write || is_wp) {
                // ── Write fault: mark dirty ──────────────────────────
                std::size_t idx = (fault_addr - base_) / ps_;
                if (idx < num_pages)
                    dirty_pages_[idx].store(1, std::memory_order_relaxed);

                if (!wp_async_) {
                    // Synchronous: unprotect and wake the faulting thread
                    struct uffdio_writeprotect unwp{};
                    unwp.range.start = page_align_down(fault_addr);
                    unwp.range.len   = ps_;
                    unwp.mode        = 0; // WP=0 to unprotect
                    ::ioctl(uffd_, UFFDIO_WRITEPROTECT, &unwp);
                }
                // With WP_ASYNC the kernel already unprotected the page
            } else {
                // ── Read fault (missing page): fetch via callback ────
                if (read_fault_cb_) {
                    std::uintptr_t page_base = page_align_down(fault_addr);
                    auto page_data = read_fault_cb_(page_base, ps_);
                    if (!page_data.empty()) {
                        struct uffdio_copy copy{};
                        copy.dst = page_base;
                        copy.src = reinterpret_cast<std::uintptr_t>(page_data.data());
                        copy.len = ps_;
                        copy.mode = 0;
                        ::ioctl(uffd_, UFFDIO_COPY, &copy);

                        // WP-protect the newly fetched page for dirty tracking
                        struct uffdio_writeprotect wp{};
                        wp.range.start = page_base;
                        wp.range.len   = ps_;
                        wp.mode        = UFFDIO_WRITEPROTECT_MODE_WP;
                        ::ioctl(uffd_, UFFDIO_WRITEPROTECT, &wp);
                    }
                }
            }
        }
    }

    std::uintptr_t base_;
    std::size_t    size_;
    std::size_t    ps_;
    bool           wp_async_;
    fault_callback read_fault_cb_;

    int             uffd_ = -1;
    std::thread     fault_thread_;
    std::atomic<bool> running_{false};
    std::unique_ptr<std::atomic<std::uint8_t>[]> dirty_pages_;
};

#endif // ZLINK_HAS_USERFAULTFD

// ══════════════════════════════════════════════════════════════════════
//  mprotect + SIGSEGV write tracker (Tier 3 — containers, any POSIX system)
// ══════════════════════════════════════════════════════════════════════
//
//  When userfaultfd is unavailable (e.g., blocked by Docker seccomp),
//  we fall back to mprotect + SIGSEGV:
//
//  - Shadow region starts as PROT_NONE (no access)
//  - Read fault (PROT_NONE) → handler fetches page via callback,
//    copies data, mprotects to PROT_READ (write-protected)
//  - Write fault (PROT_READ) → handler marks page dirty,
//    mprotects to PROT_READ|PROT_WRITE
//  - collect_dirty() re-mprotects dirty pages to PROT_READ

class mprotect_write_tracker final : public write_tracker {
public:
    using fault_callback = std::function<std::vector<std::byte>(std::uintptr_t, std::size_t)>;

    mprotect_write_tracker(std::uintptr_t base, std::size_t size,
                           fault_callback read_fault_cb)
        : base_(base)
        , size_(size)
        , ps_(page_size())
        , read_fault_cb_(std::move(read_fault_cb))
        , dirty_pages_(std::make_unique<std::atomic<std::uint8_t>[]>(
              (size + ps_ - 1) / ps_))
        , page_state_(std::make_unique<std::atomic<std::uint8_t>[]>(
              (size + ps_ - 1) / ps_))
    {
        // Start with all pages PROT_NONE (no access)
        // Page states: 0=PROT_NONE, 1=PROT_READ, 2=PROT_READ|PROT_WRITE
        install_signal_handler();
        register_tracker(this);

        mprotect_region(base_, size_, PROT_NONE);
    }

    ~mprotect_write_tracker() override {
        unregister_tracker(this);
        // Restore full access so munmap/memcpy don't fault
        mprotect_region(base_, size_, PROT_READ | PROT_WRITE);
    }

    bool is_valid() const noexcept { return true; }

    const char* backend_name() const noexcept override {
        return "mprotect+SIGSEGV";
    }

    std::error_code protect(std::uintptr_t addr, std::size_t len) override {
        if (addr < base_ || addr + len > base_ + size_)
            return std::make_error_code(std::errc::address_not_available);
        std::uintptr_t page = page_align_down(addr);
        std::uintptr_t end = page_align_down(addr + len - 1) + ps_;
        mprotect_region(page, end - page, PROT_READ);
        // Mark pages as PROT_READ state
        for (std::uintptr_t p = page; p < end; p += ps_) {
            page_state_[(p - base_) / ps_].store(1, std::memory_order_relaxed);
        }
        return {};
    }

    std::vector<page_range> collect_dirty() override {
        std::vector<page_range> dirty;
        std::size_t num_pages = (size_ + ps_ - 1) / ps_;

        for (std::size_t i = 0; i < num_pages; ++i) {
            if (dirty_pages_[i].exchange(0) != 0)
                dirty.push_back({base_ + i * ps_, ps_});
        }

        // Re-protect dirty pages to PROT_READ for the next cycle
        for (const auto& r : dirty) {
            mprotect_region(r.addr, r.len, PROT_READ);
            page_state_[(r.addr - base_) / ps_].store(1, std::memory_order_relaxed);
        }
        return dirty;
    }

    // ── Signal handler entry point ─────────────────────────────────
    // Returns true if this tracker handled the fault.
    bool handle_fault(std::uintptr_t fault_addr, int /*si_code*/) {
        if (fault_addr < base_ || fault_addr >= base_ + size_)
            return false;

        std::uintptr_t page_base = page_align_down(fault_addr);
        std::size_t idx = (page_base - base_) / ps_;
        std::uint8_t state = page_state_[idx].load(std::memory_order_relaxed);

        if (state == 0) {
            // ── Read fault: page is PROT_NONE ──────────────────────
            // Temporarily allow writes so we can copy data in
            mprotect_region(page_base, ps_, PROT_READ | PROT_WRITE);

            // Fetch the page data via callback
            if (read_fault_cb_) {
                auto page_data = read_fault_cb_(page_base, ps_);
                if (!page_data.empty()) {
                    std::memcpy(reinterpret_cast<void*>(page_base),
                                page_data.data(),
                                std::min(page_data.size(), ps_));
                }
            }

            // Write-protect for dirty tracking
            mprotect_region(page_base, ps_, PROT_READ);
            page_state_[idx].store(1, std::memory_order_relaxed);
        } else if (state == 1) {
            // ── Write fault: page is PROT_READ ─────────────────────
            dirty_pages_[idx].store(1, std::memory_order_relaxed);
            mprotect_region(page_base, ps_, PROT_READ | PROT_WRITE);
            page_state_[idx].store(2, std::memory_order_relaxed);
        }
        // state == 2: already fully accessible, shouldn't fault
        return true;
    }

private:
    static void mprotect_region(std::uintptr_t addr, std::size_t len, int prot) {
        ::mprotect(reinterpret_cast<void*>(addr), len, prot);
    }

    // ── Global signal handler + tracker registry ───────────────────
    static void install_signal_handler() {
        std::call_once(s_once_, []() {
            struct sigaction sa{};
            sa.sa_sigaction = &sigsegv_handler;
            sa.sa_flags = SA_SIGINFO | SA_NODEFER;
            sigemptyset(&sa.sa_mask);
            ::sigaction(SIGSEGV, &sa, &s_old_handler_);
        });
    }

    static void sigsegv_handler(int sig, siginfo_t* info, void* /*uctx*/) {
        if (info && info->si_addr) {
            std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(info->si_addr);
            std::lock_guard lock(s_registry_mutex_);
            for (auto* t : s_registry_) {
                if (t->handle_fault(addr, info->si_code)) {
                    return; // Handled — faulting instruction will retry
                }
            }
        }
        // Not our fault — chain to the old handler
        if (s_old_handler_.sa_flags & SA_SIGINFO) {
            if (s_old_handler_.sa_sigaction)
                s_old_handler_.sa_sigaction(sig, info, nullptr);
        } else {
            if (s_old_handler_.sa_handler == SIG_DFL) {
                ::signal(sig, SIG_DFL);
                ::raise(sig);
            } else if (s_old_handler_.sa_handler != SIG_IGN) {
                s_old_handler_.sa_handler(sig);
            }
        }
    }

    static void register_tracker(mprotect_write_tracker* t) {
        std::lock_guard lock(s_registry_mutex_);
        s_registry_.push_back(t);
    }

    static void unregister_tracker(mprotect_write_tracker* t) {
        std::lock_guard lock(s_registry_mutex_);
        s_registry_.erase(
            std::remove(s_registry_.begin(), s_registry_.end(), t),
            s_registry_.end());
    }

    std::uintptr_t base_;
    std::size_t    size_;
    std::size_t    ps_;
    fault_callback read_fault_cb_;
    std::unique_ptr<std::atomic<std::uint8_t>[]> dirty_pages_;
    std::unique_ptr<std::atomic<std::uint8_t>[]> page_state_; // 0=NONE, 1=READ, 2=RW

    static std::once_flag                          s_once_;
    static struct sigaction                        s_old_handler_;
    static std::mutex                              s_registry_mutex_;
    static std::vector<mprotect_write_tracker*>    s_registry_;
};

std::once_flag                       mprotect_write_tracker::s_once_;
struct sigaction                     mprotect_write_tracker::s_old_handler_;
std::mutex                           mprotect_write_tracker::s_registry_mutex_;
std::vector<mprotect_write_tracker*> mprotect_write_tracker::s_registry_;

} // anonymous namespace

// ══════════════════════════════════════════════════════════════════════
//  Factory — runtime tier selection
// ══════════════════════════════════════════════════════════════════════

std::unique_ptr<write_tracker>
write_tracker::create(std::uintptr_t base, std::size_t size,
                       fault_callback read_fault_cb) {
    // Tier 3 is always available, so we always succeed (unless allocation fails).
    // We try uffd first; if it fails, the mprotect tracker takes over.

#if ZLINK_HAS_USERFAULTFD
    // Make a copy so the mprotect fallback still has a valid callback if
    // the uffd tracker construction consumes (and destroys) the original.
    auto cb_copy = read_fault_cb;
    try {
        auto tracker = std::make_unique<uffd_write_tracker>(
            base, size, /*wp_async=*/true,
            uffd_write_tracker::fault_callback(std::move(read_fault_cb)));
        if (tracker->is_valid())
            return tracker;
    } catch (...) {}
    read_fault_cb = std::move(cb_copy); // restore for the fallback
#endif

    // Tier 3: mprotect + SIGSEGV (always available on POSIX)
    try {
        return std::make_unique<mprotect_write_tracker>(
            base, size,
            mprotect_write_tracker::fault_callback(std::move(read_fault_cb)));
    } catch (...) {}

    return nullptr;
}

} // namespace zlink
