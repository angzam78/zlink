#pragma once
// zlink/memory_page_tracker.hpp — Page tracking + demand paging
//
// Tracks which pages of a memory region have been written to since the last
// sync, so flush_dirty() can send only dirty pages to the server.
//
// Also serves as the single fault handler for demand paging: read faults
// (MISSING / PROT_NONE) trigger a callback that fetches the page from the
// server via chunk_cache.
//
// Three tiers, selected at runtime by the factory:
//
//   Tier 1: userfaultfd WP + WP_ASYNC  (kernel 6.2+)
//           Write faults are auto-resolved by the kernel; we only get
//           a notification. Zero ioctls per write fault.
//
//   Tier 2: userfaultfd WP synchronous (kernel 4.11+)
//           Write faults block until the handler calls UFFDIO_WRITEPROTECT
//           to unprotect and wake. One ioctl per first-write per page.
//
//   Tier 3: mprotect + SIGSEGV (any POSIX system)
//           Fallback for containers where seccomp blocks userfaultfd(2)
//           (e.g., default Docker profile). Shadow region starts as
//           PROT_NONE; read faults fetch + install the page, write faults
//           mark dirty and grant PROT_READ|PROT_WRITE. collect_dirty()
//           re-protects dirty pages to PROT_READ.
//
// The factory tries Tier 1 → Tier 2 → Tier 3 and returns the first that
// succeeds. Tier 3 is always available on POSIX, so create() only returns
// nullptr on allocation failure.
//
// One-shot model: we track "was this page written since last sync?", not
// "how many times". A page faults once, is marked dirty, and stays writable
// until collect_dirty() re-protects it.

#include <zlink/config.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>
#include <system_error>

namespace zlink {

// A contiguous range of dirty pages, expressed as absolute addresses.
struct page_range {
    std::uintptr_t addr;
    std::size_t    len;
};

// ── Abstract write tracker ──────────────────────────────────────────────
class memory_page_tracker {
public:
    virtual ~memory_page_tracker() = default;

    // Write-protect a range of pages (call after syncing pages to server).
    // After this, writes to the range will be tracked.
    virtual std::error_code protect(std::uintptr_t addr, std::size_t len) = 0;

    // Collect all dirty page ranges since the last call, clear dirty marks,
    // and re-protect those pages for the next cycle.
    // Returns ranges that need to be pushed to the server.
    virtual std::vector<page_range> collect_dirty() = 0;

    // Factory: probes the kernel at runtime and returns the best available
    // tracker. Tries uffd WP_ASYNC (Tier 1) → uffd WP sync (Tier 2) →
    // mprotect+SIGSEGV (Tier 3). Tier 3 is always available on POSIX,
    // so create() only returns nullptr on allocation failure.
    //
    // base/size define the memory region to track. The region must be
    // mmap'd by the caller before calling this.
    //
    // read_fault_cb: called on MISSING (read) faults. The callback should
    // return the page data for the faulting address. The memory_page_tracker
    // then installs it (UFFDIO_COPY or mprotect+memcpy) and write-protects
    // it for dirty tracking.
    // Return an empty vector to signal failure (fault will be retried).
    using fault_callback = std::function<std::vector<std::byte>(std::uintptr_t addr, std::size_t len)>;

    static std::unique_ptr<memory_page_tracker> create(std::uintptr_t base,
                                                   std::size_t size,
                                                   fault_callback read_fault_cb = {});

    // Human-readable name of the active tier (for logging/debugging).
    virtual const char* backend_name() const noexcept = 0;
};

} // namespace zlink
