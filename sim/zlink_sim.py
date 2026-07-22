#!/usr/bin/env python3
"""
zlink Managed Pipeline Performance Simulation
==============================================

Demonstrates the performance difference between:
  1. Direct mount (current zlink): synchronous, single connection
  2. Multiplexed (Phase 1): 3 TCP channels, eliminates HOL blocking
  3. Write-behind (Phase 2): async push, non-blocking writes
  4. Managed pipeline (Phase 3): prefetch + write-behind + multiplexed

Based on the r3map thesis findings:
  - Direct mounts collapse past ~6ms RTT
  - Managed mounts sustain 500MB/s+ at 25ms RTT
  - Prefetch workers are the single biggest win
  - Write-behind async push is the second biggest win

This simulation models iterative CUDA workloads accessing GPU memory
over the network, measuring effective throughput at various RTTs.
Supports multiple workload profiles: ML training, compute-bound,
transfer-bound, mixed, and custom.
"""

import math
import time
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import json

# ─── Constants (from zlink config.hpp + thesis) ─────────────────────

PAGE_SIZE = 4096          # 4 KiB pages (matches chunk_cache default)
BANDWIDTH = 450.0         # MB/s at 0ms RTT (thesis: ~450 MB/s for direct mount)
PREFETCH_BANDWIDTH = 480  # MB/s (slightly higher due to sequential access)
LZ4_RATIO = 0.55          # Typical compression ratio for structured numeric data
MAX_FRAME_SIZE = 64 * 1024 * 1024  # 64 MiB (from zlink config)


class AccessPattern(Enum):
    SEQUENTIAL = "sequential"
    STRIDED = "strided"
    RANDOM = "random"


class InvalidateStrategy(Enum):
    """When to invalidate cached pages after write-back"""
    NEVER = "never"           # Pages stay cached (server updates client on next read)
    ON_WRITE = "on_write"     # Invalidate immediately after write-behind push
    ON_BARRIER = "on_barrier" # Invalidate only at barrier (sync) points


# ─── Workload Phase Definition ──────────────────────────────────────

@dataclass
class WorkloadPhase:
    """A single phase within an iterative CUDA workload iteration.

    Each phase represents a logical group of operations (e.g., data upload,
    kernel execution, result download) with specific read/write sizes and
    barrier requirements. This replaces the old fixed training-loop phases
    (forward/backward/optimizer) with a general, configurable model.

    Fields:
        name:           Human-readable label (e.g. "upload_weights", "kernel_compute")
        read_mb:        Data read from remote GPU during this phase (MB)
        write_mb:       Data written to remote GPU during this phase (MB)
        readback_mb:    Data read back from GPU to host during this phase (MB)
        kernel_rtt:     Number of kernel-launch round-trips needed (0 = no kernel)
        barrier:        Whether this phase ends with a synchronization barrier
        access_pattern: Expected page access pattern for reads in this phase
        reaccess:       Whether the same pages are re-read across iterations
                        (e.g., model weights each iteration, vs. unique input data)
    """
    name: str = "phase"
    read_mb: float = 0.0
    write_mb: float = 0.0
    readback_mb: float = 0.0
    kernel_rtt: int = 1
    barrier: bool = False
    access_pattern: AccessPattern = AccessPattern.SEQUENTIAL
    reaccess: bool = True  # Pages read again in next iteration (prefetchable)


# ─── Preset Workload Profiles ───────────────────────────────────────

ML_TRAINING = [
    # Forward pass: upload weights (reaccess), write activations (new each iter)
    WorkloadPhase(name="forward_upload", read_mb=128.0, write_mb=64.0,
                  readback_mb=0.0, kernel_rtt=1, barrier=False,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=True),
    # Forward compute: kernel + sync, then readback activations
    WorkloadPhase(name="forward_compute", read_mb=0.0, write_mb=0.0,
                  readback_mb=64.0, kernel_rtt=1, barrier=True,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=False),
    # Backward pass: read activations + weights, write gradients
    WorkloadPhase(name="backward_upload", read_mb=192.0, write_mb=64.0,
                  readback_mb=0.0, kernel_rtt=1, barrier=False,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=True),
    # Backward compute: kernel + sync, then readback gradients
    WorkloadPhase(name="backward_compute", read_mb=0.0, write_mb=0.0,
                  readback_mb=64.0, kernel_rtt=1, barrier=True,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=False),
    # Optimizer: write updated weights
    WorkloadPhase(name="optimizer_write", read_mb=0.0, write_mb=128.0,
                  readback_mb=0.0, kernel_rtt=1, barrier=True,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=True),
]

COMPUTE_BOUND = [
    # Minimal data upload (small input), heavy kernel computation
    WorkloadPhase(name="upload_input", read_mb=4.0, write_mb=0.0,
                  readback_mb=0.0, kernel_rtt=0, barrier=False,
                  access_pattern=AccessPattern.RANDOM, reaccess=False),
    # Long kernel chain (many launches, minimal data)
    WorkloadPhase(name="kernel_chain", read_mb=0.0, write_mb=0.0,
                  readback_mb=0.0, kernel_rtt=5, barrier=True,
                  access_pattern=AccessPattern.RANDOM, reaccess=False),
    # Small result readback
    WorkloadPhase(name="readback_result", read_mb=0.0, write_mb=0.0,
                  readback_mb=2.0, kernel_rtt=0, barrier=True,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=False),
]

TRANSFER_BOUND = [
    # Large data upload (e.g., batch processing, video frame upload)
    WorkloadPhase(name="bulk_upload", read_mb=256.0, write_mb=0.0,
                  readback_mb=0.0, kernel_rtt=1, barrier=False,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=False),
    # Minimal kernel (just a filter/transform)
    WorkloadPhase(name="filter_kernel", read_mb=0.0, write_mb=0.0,
                  readback_mb=0.0, kernel_rtt=1, barrier=True,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=False),
    # Large readback (processed output)
    WorkloadPhase(name="bulk_readback", read_mb=0.0, write_mb=0.0,
                  readback_mb=256.0, kernel_rtt=0, barrier=True,
                  access_pattern=AccessPattern.SEQUENTIAL, reaccess=False),
]

MIXED_WORKLOAD = [
    # Upload working set (reaccess across iterations)
    WorkloadPhase(name="upload_working_set", read_mb=96.0, write_mb=0.0,
                  readback_mb=0.0, kernel_rtt=1, barrier=False,
                  access_pattern=AccessPattern.STRIDED, reaccess=True),
    # Compute + write intermediate results
    WorkloadPhase(name="compute_intermediate", read_mb=0.0, write_mb=48.0,
                  readback_mb=0.0, kernel_rtt=2, barrier=False,
                  access_pattern=AccessPattern.STRIDED, reaccess=False),
    # Read back intermediate + barrier sync
    WorkloadPhase(name="readback_intermediate", read_mb=48.0, write_mb=0.0,
                  readback_mb=48.0, kernel_rtt=1, barrier=True,
                  access_pattern=AccessPattern.STRIDED, reaccess=False),
    # Update working set (write-behind)
    WorkloadPhase(name="update_working_set", read_mb=0.0, write_mb=96.0,
                  readback_mb=0.0, kernel_rtt=1, barrier=True,
                  access_pattern=AccessPattern.STRIDED, reaccess=True),
]

# ─── Simulation Configuration ───────────────────────────────────────

@dataclass
class SimConfig:
    """Simulation configuration for any iterative CUDA workload"""
    rtt_ms: float = 10.0
    workload: List[WorkloadPhase] = field(default_factory=lambda: ML_TRAINING)
    iterations: int = 20
    pipeline_depth: int = 8
    compression: bool = True
    page_size: int = PAGE_SIZE
    invalidate_strategy: InvalidateStrategy = InvalidateStrategy.ON_WRITE
    setup_rtt_count: int = 3  # Number of setup barrier RTTs (first iteration only)

    @property
    def total_data_per_iter(self) -> float:
        """Total data moved per iteration (MB)"""
        return sum(p.read_mb + p.write_mb + p.readback_mb for p in self.workload)

    @property
    def reaccess_regions(self) -> List[Tuple[int, int]]:
        """Page ranges that are reaccessed across iterations (prefetch candidates).

        Returns list of (start_page, end_page) tuples for each phase where
        reaccess=True. Pages are assigned sequentially across phases.
        """
        regions = []
        page_offset = 0
        for phase in self.workload:
            read_pages = int(phase.read_mb * 1024 * 1024 / self.page_size)
            if phase.reaccess and read_pages > 0:
                regions.append((page_offset, page_offset + read_pages))
            page_offset += read_pages
            write_pages = int(phase.write_mb * 1024 * 1024 / self.page_size)
            page_offset += write_pages
            readback_pages = int(phase.readback_mb * 1024 * 1024 / self.page_size)
            page_offset += readback_pages
        return regions


@dataclass
class IterationResult:
    """Result of one iteration of a CUDA workload"""
    iteration: int
    time_ms: float
    throughput_mb_s: float
    barrier_count: int
    pipeline_round_trips: int
    cache_hits: int
    cache_misses: int
    write_behind_ops: int
    prefetch_ops: int


@dataclass
class SimulationResult:
    """Result of a complete simulation run"""
    mode: str
    rtt_ms: float
    iterations: List[IterationResult]
    avg_throughput_mb_s: float = 0.0
    avg_iteration_ms: float = 0.0
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_prefetch_ops: int = 0
    total_write_behind_ops: int = 0

    def __post_init__(self):
        if self.iterations:
            self.avg_throughput_mb_s = statistics.mean(
                r.throughput_mb_s for r in self.iterations)
            self.avg_iteration_ms = statistics.mean(
                r.time_ms for r in self.iterations)
            self.total_cache_hits = sum(r.cache_hits for r in self.iterations)
            self.total_cache_misses = sum(r.cache_misses for r in self.iterations)
            self.total_prefetch_ops = sum(r.prefetch_ops for r in self.iterations)
            self.total_write_behind_ops = sum(r.write_behind_ops for r in self.iterations)


# ─── Page Cache Simulation ──────────────────────────────────────────

class PageCache:
    """Simulates chunk_cache with optional prefetch and dirty tracking"""

    def __init__(self, capacity_pages: int = 100000):
        self.local_pages: set = set()       # Pages available locally
        self.dirty_pages: set = set()       # Pages that differ from remote
        self.prefetch_in_flight: set = set()  # Pages being prefetched
        self.capacity = capacity_pages
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0

    def is_local(self, page: int) -> bool:
        return page in self.local_pages

    def read_page(self, page: int) -> bool:
        """Returns True if cache hit, False if miss (needs network fetch)"""
        if page in self.local_pages:
            self.hits += 1
            return True
        self.misses += 1
        # Simulate fetch: mark as local for next time
        self.local_pages.add(page)
        return False

    def write_page(self, page: int):
        """Write to page: mark as local and dirty"""
        self.local_pages.add(page)
        self.dirty_pages.add(page)

    def prefetch_pages(self, pages: List[int]):
        """Mark pages as being prefetched (will be local soon)"""
        for p in pages:
            self.prefetch_in_flight.add(p)
            self.local_pages.add(p)
            self.prefetch_hits += 1

    def invalidate(self, pages: List[int]):
        """Mark pages as non-local (e.g., after remote write-back)"""
        for p in pages:
            self.local_pages.discard(p)

    def clear_dirty(self, pages: List[int]):
        """Mark pages as clean after push"""
        for p in pages:
            self.dirty_pages.discard(p)

    def get_dirty_pages(self) -> set:
        return set(self.dirty_pages)

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0


# ─── Access Pattern Detector ────────────────────────────────────────

class PatternDetector:
    """Simulates prefetch_worker's access pattern detection"""

    def __init__(self, lookahead: int = 8):
        self.lookahead = lookahead
        self.recent_pages: List[int] = []
        self.detected_pattern = AccessPattern.RANDOM
        self.stride = 0
        self.confidence = 0

    def record_access(self, page: int):
        """Record a page access and update pattern detection"""
        if self.recent_pages:
            last = self.recent_pages[-1]
            delta = page - last

            if delta == 1:
                if self.detected_pattern == AccessPattern.SEQUENTIAL:
                    self.confidence += 1
                else:
                    self.detected_pattern = AccessPattern.SEQUENTIAL
                    self.confidence = 1
                    self.stride = 1
            elif delta > 1 and delta < 256:
                if self.detected_pattern == AccessPattern.STRIDED and delta == self.stride:
                    self.confidence += 1
                else:
                    self.detected_pattern = AccessPattern.STRIDED
                    self.stride = delta
                    self.confidence = 1
            else:
                self.confidence = max(0, self.confidence - 2)
                if self.confidence < 3:
                    self.detected_pattern = AccessPattern.RANDOM

        self.recent_pages.append(page)
        if len(self.recent_pages) > 100:
            self.recent_pages = self.recent_pages[-50:]

    def predict_next_pages(self, current_page: int) -> List[int]:
        """Predict which pages will be accessed next"""
        if self.confidence < 3:
            return []

        if self.detected_pattern == AccessPattern.SEQUENTIAL:
            return list(range(current_page + 1, current_page + 1 + self.lookahead))
        elif self.detected_pattern == AccessPattern.STRIDED:
            return [current_page + self.stride * i for i in range(1, self.lookahead + 1)]
        else:
            return []


# ─── Transfer Time Calculations ─────────────────────────────────────

def transfer_time_ms(size_mb: float, rtt_ms: float, bandwidth_mb_s: float = BANDWIDTH,
                     compression: bool = True) -> float:
    """Calculate time to transfer data over network"""
    actual_size = size_mb * (LZ4_RATIO if compression and size_mb > 0.004 else 1.0)
    if actual_size <= 0:
        return rtt_ms
    transfer_time = (actual_size / bandwidth_mb_s) * 1000  # Convert to ms
    return rtt_ms + transfer_time  # 1 RTT for request/response + transfer time


def pipeline_batch_time_ms(total_data_mb: float, rtt_ms: float,
                           num_calls: int, bandwidth_mb_s: float = BANDWIDTH,
                           compression: bool = True) -> float:
    """Calculate time for a pipeline_mem frame (1 round-trip + data)"""
    actual_data = total_data_mb * (LZ4_RATIO if compression and total_data_mb > 0.004 else 1.0)
    transfer_time = (actual_data / bandwidth_mb_s) * 1000
    return rtt_ms + transfer_time


# ─── Simulation Modes ───────────────────────────────────────────────

def simulate_direct_mount(config: SimConfig) -> SimulationResult:
    """
    Current zlink: single TCP connection, synchronous, no prefetch.

    Every cuMemcpyHtoD is a barrier (must wait for dev_ptr).
    Every cuMemcpyDtoH is a readback (flush + sync).
    No caching between iterations.
    """
    results = []
    total_data_per_iter = config.total_data_per_iter

    for i in range(config.iterations):
        iter_time = 0.0
        barriers = 0
        round_trips = 0

        # Setup (first iteration only for context/alloc calls)
        if i == 0:
            for _ in range(config.setup_rtt_count):
                iter_time += config.rtt_ms
                barriers += 1
                round_trips += 1

        # Execute each workload phase synchronously
        for phase in config.workload:
            # Read phase: synchronous transfer
            if phase.read_mb > 0:
                iter_time += transfer_time_ms(phase.read_mb, config.rtt_ms,
                                               compression=config.compression)
                barriers += 1
                round_trips += 1

            # Write phase: synchronous transfer
            if phase.write_mb > 0:
                iter_time += transfer_time_ms(phase.write_mb, config.rtt_ms,
                                               compression=config.compression)
                barriers += 1
                round_trips += 1

            # Kernel launches
            if phase.kernel_rtt > 0:
                iter_time += config.rtt_ms * phase.kernel_rtt
                round_trips += phase.kernel_rtt

            # Readback phase: synchronous transfer
            if phase.readback_mb > 0:
                iter_time += transfer_time_ms(phase.readback_mb, config.rtt_ms,
                                               compression=config.compression)
                barriers += 1
                round_trips += 1

        throughput = (total_data_per_iter / iter_time) * 1000 if iter_time > 0 else 0

        results.append(IterationResult(
            iteration=i, time_ms=iter_time, throughput_mb_s=throughput,
            barrier_count=barriers, pipeline_round_trips=round_trips,
            cache_hits=0, cache_misses=0, write_behind_ops=0, prefetch_ops=0
        ))

    return SimulationResult(mode="direct_mount", rtt_ms=config.rtt_ms, iterations=results)


def simulate_multiplexed(config: SimConfig) -> SimulationResult:
    """
    Phase 1: 3 TCP channels (RPC, bulk, prefetch).
    Eliminates HOL blocking: bulk transfers don't stall RPC calls.
    But still synchronous — every transfer still costs RTT.

    Key improvement: bulk data and kernel RPC can overlap on separate channels.
    """
    results = []
    total_data_per_iter = config.total_data_per_iter

    for i in range(config.iterations):
        iter_time = 0.0
        barriers = 0
        round_trips = 0

        # Setup (first iteration only)
        if i == 0:
            for _ in range(config.setup_rtt_count):
                iter_time += config.rtt_ms
                barriers += 1
                round_trips += 1

        # Execute each workload phase with channel overlap
        for phase in config.workload:
            # Calculate data transfer times for this phase
            data_ops = []
            if phase.read_mb > 0:
                data_ops.append(transfer_time_ms(phase.read_mb, config.rtt_ms,
                                                  compression=config.compression))
            if phase.write_mb > 0:
                data_ops.append(transfer_time_ms(phase.write_mb, config.rtt_ms,
                                                  compression=config.compression))
            if phase.readback_mb > 0:
                data_ops.append(transfer_time_ms(phase.readback_mb, config.rtt_ms,
                                                  compression=config.compression))

            kernel_time = config.rtt_ms * phase.kernel_rtt if phase.kernel_rtt > 0 else 0

            # Multiplexed overlap: data on bulk channel, kernels on RPC channel
            if phase.barrier:
                # Barrier phase: can't overlap, must complete everything
                iter_time += sum(data_ops) + kernel_time
                round_trips += len(data_ops) + phase.kernel_rtt
                barriers += len(data_ops)
            else:
                # Non-barrier phase: overlap bulk transfers with kernel RPCs
                if data_ops and kernel_time > 0:
                    # Parallel: take max of data transfer vs kernel time
                    iter_time += max(sum(data_ops), kernel_time)
                    round_trips += max(len(data_ops), phase.kernel_rtt)
                elif data_ops:
                    iter_time += sum(data_ops)
                    round_trips += len(data_ops)
                elif kernel_time > 0:
                    iter_time += kernel_time
                    round_trips += phase.kernel_rtt

        throughput = (total_data_per_iter / iter_time) * 1000 if iter_time > 0 else 0

        results.append(IterationResult(
            iteration=i, time_ms=iter_time, throughput_mb_s=throughput,
            barrier_count=barriers, pipeline_round_trips=round_trips,
            cache_hits=0, cache_misses=0, write_behind_ops=0, prefetch_ops=0
        ))

    return SimulationResult(mode="multiplexed_3ch", rtt_ms=config.rtt_ms, iterations=results)


def simulate_write_behind(config: SimConfig) -> SimulationResult:
    """
    Phase 2: Write-behind buffer on bulk channel.
    cuMemcpyHtoD returns immediately; pusher thread streams data.
    Barrier calls drain the buffer before executing.
    """
    results = []
    total_data_per_iter = config.total_data_per_iter
    buffer_time = 0.1  # ms to buffer data locally

    for i in range(config.iterations):
        iter_time = 0.0
        barriers = 0
        round_trips = 0
        wb_ops = 0

        # Setup (first iteration only)
        if i == 0:
            for _ in range(config.setup_rtt_count):
                iter_time += config.rtt_ms
                barriers += 1
                round_trips += 1

        for phase in config.workload:
            # Read: still synchronous (no write-behind benefit for reads)
            if phase.read_mb > 0:
                iter_time += transfer_time_ms(phase.read_mb, config.rtt_ms,
                                               compression=config.compression)
                round_trips += 1

            # Write: write-behind! Returns immediately
            if phase.write_mb > 0:
                iter_time += buffer_time
                wb_ops += 1

            # Readback: still synchronous
            if phase.readback_mb > 0:
                iter_time += transfer_time_ms(phase.readback_mb, config.rtt_ms,
                                               compression=config.compression)
                round_trips += 1

            # Kernel + barrier: need to drain write-behind first
            if phase.barrier:
                # Drain time for pending write-behind data
                # Find write_mb in this phase and any previous undrained phases
                drain_mb = phase.write_mb  # Drain this phase's writes
                if drain_mb > 0:
                    drain_time = transfer_time_ms(drain_mb, config.rtt_ms,
                                                   compression=config.compression)
                    # Overlap drain + kernel RPC
                    kernel_time = config.rtt_ms * phase.kernel_rtt if phase.kernel_rtt > 0 else 0
                    iter_time += max(drain_time, kernel_time) if kernel_time > 0 else drain_time
                    round_trips += 1
                    barriers += 1
                elif phase.kernel_rtt > 0:
                    iter_time += config.rtt_ms * phase.kernel_rtt
                    round_trips += phase.kernel_rtt
                    barriers += 1
            elif phase.kernel_rtt > 0:
                # Non-barrier kernel: can proceed while write-behind pushes
                iter_time += config.rtt_ms * phase.kernel_rtt
                round_trips += phase.kernel_rtt

        throughput = (total_data_per_iter / iter_time) * 1000 if iter_time > 0 else 0

        results.append(IterationResult(
            iteration=i, time_ms=iter_time, throughput_mb_s=throughput,
            barrier_count=barriers, pipeline_round_trips=round_trips,
            cache_hits=0, cache_misses=0, write_behind_ops=wb_ops, prefetch_ops=0
        ))

    return SimulationResult(mode="write_behind", rtt_ms=config.rtt_ms, iterations=results)


def simulate_managed_pipeline(config: SimConfig) -> SimulationResult:
    """
    Phase 3: Full managed pipeline with prefetch + write-behind + multiplexed.

    First iteration: cold cache, prefetch starts.
    Subsequent iterations: reaccess regions hit cache (0ms), writes are write-behind.
    Pipeline batches all kernels + sync into 1 round-trip per barrier phase.
    """
    results = []
    total_data_per_iter = config.total_data_per_iter
    cache = PageCache()
    detector = PatternDetector(lookahead=8)
    buffer_time = 0.1  # ms to buffer data locally

    # Compute page ranges for each phase's data
    page_ranges = []  # (phase, read_start, read_end, write_start, write_end, rb_start, rb_end)
    page_offset = 0
    for phase in config.workload:
        read_pages = int(phase.read_mb * 1024 * 1024 / config.page_size) if phase.read_mb > 0 else 0
        write_pages = int(phase.write_mb * 1024 * 1024 / config.page_size) if phase.write_mb > 0 else 0
        rb_pages = int(phase.readback_mb * 1024 * 1024 / config.page_size) if phase.readback_mb > 0 else 0
        ranges = {
            'read': (page_offset, page_offset + read_pages),
            'write': (page_offset + read_pages, page_offset + read_pages + write_pages),
            'readback': (page_offset + read_pages + write_pages,
                         page_offset + read_pages + write_pages + rb_pages),
        }
        page_ranges.append((phase, ranges))
        page_offset += read_pages + write_pages + rb_pages

    for i in range(config.iterations):
        iter_time = 0.0
        barriers = 0
        round_trips = 0
        wb_ops = 0
        prefetch_ops = 0
        total_hits = 0
        total_misses = 0

        # Setup (first iteration only; cached after)
        if i == 0:
            for _ in range(config.setup_rtt_count):
                iter_time += config.rtt_ms
                barriers += 1
                round_trips += 1

        # ── Predictive prefetch for reaccess regions ──
        if detector.confidence >= 3:
            for start, end in config.reaccess_regions:
                predicted = detector.predict_next_pages(start)
                predicted_in_range = [p for p in predicted if start <= p < end]
                if predicted_in_range:
                    cache.prefetch_pages(predicted_in_range)
                    prefetch_ops += len(predicted_in_range)

        # ── Execute each workload phase ──
        for phase_idx, (phase, ranges) in enumerate(page_ranges):
            read_start, read_end = ranges['read']
            write_start, write_end = ranges['write']
            rb_start, rb_end = ranges['readback']

            # Read phase: check cache
            if phase.read_mb > 0 and read_end > read_start:
                hits = 0
                misses = 0
                for p in range(read_start, read_end):
                    detector.record_access(p)
                    if cache.read_page(p):
                        hits += 1
                    else:
                        misses += 1

                total_hits += hits
                total_misses += misses

                if misses > 0:
                    miss_mb = (misses * config.page_size) / (1024 * 1024)
                    iter_time += transfer_time_ms(miss_mb, config.rtt_ms,
                                                   compression=config.compression)
                    round_trips += 1
                else:
                    iter_time += 0.01  # Local memcpy (~10 microseconds)

            # Write phase: write-behind
            if phase.write_mb > 0 and write_end > write_start:
                for p in range(write_start, write_end):
                    cache.write_page(p)
                iter_time += buffer_time
                wb_ops += 1

            # Readback phase: check cache
            if phase.readback_mb > 0 and rb_end > rb_start:
                rb_hits = sum(1 for p in range(rb_start, rb_end) if cache.is_local(p))
                rb_misses = (rb_end - rb_start) - rb_hits
                total_hits += rb_hits
                total_misses += rb_misses

                if rb_misses > 0:
                    miss_mb = (rb_misses * config.page_size) / (1024 * 1024)
                    iter_time += transfer_time_ms(miss_mb, config.rtt_ms,
                                                   compression=config.compression)
                    round_trips += 1
                else:
                    iter_time += 0.01

            # Kernel + barrier
            if phase.barrier:
                # Drain write-behind for this phase
                drain_mb = phase.write_mb
                if drain_mb > 0:
                    drain_time = transfer_time_ms(drain_mb, config.rtt_ms,
                                                   compression=config.compression)
                    kernel_time = config.rtt_ms * phase.kernel_rtt if phase.kernel_rtt > 0 else 0
                    iter_time += max(drain_time * 0.5, kernel_time) if kernel_time > 0 else drain_time * 0.5
                    round_trips += 1
                    barriers += 1
                elif phase.kernel_rtt > 0:
                    iter_time += config.rtt_ms * phase.kernel_rtt
                    round_trips += phase.kernel_rtt
                    barriers += 1

                # Apply cache invalidation strategy
                if config.invalidate_strategy == InvalidateStrategy.ON_WRITE:
                    # Invalidate reaccess read regions that were updated via write-behind
                    # (e.g., weights that were written back by optimizer)
                    for p_idx, (p, r) in enumerate(page_ranges):
                        if p.reaccess and p.write_mb > 0:
                            # The write pages of a reaccess phase become stale on server
                            ws, we = r['write']
                            cache.invalidate(list(range(ws, we)))
                elif config.invalidate_strategy == InvalidateStrategy.ON_BARRIER:
                    # Only invalidate at explicit barriers
                    for p_idx, (p, r) in enumerate(page_ranges):
                        if p.reaccess and p.write_mb > 0:
                            ws, we = r['write']
                            cache.invalidate(list(range(ws, we)))
                # NEVER: pages stay cached, server will push updates

            elif phase.kernel_rtt > 0:
                # Non-barrier kernel: can proceed during write-behind
                iter_time += config.rtt_ms * phase.kernel_rtt
                round_trips += phase.kernel_rtt

        throughput = (total_data_per_iter / iter_time) * 1000 if iter_time > 0 else 0

        results.append(IterationResult(
            iteration=i, time_ms=iter_time, throughput_mb_s=throughput,
            barrier_count=barriers, pipeline_round_trips=round_trips,
            cache_hits=total_hits, cache_misses=total_misses,
            write_behind_ops=wb_ops, prefetch_ops=prefetch_ops
        ))

    return SimulationResult(mode="managed_pipeline", rtt_ms=config.rtt_ms,
                           iterations=results)


# ─── Main Simulation Runner ─────────────────────────────────────────

WORKLOAD_PRESETS = {
    "ml_training": ML_TRAINING,
    "compute_bound": COMPUTE_BOUND,
    "transfer_bound": TRANSFER_BOUND,
    "mixed": MIXED_WORKLOAD,
}


def run_sweep(workload_name: str = "ml_training"):
    """Run all 4 modes across a range of RTTs for a given workload profile"""

    workload = WORKLOAD_PRESETS[workload_name]
    rtts = [0, 1, 2, 5, 6, 10, 15, 25, 50, 100]
    modes = [
        ("direct_mount", simulate_direct_mount),
        ("multiplexed_3ch", simulate_multiplexed),
        ("write_behind", simulate_write_behind),
        ("managed_pipeline", simulate_managed_pipeline),
    ]

    print("=" * 90)
    print(f"zlink Managed Pipeline Performance Simulation — {workload_name}")
    print("Based on r3map thesis findings (pojntfx.github.io/networked-linux-memsync)")
    print("=" * 90)
    print()

    # Print workload profile
    print("WORKLOAD PROFILE:")
    print(f"  {'Phase':>25} | {'Read MB':>8} | {'Write MB':>8} | {'Readback MB':>8} | {'Kernel RTTs':>10} | {'Barrier':>7} | {'Reaccess':>8}")
    print("-" * 90)
    for phase in workload:
        print(f"  {phase.name:>25} | {phase.read_mb:>8.0f} | {phase.write_mb:>8.0f} | "
              f"{phase.readback_mb:>8.0f} | {phase.kernel_rtt:>10} | {str(phase.barrier):>7} | {str(phase.reaccess):>8}")
    print()

    # ── Table 1: Throughput vs RTT ────────────────────────────────
    print("TABLE 1: Effective Throughput (MB/s) vs RTT")
    print("-" * 90)
    header = f"{'RTT (ms)':>10}"
    for mode_name, _ in modes:
        header += f" | {mode_name:>18}"
    print(header)
    print("-" * 90)

    all_results = {}
    for rtt in rtts:
        row = f"{rtt:>10}"
        for mode_name, sim_fn in modes:
            cfg = SimConfig(rtt_ms=rtt, workload=workload)
            result = sim_fn(cfg)
            all_results[(mode_name, rtt)] = result
            row += f" | {result.avg_throughput_mb_s:>18.1f}"
        print(row)

    print()

    # ── Table 2: Iteration Time vs RTT ────────────────────────────
    print("TABLE 2: Average Iteration Time (ms) vs RTT")
    print("-" * 90)
    header = f"{'RTT (ms)':>10}"
    for mode_name, _ in modes:
        header += f" | {mode_name:>18}"
    print(header)
    print("-" * 90)

    for rtt in rtts:
        row = f"{rtt:>10}"
        for mode_name, _ in modes:
            result = all_results[(mode_name, rtt)]
            row += f" | {result.avg_iteration_ms:>18.1f}"
        print(row)

    print()

    # ── Table 3: Speedup vs Direct Mount ──────────────────────────
    print("TABLE 3: Speedup Factor vs Direct Mount")
    print("-" * 90)
    header = f"{'RTT (ms)':>10}"
    for mode_name, _ in modes:
        if mode_name != "direct_mount":
            header += f" | {mode_name:>18}"
    print(header)
    print("-" * 90)

    for rtt in rtts:
        baseline = all_results[("direct_mount", rtt)].avg_throughput_mb_s
        row = f"{rtt:>10}"
        for mode_name, _ in modes:
            if mode_name != "direct_mount":
                val = all_results[(mode_name, rtt)].avg_throughput_mb_s
                speedup = val / baseline if baseline > 0 else float('inf')
                row += f" | {speedup:>18.1f}x"
        print(row)

    print()

    # ── Detailed breakdown at 10ms RTT ────────────────────────────
    print("=" * 90)
    print("DETAILED BREAKDOWN at 10ms RTT (typical WAN)")
    print("=" * 90)

    for mode_name, _ in modes:
        result = all_results[(mode_name, 10)]
        print(f"\n  {mode_name}:")
        print(f"    Avg throughput:     {result.avg_throughput_mb_s:>8.1f} MB/s")
        print(f"    Avg iteration time: {result.avg_iteration_ms:>8.1f} ms")
        print(f"    Cache hits:         {result.total_cache_hits:>8d}")
        print(f"    Cache misses:       {result.total_cache_misses:>8d}")
        print(f"    Prefetch ops:       {result.total_prefetch_ops:>8d}")
        print(f"    Write-behind ops:   {result.total_write_behind_ops:>8d}")

    # ── Per-iteration detail for managed_pipeline at 10ms ─────────
    print()
    print("=" * 90)
    print("MANAGED PIPELINE: Per-Iteration Detail at 10ms RTT")
    print("(Shows cache warmup: first iteration cold, subsequent hot)")
    print("=" * 90)

    cfg = SimConfig(rtt_ms=10, workload=workload)
    result = simulate_managed_pipeline(cfg)
    print(f"\n{'Iter':>4} | {'Time(ms)':>10} | {'MB/s':>8} | {'Hits':>6} | {'Misses':>6} | {'WB':>4} | {'PF':>4}")
    print("-" * 60)
    for r in result.iterations:
        print(f"{r.iteration:>4} | {r.time_ms:>10.1f} | {r.throughput_mb_s:>8.1f} | "
              f"{r.cache_hits:>6} | {r.cache_misses:>6} | {r.write_behind_ops:>4} | {r.prefetch_ops:>4}")

    # ── Per-iteration detail for managed_pipeline at 25ms ─────────
    print()
    print("=" * 90)
    print("MANAGED PIPELINE: Per-Iteration Detail at 25ms RTT")
    print("(Thesis benchmark point: managed mount sustains 500MB/s)")
    print("=" * 90)

    cfg = SimConfig(rtt_ms=25, workload=workload)
    result = simulate_managed_pipeline(cfg)
    print(f"\n{'Iter':>4} | {'Time(ms)':>10} | {'MB/s':>8} | {'Hits':>6} | {'Misses':>6} | {'WB':>4} | {'PF':>4}")
    print("-" * 60)
    for r in result.iterations:
        print(f"{r.iteration:>4} | {r.time_ms:>10.1f} | {r.throughput_mb_s:>8.1f} | "
              f"{r.cache_hits:>6} | {r.cache_misses:>6} | {r.write_behind_ops:>4} | {r.prefetch_ops:>4}")

    # ── Extended iteration: 100 iterations at 10ms ────────────────
    print()
    print("=" * 90)
    print("EXTENDED ITERATIVE WORKLOAD: 100 iterations at 10ms RTT")
    print("(Shows steady-state performance after cache warmup)")
    print("=" * 90)

    cfg = SimConfig(rtt_ms=10, workload=workload, iterations=100)

    for mode_name, sim_fn in modes:
        result = sim_fn(cfg)
        # Skip first 3 iterations (warmup), show steady-state
        steady = result.iterations[3:]
        if steady:
            steady_throughput = statistics.mean(r.throughput_mb_s for r in steady)
            steady_time = statistics.mean(r.time_ms for r in steady)
            print(f"\n  {mode_name}:")
            print(f"    Steady-state throughput: {steady_throughput:>8.1f} MB/s")
            print(f"    Steady-state iter time:  {steady_time:>8.1f} ms")
            if result.total_cache_hits + result.total_cache_misses > 0:
                hit_rate = result.total_cache_hits / (result.total_cache_hits + result.total_cache_misses) * 100
                print(f"    Cache hit rate:          {hit_rate:>8.1f} %")

    # ── JSON export for charting ──────────────────────────────────
    print()
    print("=" * 90)
    print("JSON DATA (for external charting)")
    print("=" * 90)

    chart_data = {
        "title": f"zlink Managed Pipeline Simulation — {workload_name}",
        "x_label": "RTT (ms)",
        "y_label": "Throughput (MB/s)",
        "rtts": rtts,
        "series": []
    }

    for mode_name, _ in modes:
        series = {
            "name": mode_name,
            "values": [all_results[(mode_name, rtt)].avg_throughput_mb_s for rtt in rtts]
        }
        chart_data["series"].append(series)

    print(json.dumps(chart_data, indent=2))

    return all_results


def run_all_workloads():
    """Run simulation for all workload presets and generate comparison charts"""
    all_sweep_data = {}

    for workload_name in WORKLOAD_PRESETS:
        print()
        print("=" * 90)
        print(f"  SIMULATING: {workload_name}")
        print("=" * 90)
        results = run_sweep(workload_name)
        all_sweep_data[workload_name] = results

    # ── Cross-workload comparison at 10ms RTT ─────────────────────
    print()
    print("=" * 90)
    print("CROSS-WORKLOAD COMPARISON at 10ms RTT")
    print("(Managed pipeline speedup factor vs direct mount)")
    print("=" * 90)
    print(f"\n{'Workload':>20} | {'Direct MB/s':>12} | {'Managed MB/s':>12} | {'Speedup':>10}")
    print("-" * 70)

    for wl_name in WORKLOAD_PRESETS:
        results = all_sweep_data[wl_name]
        direct_tput = results[("direct_mount", 10)].avg_throughput_mb_s
        managed_tput = results[("managed_pipeline", 10)].avg_throughput_mb_s
        speedup = managed_tput / direct_tput if direct_tput > 0 else float('inf')
        print(f"{wl_name:>20} | {direct_tput:>12.1f} | {managed_tput:>12.1f} | {speedup:>10.1f}x")

    # ── Cross-workload comparison at 25ms RTT ─────────────────────
    print()
    print("=" * 90)
    print("CROSS-WORKLOAD COMPARISON at 25ms RTT")
    print("(Managed pipeline speedup factor vs direct mount)")
    print("=" * 90)
    print(f"\n{'Workload':>20} | {'Direct MB/s':>12} | {'Managed MB/s':>12} | {'Speedup':>10}")
    print("-" * 70)

    for wl_name in WORKLOAD_PRESETS:
        results = all_sweep_data[wl_name]
        direct_tput = results[("direct_mount", 25)].avg_throughput_mb_s
        managed_tput = results[("managed_pipeline", 25)].avg_throughput_mb_s
        speedup = managed_tput / direct_tput if direct_tput > 0 else float('inf')
        print(f"{wl_name:>20} | {direct_tput:>12.1f} | {managed_tput:>12.1f} | {speedup:>10.1f}x")

    return all_sweep_data


if __name__ == "__main__":
    # Default: run ML training workload (backward compatible)
    # For full cross-workload comparison: run_all_workloads()
    run_all_workloads()
