#!/usr/bin/env python3
"""
Generate updated charts for the workload-agnostic zlink simulation.
Produces 6 charts:
  1. Throughput vs RTT (all 4 workloads, managed pipeline only)
  2. Speedup vs RTT (all 4 workloads)
  3. Cross-workload comparison bar chart at 10ms and 25ms
  4. ML training throughput vs RTT (all 4 modes, like original)
  5. ML training speedup factor vs direct mount
  6. ML training iteration time vs RTT
"""

import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Font setup
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Import simulation
sys.path.insert(0, '/home/z/my-project/scripts')
from zlink_sim import (
    SimConfig, simulate_direct_mount, simulate_multiplexed,
    simulate_write_behind, simulate_managed_pipeline,
    WORKLOAD_PRESETS, ML_TRAINING, COMPUTE_BOUND, TRANSFER_BOUND, MIXED_WORKLOAD,
)

OUTPUT_DIR = '/home/z/my-project/download'

rtts = [0, 1, 2, 5, 6, 10, 15, 25, 50, 100]
modes = [
    ("direct_mount", simulate_direct_mount),
    ("multiplexed_3ch", simulate_multiplexed),
    ("write_behind", simulate_write_behind),
    ("managed_pipeline", simulate_managed_pipeline),
]

# Run all simulations
all_data = {}
for wl_name, wl_phases in WORKLOAD_PRESETS.items():
    for mode_name, sim_fn in modes:
        for rtt in rtts:
            cfg = SimConfig(rtt_ms=rtt, workload=wl_phases)
            result = sim_fn(cfg)
            all_data[(wl_name, mode_name, rtt)] = result

# Color scheme
COLORS = {
    'direct_mount': '#e74c3c',
    'multiplexed_3ch': '#f39c12',
    'write_behind': '#2ecc71',
    'managed_pipeline': '#3498db',
}
WL_COLORS = {
    'ml_training': '#3498db',
    'compute_bound': '#e74c3c',
    'transfer_bound': '#2ecc71',
    'mixed': '#f39c12',
}

# ─── Chart 1: Throughput vs RTT — all workloads, managed pipeline ──
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for wl_name in WORKLOAD_PRESETS:
    values = [all_data[(wl_name, 'managed_pipeline', rtt)].avg_throughput_mb_s for rtt in rtts]
    ax.plot(rtts, values, marker='o', linewidth=2, color=WL_COLORS[wl_name], label=wl_name)
    # Also plot direct mount for comparison
    direct_values = [all_data[(wl_name, 'direct_mount', rtt)].avg_throughput_mb_s for rtt in rtts]
    ax.plot(rtts, direct_values, marker='x', linewidth=1, linestyle='--',
            color=WL_COLORS[wl_name], alpha=0.4, label=f'{wl_name} (direct)')

ax.set_xlabel('RTT (ms)')
ax.set_ylabel('Throughput (MB/s)')
ax.set_title('zlink Managed Pipeline: Throughput vs RTT — All Workloads')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
fig.savefig(f'{OUTPUT_DIR}/zlink-all-workloads-throughput.png', dpi=150)
plt.close(fig)

# ─── Chart 2: Speedup vs RTT — all workloads ──
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for wl_name in WORKLOAD_PRESETS:
    speedups = []
    for rtt in rtts:
        direct = all_data[(wl_name, 'direct_mount', rtt)].avg_throughput_mb_s
        managed = all_data[(wl_name, 'managed_pipeline', rtt)].avg_throughput_mb_s
        speedups.append(managed / direct if direct > 0 else 0)
    ax.plot(rtts, speedups, marker='o', linewidth=2, color=WL_COLORS[wl_name], label=wl_name)

ax.set_xlabel('RTT (ms)')
ax.set_ylabel('Speedup Factor (x)')
ax.set_title('zlink Managed Pipeline: Speedup vs Direct Mount — All Workloads')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
fig.savefig(f'{OUTPUT_DIR}/zlink-all-workloads-speedup.png', dpi=150)
plt.close(fig)

# ─── Chart 3: Cross-workload bar chart at 10ms and 25ms ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
wl_names = list(WORKLOAD_PRESETS.keys())
for idx, (target_rtt, title) in enumerate([(10, 'at 10ms RTT'), (25, 'at 25ms RTT')]):
    ax = axes[idx]
    direct_vals = [all_data[(wl, 'direct_mount', target_rtt)].avg_throughput_mb_s for wl in wl_names]
    managed_vals = [all_data[(wl, 'managed_pipeline', target_rtt)].avg_throughput_mb_s for wl in wl_names]
    speedup_vals = [m/d if d > 0 else 0 for m, d in zip(managed_vals, direct_vals)]

    x = range(len(wl_names))
    width = 0.35
    bars1 = ax.bar([xi - width/2 for xi in x], direct_vals, width=width,
                   color='#e74c3c', label='Direct mount')
    bars2 = ax.bar([xi + width/2 for xi in x], managed_vals, width=width,
                   color='#3498db', label='Managed pipeline')

    # Add speedup annotations
    for xi, sv in zip(x, speedup_vals):
        ax.annotate(f'{sv:.1f}x', xy=(xi + width/2, managed_vals[xi]),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=8, color='#2c3e50')

    ax.set_xticks(list(x))
    ax.set_xticklabels(wl_names, fontsize=9)
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title(f'Cross-Workload Comparison {title}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

fig.savefig(f'{OUTPUT_DIR}/zlink-cross-workload-comparison.png', dpi=150)
plt.close(fig)

# ─── Chart 4: ML training throughput vs RTT — all 4 modes ──
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for mode_name, _ in modes:
    values = [all_data[('ml_training', mode_name, rtt)].avg_throughput_mb_s for rtt in rtts]
    ax.plot(rtts, values, marker='o', linewidth=2, color=COLORS[mode_name], label=mode_name)

ax.set_xlabel('RTT (ms)')
ax.set_ylabel('Throughput (MB/s)')
ax.set_title('zlink ML Training Workload: Throughput vs RTT')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
fig.savefig(f'{OUTPUT_DIR}/zlink-throughput-vs-rtt.png', dpi=150)
plt.close(fig)

# ─── Chart 5: ML training speedup factor ──
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for mode_name, _ in modes:
    if mode_name != 'direct_mount':
        speedups = []
        for rtt in rtts:
            direct = all_data[('ml_training', 'direct_mount', rtt)].avg_throughput_mb_s
            mode_val = all_data[('ml_training', mode_name, rtt)].avg_throughput_mb_s
            speedups.append(mode_val / direct if direct > 0 else 0)
        ax.plot(rtts, speedups, marker='o', linewidth=2, color=COLORS[mode_name], label=mode_name)

ax.set_xlabel('RTT (ms)')
ax.set_ylabel('Speedup Factor (x)')
ax.set_title('zlink ML Training Workload: Speedup vs Direct Mount')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
fig.savefig(f'{OUTPUT_DIR}/zlink-speedup-factor.png', dpi=150)
plt.close(fig)

# ─── Chart 6: ML training iteration time vs RTT ──
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for mode_name, _ in modes:
    values = [all_data[('ml_training', mode_name, rtt)].avg_iteration_ms for rtt in rtts]
    ax.plot(rtts, values, marker='o', linewidth=2, color=COLORS[mode_name], label=mode_name)

ax.set_xlabel('RTT (ms)')
ax.set_ylabel('Iteration Time (ms)')
ax.set_title('zlink ML Training Workload: Iteration Time vs RTT')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
fig.savefig(f'{OUTPUT_DIR}/zlink-iteration-time.png', dpi=150)
plt.close(fig)

print("All charts generated successfully!")
print(f"  {OUTPUT_DIR}/zlink-all-workloads-throughput.png")
print(f"  {OUTPUT_DIR}/zlink-all-workloads-speedup.png")
print(f"  {OUTPUT_DIR}/zlink-cross-workload-comparison.png")
print(f"  {OUTPUT_DIR}/zlink-throughput-vs-rtt.png")
print(f"  {OUTPUT_DIR}/zlink-speedup-factor.png")
print(f"  {OUTPUT_DIR}/zlink-iteration-time.png")
