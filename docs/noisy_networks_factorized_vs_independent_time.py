"""
author: Guilherme Alves Silveira
generated with AI tools (some adjustments were made)
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# Noisy Networks formula
# ε^w = f(ε^out) f(ε^in)^T
# ε^b = f(ε^out)
# f(x) = sign(x) * sqrt(|x|)

def f(x):
    return np.sign(x) * np.sqrt(np.abs(x))

# Sizes to test
sizes = [2, 4, 8, 16, 32, 64, 128]

# Number of trials for averaging
trials = 1000

factorized_times = []
independent_times = []

for size in sizes:
    # --- Factorized sampling timing ---
    t_start = time.perf_counter()
    for _ in range(trials):
        eps_in = np.random.randn(size)
        eps_out = np.random.randn(size)
        f_in = f(eps_in)
        f_out = f(eps_out)
        eps_w_factorized = np.outer(f_out, f_in)
        eps_b_factorized = f_out
    t_factorized = (time.perf_counter() - t_start) / trials
    factorized_times.append(t_factorized * 1e6)  # convert to microseconds

    # --- Independent sampling timing ---
    t_start = time.perf_counter()
    for _ in range(trials):
        eps_w_independent = np.random.randn(size, size)
        eps_b_independent = np.random.randn(size)
    t_independent = (time.perf_counter() - t_start) / trials
    independent_times.append(t_independent * 1e6)  # convert to microseconds

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Execution time comparison (linear scale)
ax1 = axes[0]
ax1.plot(sizes, factorized_times, 'o-', color='purple', linewidth=2, markersize=8, label='Factorized (Noisy Nets)')
ax1.plot(sizes, independent_times, 's-', color='darkgreen', linewidth=2, markersize=8, label='Independent (Standard)')
ax1.set_xlabel('Layer Size (n)', fontsize=12)
ax1.set_ylabel('Avg. Execution Time (μs)', fontsize=12)
ax1.set_title('Execution Time Comparison (Linear Scale)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(sizes)

# Plot 2: Execution time comparison (log-log scale)
ax2 = axes[1]
ax2.plot(sizes, factorized_times, 'o-', color='purple', linewidth=2, markersize=8, label='Factorized (Noisy Nets)')
ax2.plot(sizes, independent_times, 's-', color='darkgreen', linewidth=2, markersize=8, label='Independent (Standard)')
ax2.set_xlabel('Layer Size (n)', fontsize=12)
ax2.set_ylabel('Avg. Execution Time (μs)', fontsize=12)
ax2.set_title('Execution Time Comparison (Log-Log Scale)', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xticks(sizes)
ax2.set_xticklabels(sizes)

plt.tight_layout()
plt.savefig('noisy_nets_execution_time.png', dpi=150, bbox_inches='tight')
plt.show()

# Print numerical results
print("=" * 60)
print(f"{'Size':>6} | {'Factorized (μs)':>18} | {'Independent (μs)':>18} | {'Speedup':>10}")
print("-" * 60)
for i, size in enumerate(sizes):
    speedup = independent_times[i] / factorized_times[i]
    ratio = independent_times[i] / factorized_times[i]
    if ratio >= 1:
        speedup = ratio
        print(f"{size:>6} | {factorized_times[i]:>18.3f} | {independent_times[i]:>18.3f} | {speedup:>9.2f}x faster")
    else:
        slowdown = 1 / ratio
        print(f"{size:>6} | {factorized_times[i]:>18.3f} | {independent_times[i]:>18.3f} | {slowdown:>9.2f}x slower")
print("=" * 60)
