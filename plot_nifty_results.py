# File: plot_nifty_results.py
import matplotlib.pyplot as plt

# 1. The Hardcoded Results from the Terminal Output
costs = [0.0, 0.5, 2.0, 5.0]

cpu_energy = [61.82, 72.14, 69.84, 42.06]
cim_energy = [-18.10, -18.12, -18.13, -18.32]

cpu_latency = [10.124, 9.186, 9.830, 67.457]
cim_latency = [3.730, 3.859, 3.304, 3.673]

num_assets = 95

# 2. Build the Graph
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# --- Plot A: Energy (Solution Quality) ---
ax1.plot(costs, cpu_energy, marker='o', label='Classical CPU (Failed States)', color='#1f77b4', linewidth=2)
ax1.plot(costs, cim_energy, marker='s', label='Optiqon CIM (Optimal Ground States)', color='#2ca02c', linewidth=2)
ax1.set_title(f'Solution Quality vs Transaction Costs (Live NIFTY Data - {num_assets} Assets)', fontsize=14)
ax1.set_ylabel('Ground State Energy (Lower is Better)')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Add an annotation showing the CPU failed to satisfy the budget constraint
ax1.text(2.5, 55, 'Classical CPU Collapsed\n(Invalid Positive Energy)', 
         fontsize=12, color='red', alpha=0.7, ha='center',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# --- Plot B: Latency ---
ax2.plot(costs, cpu_latency, marker='o', label='Classical CPU', color='#1f77b4', linewidth=2)
ax2.plot(costs, cim_latency, marker='s', label='Optiqon CIM', color='#2ca02c', linewidth=2)
ax2.set_title(f'Solver Latency vs Transaction Costs (Live NIFTY Data)', fontsize=14)
ax2.set_xlabel('Market Impact / Transaction Cost (%)', fontsize=12)
ax2.set_ylabel('Execution Time (Seconds)')
ax2.set_yscale('log')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# 3. Save and Output
plt.tight_layout()
filename = 'nifty_cost_supremacy_plot.png'
plt.savefig(filename, dpi=300)
print(f"[*] Benchmark plot successfully saved as '{filename}'")