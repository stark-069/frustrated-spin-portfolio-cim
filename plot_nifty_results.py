# File: plot_nifty_results.py
# Redraws Fig. 5 (Table V values as printed in the paper).
import os
import matplotlib.pyplot as plt

costs = [0.0, 0.5, 2.0, 5.0]  # market impact, %

cpu_energy = [-20.98, -19.14, -16.48, -20.35]
cim_energy = [-18.10, -18.12, -18.13, -18.32]

cpu_latency = [1.450, 1.081, 1.028, 0.987]
cim_latency = [3.730, 3.859, 3.304, 3.673]

num_assets = 95

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

ax1.plot(costs, cpu_energy, marker='o', label='Classical CPU', color='#1f77b4', linewidth=2)
ax1.plot(costs, cim_energy, marker='^', label='Optiqon CIM', color='#2ca02c', linewidth=2)
ax1.set_title(f'Solution Quality vs Transaction Costs ({num_assets} Assets)', fontsize=14)
ax1.set_ylabel('Ground State Energy')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

ax2.plot(costs, cpu_latency, marker='o', label='Classical CPU', color='#1f77b4', linewidth=2)
ax2.plot(costs, cim_latency, marker='^', label='Optiqon CIM', color='#2ca02c', linewidth=2)
ax2.set_title('Solver Latency vs Transaction Costs (Live NIFTY Data)', fontsize=14)
ax2.set_xlabel('Market Impact / Transaction Cost (%)', fontsize=12)
ax2.set_ylabel('Execution Time (Seconds)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()
os.makedirs('reports', exist_ok=True)
filename = os.path.join('reports', 'fig5_transaction_costs_regenerated.png')
plt.savefig(filename, dpi=300)
print(f"[*] Plot saved as '{filename}'")
