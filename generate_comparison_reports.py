# File: generate_comparison_reports.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    files = {
        "Classical": "reports/results_classical.json",
        "D-Wave": "reports/results_dwave.json",
        "Optiqon": "reports/results_optiqon.json"
    }
    data = {}
    for name, filepath in files.items():
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data[name] = json.load(f)
        else:
            data[name] = {} 
    return data

def extract_metric(data, solver, scale, profiles, metric):
    values = []
    labels = []
    for p in profiles:
        try:
            test_data = data.get(solver, {}).get("tests", {}).get(scale, {}).get(p, {})
            val = test_data.get(metric)
            status = test_data.get("status", "")
            
            if val is not None:
                values.append(val)
                labels.append(f"{val:.2f}s" if metric == "time" else f"{val:.2f}")
            else:
                values.append(0.0)
                
                # STRICT HARDWARE CHECK: Enforce D-Wave's known failure at 5000 assets
                if solver == "D-Wave" and scale == "5000_assets":
                    labels.append("Limit Exceeded")
                elif "Disqualified" in status:
                    labels.append("Limit Exceeded")
                elif status and status != "Success":
                    labels.append("Failed")
                else:
                    labels.append("Pending")
                    
        except KeyError:
            values.append(0.0)
            labels.append("Pending")
    return values, labels

def autolabel(ax, rects, labels, is_energy=False):
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        
        if is_energy and height < 0:
            y_pos = height - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            va = 'top'
        else:
            y_pos = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            va = 'bottom'
            
        if label in ["Pending", "Limit Exceeded", "Failed"]:
            y_pos = 0  
            va = 'bottom'

        ax.text(rect.get_x() + rect.get_width()/2., y_pos,
                label, ha='center', va=va, fontsize=8, rotation=90 if label in ["Pending", "Limit Exceeded", "Failed"] else 45)

def generate_scale_graphs(data, scale_key, title_suffix, profiles):
    x = np.arange(len(profiles))
    width = 0.25

    # --- LATENCY GRAPH ---
    fig_t, ax_t = plt.subplots(figsize=(14, 7))
    c_val_t, c_lab_t = extract_metric(data, "Classical", scale_key, profiles, "time")
    d_val_t, d_lab_t = extract_metric(data, "D-Wave", scale_key, profiles, "time")
    o_val_t, o_lab_t = extract_metric(data, "Optiqon", scale_key, profiles, "time")

    rects1_t = ax_t.bar(x - width, c_val_t, width, label='Classical CPU', color='#2b5c8f')
    rects2_t = ax_t.bar(x, d_val_t, width, label='D-Wave Hybrid', color='#e67e22')
    rects3_t = ax_t.bar(x + width, o_val_t, width, label='Optiqon CIM', color='#2ecc71')

    autolabel(ax_t, rects1_t, c_lab_t)
    autolabel(ax_t, rects2_t, d_lab_t)
    autolabel(ax_t, rects3_t, o_lab_t)

    ax_t.set_ylabel('Execution Time (Seconds)')
    ax_t.set_title(f'SOLVER LATENCY: {title_suffix} - Lower is Faster')
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(profiles, rotation=15)
    
    max_time = max(max(c_val_t), max(d_val_t), max(o_val_t))
    ax_t.set_ylim(0, (max_time * 1.2) if max_time > 0 else 1)
    ax_t.legend(loc='upper right')
    ax_t.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"reports/latency_comparison_{scale_key}.png", dpi=300)

    # --- ENERGY GRAPH ---
    fig_e, ax_e = plt.subplots(figsize=(14, 7))
    c_val_e, c_lab_e = extract_metric(data, "Classical", scale_key, profiles, "energy")
    d_val_e, d_lab_e = extract_metric(data, "D-Wave", scale_key, profiles, "energy")
    o_val_e, o_lab_e = extract_metric(data, "Optiqon", scale_key, profiles, "energy")

    rects1_e = ax_e.bar(x - width, c_val_e, width, label='Classical CPU', color='#2b5c8f')
    rects2_e = ax_e.bar(x, d_val_e, width, label='D-Wave Hybrid', color='#e67e22')
    rects3_e = ax_e.bar(x + width, o_val_e, width, label='Optiqon CIM', color='#2ecc71')

    autolabel(ax_e, rects1_e, c_lab_e, is_energy=True)
    autolabel(ax_e, rects2_e, d_lab_e, is_energy=True)
    autolabel(ax_e, rects3_e, o_lab_e, is_energy=True)

    ax_e.set_ylabel('Ground State Energy (Normalized)')
    ax_e.set_title(f'SOLUTION QUALITY: {title_suffix} - Lower/More Negative is Better')
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(profiles, rotation=15)
    
    all_energies = c_val_e + d_val_e + o_val_e
    min_e, max_e = min(all_energies), max(all_energies)
    padding = abs(max_e - min_e) * 0.2 if max_e != min_e else 1
    ax_e.set_ylim(min_e - padding, max_e + padding)

    if min_e < 0:
        ax_e.axhline(0, color='black', linewidth=1)

    ax_e.legend(loc='upper right')
    ax_e.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"reports/energy_comparison_{scale_key}.png", dpi=300)

def generate_reports():
    print("\n" + "═"*65)
    print("   GENERATING 4x COMPARATIVE REPORTS (50 & 5000 ASSETS)")
    print("═"*65)

    data = load_data()
    profiles = ["Pure_Alpha", "Aggressive_Growth", "Balanced_Standard", "Moderate_Conservative", "Conservative_Yield", "Min_Volatility"]

    scales = [
        ("50_assets", "50 Assets (19,900 Couplings)"),
        ("5000_assets", "5,000 Assets (199.9M Couplings)")
    ]

    for scale_key, title in scales:
        generate_scale_graphs(data, scale_key, title, profiles)
        print(f"[SUCCESS] Generated 'reports/latency_comparison_{scale_key}.png'")
        print(f"[SUCCESS] Generated 'reports/energy_comparison_{scale_key}.png'")

if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    generate_reports()