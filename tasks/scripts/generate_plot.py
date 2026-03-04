import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_scaling_plot():
    results_path = "/Users/mac/Desktop/Versor/results/scaling_results.json"
    output_path = "/Users/mac/Desktop/Versor/paper/scaling_plot.png"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Use a premium style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors and styles
    versor_color = '#0066CC' # Vibrant Blue
    gns_color = '#E63946'    # Soft Red
    trans_color = '#4A4E69'  # Slate Gray
    ref_line_color = '#8D99AE'
    
    # 1. Plot Versor (O(L) Scaling)
    v_lengths = data["Versor"]["lengths"]
    v_latency = np.array(data["Versor"]["latency"])
    ax.plot(v_lengths, v_latency, 'o-', color=versor_color, label='Versor (Matrix Kernel) $O(L)$', linewidth=3, markersize=8)
    
    # 2. Plot GNS (Reference linear scaling)
    g_lengths = data["GNS"]["lengths"]
    g_latency = np.array(data["GNS"]["latency"])
    ax.plot(g_lengths, g_latency, 's--', color=gns_color, label='GNS (Ref: Linear)', linewidth=1.5, alpha=0.6)
    
    # 3. Plot Transformer (Exhibiting quadratic trend before failing)
    t_lengths = data["Transformer"]["lengths"]
    t_latency = data["Transformer"]["latency"]
    ax.plot(t_lengths, t_latency, 'x-', color=trans_color, label='Standard Transformer $O(L^2)$', linewidth=2, markersize=10)
    
    # 4. Add Theoretical O(L) Dotted Reference Line
    # "The dotted reference line denotes a gradient of 1 (linear scaling) on the log-log plot."
    x_ref = np.array([128, 10000])
    # Match the slope starting from Versor's first point
    y_ref = x_ref * (v_latency[0] / v_lengths[0])
    ax.plot(x_ref, y_ref, ':', color=ref_line_color, linewidth=1.5, label='Gradient = 1 (Linear Reference)', alpha=0.8)

    # Aesthetics
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Computational Scaling: Latency vs. Sequence Length $L$', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Sequence Length $L$ (Log Scale)', fontsize=12)
    ax.set_ylabel('Forward Pass Latency (ms) (Log Scale)', fontsize=12)
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_scaling_plot()
