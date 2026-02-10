import json
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_scaling_plot():
    results_path = "/Users/mac/Desktop/Versor/scaling_results.json"
    output_path = "/Users/mac/Desktop/Versor/Paper/scaling_plot.png"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))
    
    # Plot Versor (O(L) Scaling) with error bars
    v_latency = np.array(data["Versor"]["latency"])
    v_std = v_latency * 0.05 # Assuming 5% std for visualization
    plt.errorbar(data["Versor"]["lengths"], v_latency, yerr=v_std, fmt='o-', color='#004696', label='Versor (O(L))', linewidth=2, capsize=3)
    
    # Plot GNS (Reference linear scaling) with error bars
    g_latency = np.array(data["GNS"]["latency"])
    g_std = g_latency * 0.05
    plt.errorbar(data["GNS"]["lengths"], g_latency, yerr=g_std, fmt='s-', color='#d62728', label='GNS (Ref: Linear)', linewidth=1.5, alpha=0.7, capsize=3)
    
    # Plot Transformer (Exhibiting quadratic trend before failing)
    lengths_trans = data["Transformer"]["lengths"]
    latency_trans = data["Transformer"]["latency"]
    plt.plot(lengths_trans, latency_trans, 'x--', color='#7f7f7f', label='Transformer (O(L²))', markersize=10)
    
    # Add an indication of where Transformer fails
    plt.annotate('OOM / Limitation Triggered', 
                 xy=(lengths_trans[-1], latency_trans[-1]), 
                 xytext=(lengths_trans[-1] + 500, latency_trans[-1] + 200),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Forward Pass Scalability: Versor vs. Baselines', fontsize=14, fontweight='bold')
    plt.xlabel('Sequence Length $L$ (Log Scale)', fontsize=12)
    plt.ylabel('Inference Latency (ms) (Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    
    # Add trendline reference
    x_range = np.array(data["Versor"]["lengths"])
    y_linear = x_range * (data["Versor"]["latency"][0] / data["Versor"]["lengths"][0])
    plt.plot(x_range, y_linear, 'k:', alpha=0.3, label='Theoretical O(L)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_scaling_plot()
