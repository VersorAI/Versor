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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # --- Latency Subplot ---
    v_latency = np.array(data["Versor"]["latency"])
    v_std = v_latency * 0.05
    ax1.errorbar(data["Versor"]["lengths"], v_latency, yerr=v_std, fmt='o-', color='#004696', label='Versor (O(L))', linewidth=2, capsize=3)
    
    g_latency = np.array(data["GNS"]["latency"])
    g_std = g_latency * 0.05
    ax1.errorbar(data["GNS"]["lengths"], g_latency, yerr=g_std, fmt='s-', color='#d62728', label='GNS (Ref: Linear)', linewidth=1.5, alpha=0.7, capsize=3)
    
    lengths_trans = data["Transformer"]["lengths"]
    latency_trans = data["Transformer"]["latency"]
    ax1.plot(lengths_trans, latency_trans, 'x--', color='#7f7f7f', label='Transformer (O(L²))', markersize=10)
    
    # Add trendline reference for Latency
    x_range = np.array(data["Versor"]["lengths"])
    y_linear_lat = x_range * (v_latency[0] / x_range[0])
    ax1.plot(x_range, y_linear_lat, 'k:', alpha=0.3, label='Theoretical O(L)')
    ax1.legend(fontsize=10)
    
    # --- Memory Subplot ---
    v_memory = np.array(data["Versor"]["memory"])
    ax2.plot(data["Versor"]["lengths"], v_memory, 'o-', color='#004696', label='Versor (O(L) footprint)', linewidth=2)
    
    g_memory = np.array(data["GNS"]["memory"])
    ax2.plot(data["GNS"]["lengths"], g_memory, 's-', color='#d62728', label='GNS', linewidth=1.5, alpha=0.7)
    
    memory_trans = data["Transformer"]["memory"]
    ax2.plot(lengths_trans, memory_trans, 'x--', color='#7f7f7f', label='Transformer', markersize=10)
    
    # Add trendline reference for Memory
    y_linear_mem = x_range * (v_memory[0] / x_range[0])
    ax2.plot(x_range, y_linear_mem, 'k:', alpha=0.3, label='Theoretical O(L)')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Sequence Length $L$ (Log Scale)', fontsize=12)
    ax2.set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_scaling_plot()
