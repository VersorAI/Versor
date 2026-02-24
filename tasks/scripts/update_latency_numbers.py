import torch
import torch.nn as nn
import time
import json
import os
import sys

# Ensure library is in path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)

from Model.model import VersorTransformer

def run_standard_bench(method, embed_dim=24, n_heads=4, n_layers=4, seq_len=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VersorTransformer(
        embed_dim=embed_dim, 
        n_heads=n_heads, 
        n_layers=n_layers, 
        n_classes=6, 
        use_rotor_pool=False,
        method=method
    ).to(device)
    model.eval()
    
    dummy_input = torch.randn(1, seq_len, embed_dim, 32).to(device)
    
    # Warmup
    for _ in range(10): _ = model(dummy_input)
    
    # Benchmark
    iters = 100
    if device.type == "mps": torch.mps.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = model(dummy_input)
    if device.type == "mps": torch.mps.synchronize()
    
    latency = (time.time() - start) / iters * 1000
    return latency

if __name__ == "__main__":
    print("Benchmarking for Paper Update...")
    lat_bitmasked = run_standard_bench("bitmasked")
    print(f"Bit-Masked Latency: {lat_bitmasked:.3f} ms")
    
    lat_matrix = run_standard_bench("matrix")
    print(f"Matrix Latency: {lat_matrix:.3f} ms")
    
    speedup_engine = lat_bitmasked / lat_matrix
    print(f"Engine Speedup: {speedup_engine:.2f}x")
    
    # Create report
    report = {
        "bitmasked_base": lat_bitmasked,
        "matrix_base": lat_matrix,
        "engine_speedup": speedup_engine
    }
    
    with open("results/new_latency_results.json", "w") as f:
        json.dump(report, f, indent=4)
