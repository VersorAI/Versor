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

def benchmark_engine_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # Model configuration (Cl(4,1))
    embed_dim = 64
    n_heads = 8
    n_layers = 6
    n_classes = 10
    seq_len = 128
    
    # We'll compare both engines
    configs = ["bitmasked", "matrix"]
    results = {}

    dummy_input = torch.randn(1, seq_len, embed_dim, 32).to(device)

    for method in configs:
        print(f"\nBenchmarking engine: {method}")
        model = VersorTransformer(
            embed_dim=embed_dim, 
            n_heads=n_heads, 
            n_layers=n_layers, 
            n_classes=n_classes, 
            method=method
        ).to(device)
        
        # Warmup
        for _ in range(10): 
            _ = model(dummy_input)
        
        # Benchmark
        iters = 50
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
            
        start = time.time()
        for _ in range(iters):
            _ = model(dummy_input)
            
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
            
        latency = (time.time() - start) / iters * 1000
        results[method] = latency
        print(f"  Latency: {latency:.2f} ms")

    # Save results
    with open(os.path.join(root_dir, "results/engine_comparison.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    speedup = results["bitmasked"] / results["matrix"]
    print(f"\nMatrix Isomorphism Speedup vs Bit-Masked: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_engine_comparison()
