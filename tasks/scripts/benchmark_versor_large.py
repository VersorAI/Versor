import torch
import torch.nn as nn
import time
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.model import VersorTransformer

def benchmark_versor_large():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Target approx 37M params or similar compute
    # embed_dim=128, n_layers=6?
    # From earlier: 128, 12 -> 76M
    # So 128, 6 -> ~38M
    model = VersorTransformer(embed_dim=128, n_heads=16, n_layers=6, n_classes=6, use_rotor_pool=False).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Large Versor Parameters: {params:,}")
    
    # Measure Latency
    # Input format: (B, S, D, 32)
    dummy_input = torch.randn(1, 50, 128, 32).to(device)
    
    # Warmup
    for _ in range(10): _ = model(dummy_input)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    latency = (time.time() - start) * 10
    
    print(f"Large Versor Latency: {latency:.2f} ms")
    
    results = {
        "model": "Large Versor",
        "params": params,
        "latency_ms": latency
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/versor_large_bench.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    benchmark_versor_large()
