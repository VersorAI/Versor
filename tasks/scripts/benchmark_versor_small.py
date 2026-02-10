import torch
import torch.nn as nn
import time
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.model import VersorTransformer

def benchmark_versor_small():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1M param Versor
    # embed_dim=24, n_layers=4 -> ~0.9M
    model = VersorTransformer(embed_dim=24, n_heads=4, n_layers=4, n_classes=6, use_rotor_pool=False).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Small Versor (1M) Parameters: {params:,}")
    
    dummy_input = torch.randn(1, 50, 24, 32).to(device)
    
    # Warmup
    for _ in range(10): _ = model(dummy_input)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    latency = (time.time() - start) * 10
    
    print(f"Small Versor (1M) Latency: {latency:.2f} ms")

if __name__ == "__main__":
    benchmark_versor_small()
