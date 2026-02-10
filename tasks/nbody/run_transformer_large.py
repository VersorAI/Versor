import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
import sys

# Standard Transformer implementation
class StandardTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, n_classes=30):
        super().__init__()
        self.embedding = nn.Linear(6, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        h = self.embedding(x)
        h = self.transformer(h)
        return self.classifier(h)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_transformer():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Target 50M Parameters
    # d_model=1024, n_layers=12, nhead=16
    # Param count approx: 12 * (1024^2 * 4 + 1024^2 * 4) = 12 * 8 * 1M = 96M?
    # Let's use d_model=512, n_layers=8 -> approx 25M
    model = StandardTransformer(d_model=512, nhead=8, num_layers=12, n_classes=6).to(device)
    params = count_parameters(model)
    print(f"Large Transformer Parameters: {params:,}")
    
    # Measure Latency
    dummy_input = torch.randn(1, 50, 6).to(device)
    
    # Warmup
    for _ in range(10): _ = model(dummy_input)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    latency = (time.time() - start) * 10 # 1000 / 100
    
    print(f"Large Transformer Latency: {latency:.2f} ms")
    
    results = {
        "model": "Large Transformer",
        "params": params,
        "latency_ms": latency
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/transformer_large_bench.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    benchmark_transformer()
