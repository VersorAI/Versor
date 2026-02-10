import torch
import torch.nn as nn
import time
import json
import numpy as np
import os
import sys

# Ensure we can import the models
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'Physics')))
from models import VersorRotorRNN, StandardTransformer, GraphNetworkSimulator

def measure_resource_usage(model, device, seq_length, n_particles=5, input_dim=6):
    """
    Measures memory usage and latency for a single forward pass.
    """
    batch_size = 1
    x = torch.randn(batch_size, seq_length, n_particles, input_dim).to(device)
    
    # Synchronize for timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    # Reset memory stats
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    try:
        with torch.no_grad():
            _ = model(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000 # ms
        memory = 0
        if device == 'cuda':
            memory = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
        
        return {"latency": latency, "memory": memory, "success": True}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"success": False, "error": "OOM"}
        return {"success": False, "error": str(e)}

def run_scaling_benchmark():
    print("="*60)
    print("OPTION A: LONG SEQUENCE SCALING BENCHMARK")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    
    lengths = [128, 512, 1024, 2048, 5120, 10000]
    results = {
        "Versor": {"latency": [], "memory": [], "lengths": []},
        "Transformer": {"latency": [], "memory": [], "lengths": []},
        "GNS": {"latency": [], "memory": [], "lengths": []}
    }
    
    # Initialize models
    models = {
        "Versor": VersorRotorRNN(n_particles=5).to(device),
        "Transformer": StandardTransformer(n_particles=5).to(device),
        "GNS": GraphNetworkSimulator(n_particles=5).to(device)
    }
    
    for L in lengths:
        print(f"\nTesting Sequence Length T = {L}")
        for name, model in models.items():
            print(f"  Measuring {name}...", end=" ", flush=True)
            res = measure_resource_usage(model, device, L)
            
            if res["success"]:
                print(f"Done. Latency: {res['latency']:.1f}ms, Mem: {res['memory']:.1f}MB")
                results[name]["latency"].append(res["latency"])
                results[name]["memory"].append(res["memory"])
                results[name]["lengths"].append(L)
            else:
                print(f"FAILED: {res['error']}")
                
    # Save results
    with open("scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to scaling_results.json")

if __name__ == "__main__":
    run_scaling_benchmark()
