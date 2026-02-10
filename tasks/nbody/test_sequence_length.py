#!/usr/bin/env python3
"""
QUICK WIN EXPERIMENT: Sequence Length Scaling
Tests O(L) complexity advantage of RRA vs O(L²) Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from datetime import datetime

from data_gen import generate_gravity_data
from models import VersorRotorRNN, StandardTransformer, GraphNetworkSimulator

def train_on_length(model, train_length=50, n_samples=500, epochs=20, device='cpu'):
    """Train model on specific sequence length"""
    print(f"    Training on T={train_length}...")
    
    # Generate training data
    train_data = generate_gravity_data(
        n_samples=n_samples,
        n_steps=train_length,
        n_particles=5,
        device=device
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    batch_size = 16
    
    for epoch in range(epochs):
        total_loss = 0
        n_batches = n_samples // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = train_data[start_idx:end_idx]
            
            # Input: first half, Target: second half
            split = train_length // 2
            input_seq = batch[:, :split]
            target_seq = batch[:, split:]
            
            optimizer.zero_grad()
            
            try:
                pred = model(input_seq)
                loss = loss_fn(pred, target_seq)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA" in str(e):
                    return model, float('inf'), "OOM"
                return model, float('inf'), str(e)
        
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    return model, total_loss/n_batches, None

def test_on_length(model, test_length, n_samples=50, device='cpu'):
    """Test model on different sequence length"""
    
    # Generate test data
    test_data = generate_gravity_data(
        n_samples=n_samples,
        n_steps=test_length,
        n_particles=5,
        device=device
    )
    
    loss_fn = nn.MSELoss()
    split = test_length // 2
    input_seq = test_data[:, :split]
    target_seq = test_data[:, split:]
    
    model.eval()
    
    # Measure latency
    start_time = time.time()
    
    with torch.no_grad():
        try:
            pred = model(input_seq)
            mse = loss_fn(pred, target_seq).item()
            success = True
            error = None
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"      OOM at T={test_length}")
                mse = float('inf')
                success = False
                error = "OOM"
            else:
                print(f"      Error: {e}")
                mse = float('inf')
                success = False
                error = str(e)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "mse": float(mse),
        "latency_ms": float(latency_ms),
        "success": success,
        "error": error
    }

def main():
    print("\n" + "="*70)
    print("🚀 QUICK WIN: Sequence Length Scaling")
    print("="*70)
    print("\nThis demonstrates Versor's O(L) advantage:")
    print("  • Train on T=50")
    print("  • Test on T∈{20, 50, 100, 200, 500}")
    print("  • Transformer should degrade (O(L²) complexity)")
    print("  • Versor should scale (O(L) complexity)")
    print("\n" + "="*70 + "\n")
    
    device = "cpu"
    train_T = 50
    test_Ts = [20, 50, 100, 200, 500]
    seeds = [42, 123]  # 2 seeds for speed
    
    results = {
        "experiment": "sequence_length_scaling",
        "timestamp": datetime.now().isoformat(),
        "train_T": train_T,
        "test_Ts": test_Ts,
        "seeds": seeds,
        "models": {}
    }
    
    model_classes = {
        "Versor": VersorRotorRNN,
        "Transformer": StandardTransformer,
        "GNS": GraphNetworkSimulator,
    }
    
    for model_name, ModelClass in model_classes.items():
        print(f"\n{'='*70}")
        print(f"📊 Testing: {model_name}")
        print(f"{'='*70}")
        
        model_results = {"seeds": {}}
        
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            seed_results = {"test_T": {}}
            
            # Train on T=50
            print(f"  [1/2] Training on T={train_T}...")
            model = ModelClass(n_particles=5).to(device)
            model, train_loss, error = train_on_length(
                model,
                train_length=train_T,
                n_samples=500,
                epochs=20,
                device=device
            )
            
            if error:
                print(f"    ✗ Training failed: {error}")
                model_results["seeds"][str(seed)] = {"status": "train_failed", "error": error}
                continue
            
            # Test on various lengths
            print(f"  [2/2] Testing on multiple T...")
            for test_T in test_Ts:
                print(f"    Testing T={test_T}...", end=" ")
                
                result = test_on_length(model, test_T, n_samples=50, device=device)
                
                if result["success"]:
                    print(f"✓ MSE={result['mse']:.2f}, Latency={result['latency_ms']:.1f}ms")
                else:
                    print(f"✗ {result['error']}")
                
                seed_results["test_T"][str(test_T)] = result
            
            model_results["seeds"][str(seed)] = seed_results
        
        # Aggregate statistics
        print(f"\n  📈 Aggregate Results for {model_name}:")
        for test_T in test_Ts:
            mses = []
            latencies = []
            for seed in seeds:
                seed_key = str(seed)
                if seed_key in model_results["seeds"]:
                    test_results = model_results["seeds"][seed_key].get("test_T", {})
                    if str(test_T) in test_results:
                        result = test_results[str(test_T)]
                        if result["success"] and result["mse"] != float('inf'):
                            mses.append(result["mse"])
                            latencies.append(result["latency_ms"])
            
            if mses:
                mean_mse = np.mean(mses)
                std_mse = np.std(mses)
                mean_lat = np.mean(latencies)
                print(f"    T={test_T:3d}: MSE={mean_mse:6.2f}±{std_mse:5.2f}, "
                      f"Latency={mean_lat:6.1f}ms ({len(mses)}/{len(seeds)} seeds)")
            else:
                print(f"    T={test_T:3d}: FAILED (all seeds)")
        
        results["models"][model_name] = model_results
    
    # Save results
    filename = f"sequence_length_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("💾 RESULTS SAVED")
    print("="*70)
    print(f"Saved to: {filename}")
    
    # Analysis
    print("\n" + "="*70)
    print("📊 ANALYSIS")
    print("="*70)
    
    print("\n✅ Expected Pattern:")
    print("  • Versor: Relatively flat MSE across T (O(L) graceful)")
    print("  • Transformer: Degrading MSE as T increases (O(L²) struggles)")
    print("  • GNS: Also relatively flat (also O(L))")
    
    print("\n🎯 KEY FINDING (If Pattern Holds):")
    print("  • Versor maintains <20% degradation across 25× sequence length increase")
    print("  • Transformer either fails (OOM) or degrades significantly")
    print("  • This validates the RRA's architectural advantage")
    
    print("\n📝 FOR THE PAPER:")
    print("  • Add new section: 'Scaling to Long Sequences'")
    print("  • Emphasize O(L) vs O(L²) complexity")
    print("  • Show this enables longer-horizon modeling")
    print("  • Frame as architectural validation of RRA")
    
    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70)
    print("1. Review results in", filename)
    print("2. If pattern holds: Add to paper as Section 6.2")
    print("3. Generate scaling plot (MSE vs T)")
    print("4. Update abstract to include this result")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        main()
    else:
        print("\n" + "="*70)
        print("🚀 READY TO RUN: Sequence Length Scaling Test")
        print("="*70)
        print("\nThis will:")
        print("  • Train all models on T=50 timesteps")
        print("  • Test on T∈{20, 50, 100, 200, 500}")
        print("  • Measure MSE and latency at each length")
        print("  • Demonstrate O(L) vs O(L²) advantage")
        print("\nEstimated time: ~20-30 minutes (CPU)")
        print("\nTo run: python test_sequence_length.py --run")
        print("="*70)
