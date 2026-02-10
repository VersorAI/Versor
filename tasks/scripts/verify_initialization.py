
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os

# Add local directory to path to find Model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Model.layers import VersorLinear
from Model.core import normalize_cl41

def test_initialization_stability(depth=20, width=128, batch_size=64):
    print("="*60)
    print(f"TESTING VERSOR INITIALIZATION STABILITY")
    print(f"Depth: {depth} layers")
    print(f"Width: {width} channels (multivectors)")
    print(f"Batch: {batch_size}")
    print("="*60)
    
    # 1. Construct a deep network
    layers = nn.ModuleList([
        VersorLinear(width, width) for _ in range(depth)
    ])
    
    # 2. Create grade-rich random input
    # Standard Normal input to simulate "previous layer" output
    x = torch.randn(batch_size, 1, width, 32)
    
    # Normalize input to start on the manifold
    x = normalize_cl41(x)
    
    print("\nLayer | Mean Variance | Max Component | Min Component | Status")
    print("-" * 65)
    
    variances = []
    
    # 3. Propagate and measure
    with torch.no_grad():
        current = x
        
        # Initial variance
        initial_var = current.var().item()
        print(f"Input | {initial_var:.6f}      | {current.max().item():.6f}      | {current.min().item():.6f}      | OK")
        variances.append(initial_var)
        
        for i, layer in enumerate(layers):
            # Forward pass
            current = layer(current)
            
            # Measure stats
            curr_var = current.var().item()
            curr_max = current.max().item()
            curr_min = current.min().item()
            
            # Check for explosion/vanishing
            if curr_var > 10.0:
                status = "EXPLODING 💥"
            elif curr_var < 0.01:
                status = "VANISHING ❄️"
            else:
                status = "STABLE ✅"
                
            print(f"{i+1:5d} | {curr_var:.6f}      | {curr_max:.6f}      | {curr_min:.6f}      | {status}")
            variances.append(curr_var)
            
    print("-" * 65)
    
    # 4. Summary
    ratio = variances[-1] / variances[0]
    print(f"\nFinal/Initial Variance Ratio: {ratio:.4f}")
    
    if 0.5 < ratio < 2.0:
        print("\n✅ SUCCESS: Signal preserved within factor of 2 over 20 layers.")
        print("   This confirms the Versor Initialization formula derivation.")
    else:
        print("\n❌ FAILURE: Signal unstable.")

if __name__ == "__main__":
    test_initialization_stability()
