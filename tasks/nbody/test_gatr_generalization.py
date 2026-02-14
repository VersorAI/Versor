import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "gatr"))

from tasks.nbody.run_gatr import GATrAdapter
from tasks.nbody.data_gen import generate_gravity_data

def test_gatr_generalization():
    device = "cpu"
    print("Initializing GATr model...")
    model = GATrAdapter().to(device)
    model.eval()
    
    # Test on N=5
    print("Testing on N=5...")
    data_5 = generate_gravity_data(n_samples=1, n_steps=10, n_particles=5, device=device)
    with torch.no_grad():
        out_5 = model(data_5)
    print(f"N=5 success, output shape: {out_5.shape}")
    
    # Test on N=3
    print("Testing on N=3...")
    data_3 = generate_gravity_data(n_samples=1, n_steps=10, n_particles=3, device=device)
    try:
        with torch.no_grad():
            out_3 = model(data_3)
        print(f"N=3 success, output shape: {out_3.shape}")
        print("GATr can at least RUN on variable N.")
    except Exception as e:
        print(f"GATr FAILED on N=3: {e}")

if __name__ == "__main__":
    test_gatr_generalization()
