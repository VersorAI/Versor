import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
from datetime import datetime

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "gatr"))
sys.path.append(os.path.join(root_dir, "Physics"))

import mock_dependencies
mock_dependencies.apply_mocks()

from Physics.run_gatr import GATrAdapter
from Physics.models import MambaSimulator
from Physics.data_gen import generate_gravity_data

def train_and_eval_model(name, model, device, train_data, test_data_5, test_data_3):
    print(f"\n--- Training {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        try:
            pred = model(X_train)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"  Training failed for {name}: {e}")
            return None

    model.eval()
    with torch.no_grad():
        # Test N=5
        try:
            mse_5 = loss_fn(model(test_data_5[:, :-1]), test_data_5[:, 1:]).item()
            print(f"  Final Test MSE (N=5): {mse_5:.4f}")
        except Exception as e:
            print(f"  Test N=5 failed for {name}: {e}")
            mse_5 = float('inf')
            
        # Test N=3
        try:
            mse_3 = loss_fn(model(test_data_3[:, :-1]), test_data_3[:, 1:]).item()
            print(f"  Final Test MSE (N=3): {mse_3:.4f}")
        except Exception as e:
            print(f"  Test N=3 failed for {name}: {e}")
            mse_3 = float('inf')
            
    return {"mse_n5": mse_5, "mse_n3": mse_3}

def main():
    device = "cpu"
    torch.manual_seed(42)
    
    # Pre-generate data
    print("Generating data...")
    train_data_5 = generate_gravity_data(n_samples=50, n_steps=100, n_particles=5, device=device)
    test_data_5 = generate_gravity_data(n_samples=5, n_steps=100, n_particles=5, device=device)
    test_data_3 = generate_gravity_data(n_samples=5, n_steps=100, n_particles=3, device=device)
    
    models = {
        "GATr": GATrAdapter().to(device),
        "Mamba": MambaSimulator(n_particles=5).to(device)
    }
    
    results = {}
    
    for name, model in models.items():
        res = train_and_eval_model(name, model, device, train_data_5, test_data_5, test_data_3)
        if res:
            results[name] = res
            if res["mse_n3"] != float('inf'):
                ratio = res["mse_n3"] / res["mse_n5"]
                print(f"  Degradation Ratio: {ratio:.2f}x")
            else:
                print(f"  {name} FAILED Generalization.")

    # Save results
    os.makedirs('results', exist_ok=True)
    filename = f"results/competitors_generalization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    print(f"\n✓ All results saved to {filename}")

if __name__ == "__main__":
    main()
