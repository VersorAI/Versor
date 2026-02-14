import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import json
from datetime import datetime

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "gatr"))

from tasks.nbody.run_gatr import GATrAdapter
from tasks.nbody.models import (
    VersorRotorRNN, 
    MambaSimulator, 
    GraphNetworkSimulator, 
    EquivariantGNN,
    StandardTransformer
)
from tasks.nbody.data_gen import generate_gravity_data

# --- ADAPTERS FOR 3D POS INPUT ---

class FrameModelAdapter(nn.Module):
    """Adapts a model that expects (B, N, 6) to handle (B, S, N, 3) frame-by-frame."""
    def __init__(self, base_model, in_dim=3, out_dim=3):
        super().__init__()
        self.base_model = base_model
        self.in_proj = nn.Linear(in_dim, 6)
        self.out_proj = nn.Linear(6, out_dim)

    def forward(self, x):
        # x: (B, S, N, 3)
        B, S, N, _ = x.shape
        x_flat = x.reshape(B * S, N, 3)
        x_6 = self.in_proj(x_flat)
        
        # Base model forward
        try:
            out_6 = self.base_model(x_6) # Expects (BS, N, 6)
        except:
            # Some models expect (BS, S_dummy, N, 6) or (BS, N*6)
            out_6 = self.base_model(x_6.unsqueeze(1)).squeeze(1)
            
        out_3 = self.out_proj(out_6)
        return out_3.reshape(B, S, N, 3)

class TemporalModelAdapter(nn.Module):
    """Adapts models that handle sequences (B, S, N*dim)."""
    def __init__(self, base_model, n_particles):
        super().__init__()
        self.base_model = base_model
        self.n_particles = n_particles

    def forward(self, x):
        # x: (B, S, N, 3)
        B, S, N, D = x.shape
        x_flat = x.reshape(B, S, N * D)
        out_flat = self.base_model(x_flat)
        return out_flat.reshape(B, S, N, D)

def train_and_eval(name, model, train_data, test_data, device):
    print(f"\n--- Training {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Hidden Velocity Task: Input (B, S, N, 3), Target (B, S, N, 3)
    X_train = train_data[:, :-1, :, :3].to(device)
    Y_train = train_data[:, 1:, :, :3].to(device)
    X_test = test_data[:, :-1, :, :3].to(device)
    Y_test = test_data[:, 1:, :, :3].to(device)

    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        try:
            pred = model(X_train)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
        except Exception as e:
            print(f"  Training FAILED for {name}: {e}")
            return float('inf')

    model.eval()
    with torch.no_grad():
        mse = loss_fn(model(X_test), Y_test).item()
        print(f"  Final MSE: {mse:.6f}")
    return mse

def main():
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Generating data (Hidden Velocity Task)...")
    train_data = generate_gravity_data(n_samples=100, n_steps=50, n_particles=5, device=device)
    test_data = generate_gravity_data(n_samples=20, n_steps=50, n_particles=5, device=device)

    # Note: VersorRotorRNN in models.py needs to handle input_dim=3
    # I will instantiate it with input_dim=3
    
    models = {
        "Versor (RNN)": VersorRotorRNN(input_dim=3, n_particles=5, hidden_channels=16).to(device),
        "Mamba": TemporalModelAdapter(MambaSimulator(input_dim=15, n_particles=5), n_particles=5).to(device), # N*3 = 15
        "GATr": FrameModelAdapter(GATrAdapter().to(device)),
        "GNS": FrameModelAdapter(GraphNetworkSimulator(n_particles=5, input_dim=6).to(device)),
        "EGNN": FrameModelAdapter(EquivariantGNN(n_particles=5).to(device)),
        "Transformer (Frame)": FrameModelAdapter(StandardTransformer(n_particles=5, d_model=256).to(device))
    }

    results = {}
    for name, model in models.items():
        mse = train_and_eval(name, model, train_data, test_data, device)
        results[name] = mse

    # Save
    os.makedirs('results', exist_ok=True)
    filename = f"results/hidden_velocity_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print("🏆 HIDDEN VELOCITY BENCHMARK RESULTS")
    print("="*50)
    sorted_res = sorted(results.items(), key=lambda x: x[1])
    for i, (name, mse) in enumerate(sorted_res):
        print(f"{i+1}. {name:20}: MSE = {mse:.6f}")
    print("="*50)
    print(f"✓ Results saved to {filename}")

if __name__ == "__main__":
    main()
