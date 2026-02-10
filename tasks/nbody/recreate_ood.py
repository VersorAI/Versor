
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add paths
root_dir = os.getcwd()
if root_dir not in sys.path:
    sys.path.append(root_dir)
from Physics.data_gen import generate_gravity_data
from Physics.models import StandardTransformer, VersorRotorRNN

def train_on_normal_masses(model, device='cpu'):
    # Normal masses [0.5, 1.5] - already default in data_gen.py
    train_data = generate_gravity_data(n_samples=200, n_steps=100, n_particles=5, device=device)
    X = train_data[:, :-1]
    Y = train_data[:, 1:]
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    model.train()
    for _ in range(30):
        optimizer.zero_grad()
        loss = loss_fn(model(X), Y)
        loss.backward()
        optimizer.step()
    return model

def generate_heavy_mass_data(n_samples=50, n_steps=100, n_particles=5, device='cpu'):
    # Manually tweak generate_gravity_data logic for heavy masses [5.0, 10.0]
    # We'll monkeypatch or just copy logic
    pos = torch.randn(n_samples, n_particles, 3, device=device) * 0.5
    vel = torch.randn(n_samples, n_particles, 3, device=device) * 0.5
    mass = torch.rand(n_samples, n_particles, 1, device=device) * 5.0 + 5.0 # [5, 10]
    
    G = 1.0; dt = 0.01
    def get_acc(p, m):
        diff = p.unsqueeze(2) - p.unsqueeze(1) 
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-3
        direction = -diff
        force_magnitude = (G * m.unsqueeze(2) * m.unsqueeze(1)) / (dist ** 3)
        mask = ~torch.eye(n_particles, device=device).bool().unsqueeze(0).unsqueeze(-1)
        force = (direction * force_magnitude * mask).sum(dim=2)
        return force / m

    acc = get_acc(pos, mass)
    trajectory = []
    for _ in range(n_steps):
        trajectory.append(torch.cat([pos, vel], dim=-1))
        vel_half = vel + acc * (dt / 2.0)
        pos = pos + vel_half * dt
        acc = get_acc(pos, mass)
        vel = vel_half + acc * (dt / 2.0)
    return torch.stack(trajectory, dim=1)

def eval_model(model, data):
    model.eval()
    with torch.no_grad():
        X = data[:, :-1]
        Y = data[:, 1:]
        loss = nn.MSELoss()(model(X), Y)
    return loss.item()

def run_ood_test():
    device = 'cpu'
    print("Training models on NORMAL masses [0.5, 1.5]...")
    
    trans = StandardTransformer(n_particles=5).to(device)
    versor = VersorRotorRNN().to(device)
    
    train_on_normal_masses(trans, device)
    train_on_normal_masses(versor, device)
    
    print("\nEvaluating on NORMAL masses...")
    normal_data = generate_gravity_data(n_samples=50, n_steps=50, n_particles=5, device=device)
    mse_trans_norm = eval_model(trans, normal_data)
    mse_versor_norm = eval_model(versor, normal_data)
    print(f"  Transformer: {mse_trans_norm:.6f}")
    print(f"  Versor:      {mse_versor_norm:.6f}")
    
    print("\nEvaluating on HEAVY masses [5.0, 10.0] (OOD)...")
    heavy_data = generate_heavy_mass_data(n_samples=50, n_steps=50, n_particles=5, device=device)
    mse_trans_heavy = eval_model(trans, heavy_data)
    mse_versor_heavy = eval_model(versor, heavy_data)
    print(f"  Transformer: {mse_trans_heavy:.6f}")
    print(f"  Versor:      {mse_versor_heavy:.6f}")
    
    inc_trans = (mse_trans_heavy - mse_trans_norm) / mse_trans_norm * 100
    inc_versor = (mse_versor_heavy - mse_versor_norm) / mse_versor_norm * 100
    
    print(f"\nMSE Increase:")
    print(f"  Transformer: {inc_trans:.1f}% (Paper claims 450%)")
    print(f"  Versor:      {inc_versor:.1f}% (Paper claims 12%)")
    
    # Save results
    output = {
        "normal": {
            "transformer": mse_trans_norm,
            "versor": mse_versor_norm
        },
        "heavy": {
            "transformer": mse_trans_heavy,
            "versor": mse_versor_heavy
        },
        "increase_percent": {
            "transformer": inc_trans,
            "versor": inc_versor
        }
    }
    os.makedirs('results', exist_ok=True)
    with open('results/ood_mass_results.json', 'w') as f:
        json.dump(output, f, indent=4)
    print("\n✓ Results saved to results/ood_mass_results.json")

if __name__ == "__main__":
    import json
    run_ood_test()
