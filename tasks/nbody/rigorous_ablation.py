import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
import os
import time

# Add paths
root_dir = os.getcwd()
if root_dir not in sys.path:
    sys.path.append(root_dir)

from Physics.data_gen import generate_gravity_data
from Physics.models import StandardTransformer
import kernel as algebra

class VersorAblation(nn.Module):
    def __init__(self, use_norm=True, use_recursive=True, input_dim=6, hidden_channels=16, n_particles=5):
        super().__init__()
        self.use_norm = use_norm
        self.use_recursive = use_recursive
        self.hidden_channels = hidden_channels
        self.n_particles = n_particles
        
        self.proj_in = nn.Linear(input_dim, hidden_channels * 32)
        self.proj_out = nn.Linear(hidden_channels * 32, input_dim)
        
    def forward(self, x):
        B, S, N, D = x.shape
        psi = torch.zeros(B, N, self.hidden_channels, 32, device=x.device)
        psi[..., 0] = 1.0 
        
        outputs = []
        x_embs = self.proj_in(x).reshape(B, S, N, self.hidden_channels, 32)
        
        for t in range(S):
            u_t = x_embs[:, t]
            
            if self.use_recursive:
                # Delta-Rotor generator
                delta_r = u_t.clone()
                delta_r[..., 0] += 1.0 
                if self.use_norm:
                    delta_r = algebra.manifold_normalization(delta_r)
                
                # Multiplicative Update
                psi = algebra.geometric_product(delta_r, psi)
                if self.use_norm:
                    psi = algebra.manifold_normalization(psi)
            else:
                # Simple linear recurrence in MV space (Ablation)
                psi = psi + u_t
                if self.use_norm:
                    psi = algebra.manifold_normalization(psi)
            
            out_emb = psi
            pred_delta = self.proj_out(out_emb.reshape(B, N, -1))
            outputs.append(x[:, t] + pred_delta)
            
        return torch.stack(outputs, dim=1)

def train_and_eval(model, train_data, test_data, epochs=20, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if torch.isnan(loss):
            return float('inf')
            
    model.eval()
    with torch.no_grad():
        X_test = test_data[:, :-1]
        Y_test = test_data[:, 1:]
        try:
            pred = model(X_test)
            mse = loss_fn(pred, Y_test).item()
            return float(mse)
        except:
            return float('inf')

def main():
    print("="*60)
    print("RUNNING RIGOROUS ABLATION STUDY")
    print("="*60)
    
    device = "cpu"
    seeds = [42, 123, 456]
    
    configs = [
        {"name": "Full Versor", "use_norm": True, "use_recursive": True},
        {"name": "w/o Manifold Norm", "use_norm": False, "use_recursive": True},
        {"name": "w/o Recursive Rotor", "use_norm": True, "use_recursive": False},
        {"name": "Standard Transformer", "type": "std"}
    ]
    
    results = {}
    
    for config in configs:
        name = config["name"]
        print(f"\nEvaluating: {name}")
        scores = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            train_data = generate_gravity_data(n_samples=100, n_steps=50, device=device)
            test_data = generate_gravity_data(n_samples=20, n_steps=50, device=device)
            
            if config.get("type") == "std":
                model = StandardTransformer(n_particles=5).to(device)
            else:
                model = VersorAblation(use_norm=config["use_norm"], use_recursive=config["use_recursive"]).to(device)
            
            print(f"  Seed {seed}...", end=" ", flush=True)
            score = train_and_eval(model, train_data, test_data, epochs=15, device=device)
            scores.append(score)
            print(f"MSE: {score:.4f}")
            
        results[name] = {
            "mean": float(np.mean([s for s in scores if s != float('inf')])) if any(s != float('inf') for s in scores) else "Diverged (NaN)",
            "std": float(np.std([s for s in scores if s != float('inf')])) if any(s != float('inf') for s in scores) else 0.0,
            "raw": scores
        }

    # Final Summary
    print("\n" + "="*60)
    print(f"{'Configuration':<25} | {'MSE (Mean ± Std)':<20}")
    print("-" * 60)
    for name, stat in results.items():
        if isinstance(stat["mean"], str):
            res_str = stat["mean"]
        else:
            res_str = f"{stat['mean']:.2f} ± {stat['std']:.2f}"
        print(f"{name:<25} | {res_str}")
        
    with open("results/ablation_stats.json", "w") as f:
        # Format for paper matching
        paper_format = {k: (f"{v['mean']:.2f} ± {v['std']:.2f}" if not isinstance(v['mean'], str) else v['mean']) for k, v in results.items()}
        json.dump(paper_format, f, indent=2)
    
    print(f"\n✓ Saved results to results/ablation_stats.json")

if __name__ == "__main__":
    main()
