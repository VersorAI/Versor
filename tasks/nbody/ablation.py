import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys

# Add path for library
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(root_dir, "tasks/nbody"))
sys.path.append(os.path.join(root_dir, "library"))

from data_gen import generate_gravity_data
import gacore.kernel as algebra

class HybridAblationModel(nn.Module):
    def __init__(self, mode="baseline", n_particles=5, input_dim=6, hidden_dim=512):
        super().__init__()
        self.mode = mode
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # Total hidden bits
        
        # d_model for Euclidean parts
        self.d_model = hidden_dim
        
        # For GA parts: 16 channels * 32 blades = 512
        self.n_channels = hidden_dim // 32
        
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, input_dim)
        
        # Euclidean Components
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Geometric Components
        # Note: We use the weight matrix for a Geometric Linear Layer
        self.ga_weight = nn.Parameter(torch.randn(self.n_channels, self.n_channels, 32) * 0.01)
        
    def forward(self, x):
        B, S, N, D = x.shape
        x_flat = x.reshape(B * S, N, D)
        h = self.proj_in(x_flat) # (BS, N, 512)
        
        # Condition 1: Normalization
        if self.mode in ["baseline", "strip_mlp"]:
            # Use Euclidean LayerNorm
            h = self.ln(h)
        elif self.mode in ["strip_ln", "full"]:
            # Use Manifold Normalization
            # Reshape to (BS, N, 16, 32)
            h_ga = h.view(B * S, N, self.n_channels, 32)
            h_ga = algebra.manifold_normalization(h_ga, [1,1,1,1,-1])
            h = h_ga.reshape(B * S, N, -1)
            
        # Condition 2: Mixing/MLP
        if self.mode in ["baseline", "strip_ln"]:
            # Use Euclidean MLP
            h = h + self.mlp(h)
        elif self.mode in ["strip_mlp", "full"]:
            # Use Clifford Layer
            h_ga = h.view(B * S, N, self.n_channels, 32)
            # Geometric Product with weighted channels
            h_ga = algebra.geometric_linear_layer(h_ga, self.ga_weight, [1,1,1,1,-1])
            h_ga = algebra.manifold_normalization(h_ga, [1,1,1,1,-1])
            h = h + h_ga.reshape(B * S, N, -1)
            
        # Output
        delta = self.proj_out(h)
        next_state = x_flat + delta
        return next_state.reshape(B, S, N, D)

def run_rigorous_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    # Quick paper-patch settings
    n_samples = 100
    n_steps = 50
    epochs = 10
    seeds = [42, 123, 456]
    
    modes = [
        ("Baseline (Transformer)", "baseline"),
        ("Strip LayerNorm", "strip_ln"),
        ("Strip MLP", "strip_mlp"),
        ("Full Versor", "full")
    ]
    
    results = {}
    
    for name, mode in modes:
        print(f"\nExperiment: {name}")
        mses = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            train_data, _ = generate_gravity_data(n_samples=n_samples, n_steps=n_steps, device=device)
            test_data, _ = generate_gravity_data(n_samples=20, n_steps=n_steps, device=device)
            
            model = HybridAblationModel(mode=mode).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            
            # Train
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = model(train_data[:, :-1])
                loss = loss_fn(pred, train_data[:, 1:])
                loss.backward()
                optimizer.step()
                if torch.isnan(loss):
                    print("Diverged!")
                    break
            
            # Eval (Rollout)
            model.eval()
            with torch.no_grad():
                rollout_steps = 49
                current_state = test_data[:, :1]
                ground_truth = test_data[:, 1:] # 49 steps
                
                preds = []
                for _ in range(rollout_steps):
                    out = model(current_state)
                    next_step = out[:, -1:]
                    preds.append(next_step)
                    current_state = torch.cat([current_state, next_step], dim=1)
                
                preds = torch.cat(preds, dim=1)
                mse = loss_fn(preds, ground_truth).item()
                mses.append(mse)
                print(f"  Seed {seed}: Rollout MSE = {mse:.4f}")
        
        if mses:
            mean = np.mean(mses)
            std = np.std(mses)
            results[name] = f"{mean:.2f} ± {std:.2f}"
        else:
            results[name] = "Diverged"

    print("\nFinal Ablation Results:")
    print(json.dumps(results, indent=2))
    
    with os.fdopen(os.open("results/ablation_stats_rigorous.json", os.O_WRONLY | os.O_CREAT, 0o644), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_rigorous_ablation()
