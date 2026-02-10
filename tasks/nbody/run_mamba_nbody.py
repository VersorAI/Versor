
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import os
import json

# Ensure we can import from Physics directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mock_dependencies
mock_dependencies.apply_mocks()

from data_gen import generate_gravity_data
from models import MambaSimulator

def compute_energy(data, mass=1.0, G=1.0):
    """
    Computes total energy of the system.
    Data: (B, T, N, 6) -> (pos, vel)
    Returns: (B, T) energy
    """
    pos = data[..., :3]
    vel = data[..., 3:]
    
    v_sq = torch.sum(vel**2, dim=-1) # (B, T, N)
    ke = 0.5 * torch.sum(v_sq, dim=-1) # (B, T)
    
    pe = torch.zeros_like(ke)
    B, T, N, _ = pos.shape
    
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[..., i, :] - pos[..., j, :]
            dist = torch.norm(diff, dim=-1) + 1e-3
            pe -= (G * 1.0 * 1.0) / dist
            
    return ke + pe

def autoregressive_rollout(model, seed_data, steps=100):
    current_seq = seed_data
    preds = []
    
    with torch.no_grad():
        for _ in range(steps):
            out = model(current_seq)
            next_step = out[:, -1:, :, :] # (B, 1, N, 6)
            preds.append(next_step)
            
            current_seq = torch.cat([current_seq, next_step], dim=1)
            if current_seq.shape[1] > 100:
                current_seq = current_seq[:, -100:, :, :]
                
    return torch.cat(preds, dim=1)

def run_mamba_experiment():
    seeds = [42, 123, 456]
    mses = []
    drifts = []
    latencies_all = []
    
    print("="*60)
    print("RUNNING MAMBA (Table 2) - Multi-Seed")
    print("="*60)
    
    for seed in seeds:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"\nSeed: {seed}")
        
        # Hyperparams
        BATCH_SIZE = 16
        STEPS = 100
        EPOCHS = 30 # Match epochs=30 in run_multi_seed.py
        LR = 1e-3 # Match lr=1e-3 in run_multi_seed.py
        
        # Generate Training Data
        train_data = generate_gravity_data(n_samples=200, n_steps=STEPS, device=device)
        X_train = train_data[:, :-1]
        Y_train = train_data[:, 1:]
        
        # Init Mamba (Scaled to ~15K params)
        model = MambaSimulator(n_particles=5, input_dim=6, d_model=128, d_state=32).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params:,}")
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        
        # Train
        for epoch in range(EPOCHS):
            model.train()
            perm = torch.randperm(X_train.shape[0])
            for i in range(0, X_train.shape[0], BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                batch_x = X_train[idx]
                batch_y = Y_train[idx]
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_data = generate_gravity_data(n_samples=20, n_steps=200, device=device)
        seed_window = test_data[:, :100]
        ground_truth = test_data[:, 100:]
        
        # Latency measurement (Per-step, single sample)
        latencies = []
        one_frame = seed_window[:1, :1] # (1, 1, 5, 6)
        with torch.no_grad():
            # Warmup
            _ = model(one_frame)
            for _ in range(100):
                start = time.time()
                _ = model(one_frame)
                latencies.append((time.time() - start) * 1000) # ms
        
        avg_latency = np.mean(latencies)
        
        preds = autoregressive_rollout(model, seed_window, steps=100)
        
        mse = loss_fn(preds, ground_truth).item()
        
        seed_last = seed_window[:, -1:]
        e_start = compute_energy(seed_last)
        e_end = compute_energy(preds[:, -1:])
        drift_pct = torch.mean(torch.abs(e_end - e_start) / (torch.abs(e_start) + 1e-6)).item() * 100
        
        print(f"Seed {seed} -> MSE: {mse:.4f}, Drift: {drift_pct:.2f}%, Latency: {avg_latency:.2f}ms")
        mses.append(mse)
        drifts.append(drift_pct)
        latencies_all.append(avg_latency)
        
    print("\nMAMBA RESULTS:")
    print(f"MSE: {np.mean(mses):.2f} ± {np.std(mses):.2f}")
    print(f"Drift: {np.mean(drifts):.2f} ± {np.std(drifts):.2f}%")
    print(f"Latency: {np.mean(latencies_all):.2f} ms")
    
    # Save for LaTeX
    with open("Physics/results/mamba_stats.json", "w") as f:
        json.dump({
            "mean_mse": np.mean(mses),
            "std_mse": np.std(mses),
            "mean_drift": np.mean(drifts),
            "std_drift": np.std(drifts),
            "mean_latency": np.mean(latencies_all)
        }, f)

if __name__ == "__main__":
    run_mamba_experiment()
