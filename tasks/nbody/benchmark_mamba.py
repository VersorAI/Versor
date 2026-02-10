import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from data_gen import generate_gravity_data
from models import MambaSimulator

def compute_energy(data, G=1.0):
    pos = data[..., :3]
    vel = data[..., 3:]
    v_sq = torch.sum(vel**2, dim=-1)
    ke = 0.5 * torch.sum(v_sq, dim=-1)
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
            next_step = out[:, -1:, :, :]
            preds.append(next_step)
            current_seq = torch.cat([current_seq, next_step], dim=1)
            if current_seq.shape[1] > 100:
                current_seq = current_seq[:, -100:, :, :]
    return torch.cat(preds, dim=1)

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking Mamba on {device}...")
    
    # Model Params
    n_particles = 5
    mamba = MambaSimulator(n_particles=n_particles).to(device)
    
    # Parameter Count
    param_count = sum(p.numel() for p in mamba.parameters())
    print(f"Mamba Parameter Count: {param_count}")
    
    # Data
    STEPS = 100
    train_data = generate_gravity_data(n_samples=200, n_steps=STEPS, device=device)
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    optimizer = optim.Adam(mamba.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Training
    print("Training Mamba for 30 epochs...")
    for epoch in range(30):
        mamba.train()
        optimizer.zero_grad()
        pred = mamba(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
            
    # Evaluation
    print("Evaluating rollout stability...")
    test_data = generate_gravity_data(n_samples=10, n_steps=200, device=device)
    seed = test_data[:, :100]
    ground_truth = test_data[:, 100:]
    
    # Measure Runtime
    # We measure the time for a single 100-step rollout
    start_time = time.time()
    p_mamba = autoregressive_rollout(mamba, seed, steps=100)
    end_time = time.time()
    
    runtime_ms = (end_time - start_time) * 1000 / seed.shape[0] # Average per sample
    
    # Metrics
    mse = loss_fn(p_mamba, ground_truth).item()
    e_start = compute_energy(seed[:, -1:])
    e_end = compute_energy(p_mamba[:, -1:])
    drift = torch.mean(torch.abs(e_end - e_start)).item()
    
    print("\nBENCHMARK RESULTS (Mamba):")
    print("-" * 30)
    print(f"MSE:          {mse:.6f}")
    print(f"Energy Drift: {drift:.6f}")
    print(f"Runtime:      {runtime_ms:.2f} ms / sample")
    print(f"Params:       {param_count}")
    print("-" * 30)

if __name__ == "__main__":
    run_benchmark()
