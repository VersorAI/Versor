#!/usr/bin/env python3
"""
Modified training script that SAVES RESULTS to JSON
This is what should have been used for the paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from datetime import datetime
from data_gen import generate_gravity_data
from models import StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, HamiltonianNN

def compute_energy(data, mass=1.0, G=1.0):
    """Computes total energy of the system."""
    pos = data[..., :3]
    vel = data[..., 3:]
    
    v_sq = torch.sum(vel**2, dim=-1)
    ke = 0.5 * torch.sum(v_sq, dim=-1)
    
    pe = torch.zeros_like(ke)
    B, T, N, _ = pos.shape
    
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[..., i, :] - pos[..., j, :]
            dist = torch.norm(diff, dim=-1) + 1e-3  # Softening
            pe -= (G * 1.0 * 1.0) / dist
            
    return ke + pe

def autoregressive_rollout(model, seed_data, steps=100):
    """Predicts next 'steps' frames autoregressively."""
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

def train_model(model, optimizer, X_train, Y_train, epochs=30, batch_size=16, model_name="Model"):
    """Train a single model and return training history"""
    loss_fn = nn.MSELoss()
    history = []
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_train[idx]
            batch_y = Y_train[idx]
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  {model_name} - Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}")
    
    return history

def evaluate_model(model, test_data, seed_steps=100, rollout_steps=100):
    """Evaluate model on test set"""
    model.eval()
    
    seed = test_data[:, :seed_steps]
    ground_truth = test_data[:, seed_steps:seed_steps+rollout_steps]
    
    predictions = autoregressive_rollout(model, seed, steps=rollout_steps)
    
    # Calculate MSE
    loss_fn = nn.MSELoss()
    mse = loss_fn(predictions, ground_truth).item()
    
    # Calculate Energy Drift
    seed_last = seed[:, -1:]
    e_start = compute_energy(seed_last)
    e_end = compute_energy(predictions[:, -1:])
    
    # IMPORTANT: Calculate percentage drift correctly
    drift_abs = torch.mean(torch.abs(e_end - e_start)).item()
    drift_pct = (drift_abs / torch.mean(torch.abs(e_start)).item()) * 100
    
    # Also return raw energy values for transparency
    e_start_mean = torch.mean(e_start).item()
    e_end_mean = torch.mean(e_end).item()
    
    return {
        "mse": float(mse),
        "energy_drift_pct": float(drift_pct),
        "energy_drift_abs": float(drift_abs),
        "energy_start": float(e_start_mean),
        "energy_end": float(e_end_mean)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Hyperparameters
    SEED = 42
    BATCH_SIZE = 16
    TRAIN_SAMPLES = 200
    TRAIN_STEPS = 100
    TEST_SAMPLES = 10
    TEST_STEPS = 200
    EPOCHS = 30
    LR = 1e-3
    
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Generate data
    print("\nGenerating training data...")
    train_data = generate_gravity_data(n_samples=TRAIN_SAMPLES, n_steps=TRAIN_STEPS, device=device)
    test_data = generate_gravity_data(n_samples=TEST_SAMPLES, n_steps=TEST_STEPS, device=device)
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    # Initialize models
    models = {
        "Standard Transformer": StandardTransformer(n_particles=5).to(device),
        "Versor": VersorRotorRNN(n_particles=5).to(device),
        "GNS": GraphNetworkSimulator(n_particles=5).to(device),
        "HNN": HamiltonianNN(n_particles=5).to(device)
    }
    
    optimizers = {
        name: optim.Adam(model.parameters(), lr=LR)
        for name, model in models.items()
    }
    
    # Train all models
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    training_histories = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        history = train_model(
            model, optimizers[name], X_train, Y_train, 
            epochs=EPOCHS, batch_size=BATCH_SIZE, model_name=name
        )
        training_histories[name] = history
    
    # Evaluate all models
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, test_data, seed_steps=100, rollout_steps=100)
        results[name] = metrics
        
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Energy Drift: {metrics['energy_drift_pct']:.2f}%")
        print(f"  Energy Start: {metrics['energy_start']:.4f}")
        print(f"  Energy End: {metrics['energy_end']:.4f}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Model':<25} | {'MSE':<12} | {'Energy Drift %':<15}")
    print("-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['mse']:<12.6f} | {metrics['energy_drift_pct']:<15.2f}")
    
    # Save results to JSON
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "pytorch_version": torch.__version__,
            "seed": SEED
        },
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "train_samples": TRAIN_SAMPLES,
            "train_steps": TRAIN_STEPS,
            "test_samples": TEST_SAMPLES,
            "test_steps": TEST_STEPS,
            "epochs": EPOCHS,
            "learning_rate": LR
        },
        "training_histories": {
            name: [float(x) for x in history]
            for name, history in training_histories.items()
        },
        "results": results,
        "model_counts": {
            name: sum(p.numel() for p in model.parameters())
            for name, model in models.items()
        }
    }
    
    # Save to file
    output_file = f"nbody_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Check if results match paper claims
    print("\n" + "="*60)
    print("VERIFICATION AGAINST PAPER CLAIMS")
    print("="*60)
    
    paper_claims = {
        "Standard Transformer": {"mse": 0.04928, "drift": 438.94},  # From Table 2
        "Versor": {"mse": 0.03466, "drift": 348.86},
        "GNS": {"mse": 0.09070, "drift": 2068.35},
        "HNN": {"mse": 0.06051, "drift": 58.11}
    }
    
    readme_claims = {
        "Standard Transformer": {"mse": 18.63, "drift": 214.31},  # From README
        "Versor": {"mse": 5.45, "drift": 66.13},
        "GNS": {"mse": 23.83, "drift": 1261.01},
        "HNN": {"mse": 11.12, "drift": 61.86}
    }
    
    print("\nComparing to PAPER claims (Table 2, line 548):")
    for name in results:
        if name in paper_claims:
            actual_mse = results[name]["mse"]
            claimed_mse = paper_claims[name]["mse"]
            diff_mse = abs(actual_mse - claimed_mse)
            
            actual_drift = results[name]["energy_drift_pct"]
            claimed_drift = paper_claims[name]["drift"]
            diff_drift = abs(actual_drift - claimed_drift)
            
            print(f"\n{name}:")
            print(f"  MSE:    Actual={actual_mse:.6f}, Claimed={claimed_mse:.6f}, Diff={diff_mse:.6f}")
            print(f"  Drift:  Actual={actual_drift:.2f}%, Claimed={claimed_drift:.2f}%, Diff={diff_drift:.2f}%")
            
            if diff_mse > 0.01 or diff_drift > 50:
                print(f"  ⚠️  WARNING: Large discrepancy!")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()
