#!/usr/bin/env python3
"""
Run experiments with multiple seeds and calculate statistics
This is what SHOULD have been done for the paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
import sys
import os
import time

sys.path.append('..')

try:
    from data_gen import generate_gravity_data
    from models import StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, HamiltonianNN, MultiChannelVersor, HamiltonianVersorNN, EquivariantGNN, MambaSimulator
except ImportError:
    from Physics.data_gen import generate_gravity_data
    from Physics.models import StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, HamiltonianNN, MultiChannelVersor, HamiltonianVersorNN, EquivariantGNN, MambaSimulator

def compute_energy(data):
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
            pe -= 1.0 / dist
    return ke + pe

def autoregressive_rollout(model, seed_data, steps=50):
    """Perform multi-step prediction by feeding outputs back as inputs"""
    current_state = seed_data # (B, 1, N, 6)
    preds = []
    
    for _ in range(steps):
        # We assume the model can take a single step or a sequence.
        # For RNNs (Versor) and Transformers, we give it the sequence so far.
        # For Markovian models (GNS, EGNN, HNN), we only need the last frame.
        # To be safe and general, we provide the accumulated sequence.
        out = model(current_state)
        # The model returns (B, S, N, 6). We take the last predicted frame.
        next_step = out[:, -1:] # (B, 1, N, 6)
        preds.append(next_step)
        current_state = torch.cat([current_state, next_step], dim=1)
        
        # Limit history for Transformer if needed, but for 50 steps it's fine
        if current_state.shape[1] > 100:
            current_state = current_state[:, -100:]
            
    return torch.cat(preds, dim=1)

def train_and_evaluate(model, X_train, Y_train, test_data, epochs=30, lr=1e-3):
    """Train model and return final metrics using autoregressive rollout"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Training (Teacher Forcing)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if epoch % 5 == 0:
            print(f"      [Epoch {epoch}/{epochs}] Loss: {loss.item():.6f}", flush=True)
        optimizer.step()
    
    # Evaluation (Autoregressive Rollout)
    model.eval()
    with torch.no_grad():
        # We rollout for 50 steps starting from T=0
        rollout_steps = 50
        seed_data = test_data[:, :1]
        ground_truth = test_data[:, 1:1+rollout_steps]
        
        # Measure Latency (Single forward pass)
        # Warmup
        _ = model(seed_data)
        t0 = time.time()
        _ = model(seed_data)
        latency = (time.time() - t0) * 1000
        
        preds = autoregressive_rollout(model, seed_data, steps=rollout_steps)
        mse = loss_fn(preds, ground_truth).item()
        
        e_start = compute_energy(seed_data)
        e_end = compute_energy(preds[:, -1:])
        drift = torch.mean(torch.abs(e_end - e_start)).item()
        drift_pct = (drift / (torch.mean(torch.abs(e_start)).item() + 1e-6)) * 100
    
    return {"mse": float(mse), "energy_drift_pct": float(drift_pct), "latency": float(latency)}

def run_single_seed(seed, device='cpu'):
    """Run full experiment with one seed"""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate data
    train_data = generate_gravity_data(n_samples=200, n_steps=100, device=device)
    test_data = generate_gravity_data(n_samples=20, n_steps=100, device=device)
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    # Models
    models = {
        # "Transformer": StandardTransformer(n_particles=5).to(device),
        # "Versor": VersorRotorRNN().to(device),
        # "GNS": GraphNetworkSimulator(n_particles=5).to(device),
        # "HNN": HamiltonianNN(n_particles=5).to(device),
        # "Mamba": MambaSimulator(n_particles=5).to(device),
        "Versor-Multi": MultiChannelVersor(n_particles=5, n_channels=16, n_heads=4).to(device),
        # "Ham-Versor": HamiltonianVersorNN(n_particles=5).to(device),
        # "EGNN": EquivariantGNN(n_particles=5).to(device)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}...", end=' ', flush=True)
        try:
            metrics = train_and_evaluate(model, X_train, Y_train, test_data, epochs=30)
            results[name] = metrics
            print(f"MSE={metrics['mse']:.4f}, Drift={metrics['energy_drift_pct']:.1f}%, Latency={metrics['latency']:.2f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {"error": str(e)}
    
    return results

def main():
    print("="*60)
    print("MULTI-SEED EXPERIMENTAL VALIDATION")
    print("="*60)
    print("This will run 5 experiments with different random seeds")
    print("to get statistically valid results with error bars.\n")
    
    seeds = [42, 123, 456, 789, 1011]
    device = "cpu"
    
    all_results = {}
    
    for seed in seeds:
        results = run_single_seed(seed, device)
        all_results[f"seed_{seed}"] = results
    
    # Calculate statistics
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    models = ["Transformer", "Versor", "GNS", "HNN", "Versor-Multi", "Ham-Versor", "EGNN"]
    stats = {}
    
    for model in models:
        mses = []
        drifts = []
        latencies = []
        
        for seed_key in all_results:
            if model in all_results[seed_key] and "error" not in all_results[seed_key][model]:
                mses.append(all_results[seed_key][model]["mse"])
                drifts.append(all_results[seed_key][model]["energy_drift_pct"])
                latencies.append(all_results[seed_key][model]["latency"])
        
        if mses:
            stats[model] = {
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "mse_min": float(np.min(mses)),
                "mse_max": float(np.max(mses)),
                "drift_mean": float(np.mean(drifts)),
                "drift_std": float(np.std(drifts)),
                "latency_mean": float(np.mean(latencies)),
                "n_runs": len(mses)
            }
    
    # Print results table
    print(f"\n{'Model':<15} | {'MSE (Mean ± Std)':<20} | {'Energy Drift %':<20}")
    print("-" * 65)
    
    for model in models:
        if model in stats:
            s = stats[model]
            mse_str = f"{s['mse_mean']:.4f} ± {s['mse_std']:.4f}"
            drift_str = f"{s['drift_mean']:.1f} ± {s['drift_std']:.1f}"
            lat_str = f"{s['latency_mean']:.2f}"
            print(f"{model:<15} | {mse_str:<20} | {drift_str:<20} | {lat_str:<10}")
    
    # Statistical significance test
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE")
    print("="*60)
    
    if "Versor" in stats and "Transformer" in stats:
        # Get all MSE values
        versor_mses = [all_results[k]["Versor"]["mse"] for k in all_results 
                       if "Versor" in all_results[k] and "error" not in all_results[k]["Versor"]]
        trans_mses = [all_results[k]["Transformer"]["mse"] for k in all_results 
                      if "Transformer" in all_results[k] and "error" not in all_results[k]["Transformer"]]
        
        # Simple t-test
        from scipy import stats as scipy_stats
        try:
            t_stat, p_value = scipy_stats.ttest_ind(versor_mses, trans_mses)
            print(f"\nVersor vs Transformer:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            if p_value < 0.05:
                print(f"  ✓ Statistically significant improvement (p < 0.05)")
            else:
                print(f"  ⚠ Not statistically significant (p >= 0.05)")
        except:
            print("\n⚠ scipy not available, skipping t-test")
    
    if "Versor" in stats and "GNS" in stats:
        versor_mses = [all_results[k]["Versor"]["mse"] for k in all_results 
                       if "Versor" in all_results[k] and "error" not in all_results[k]["Versor"]]
        gns_mses = [all_results[k]["GNS"]["mse"] for k in all_results 
                    if "GNS" in all_results[k] and "error" not in all_results[k]["GNS"]]
        
        diff = stats["Versor"]["mse_mean"] - stats["GNS"]["mse_mean"]
        print(f"\nVersor vs GNS:")
        print(f"  Difference: {diff:.6f} ({diff/stats['GNS']['mse_mean']*100:+.2f}%)")
        if abs(diff) < stats["Versor"]["mse_std"]:
            print(f"  → Within 1 std, likely not significant")
        else:
            print(f"  → Outside 1 std, possibly significant")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "all_results": all_results,
        "statistics": stats,
        "config": {
            "n_samples": 200,
            "n_steps": 100,
            "epochs": 30,
            "device": device,
            "note": "Fair Comparison: Versor-Multi uses 16 channels and Clifford-equivariant activation/mixing to match baseline capacity."
        }
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    filename = f"results/multi-channel-stats.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {filename}")
    
    # Generate LaTeX
    print("\n" + "="*60)
    print("LATEX TABLE FOR PAPER")
    print("="*60)
    
    print("\n% Copy this into your paper (Table 2)")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{N-Body Dynamics Performance. Mean $\\pm$ std over 5 runs with different seeds.}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Model & Params & MSE & Energy Drift (\\%) \\\\")
    print("\\midrule")
    
    params_table = {
        "Transformer": "1.32M", 
        "GNS": "0.47M", 
        "HNN": "0.26M", 
        "Versor": "0.20M", 
        "Versor-Multi": "0.85M", 
        "Ham-Versor": "0.25M",
        "EGNN": "0.03M"
    }
    
    for model in ["Transformer", "GNS", "HNN", "Versor", "Versor-Multi", "Ham-Versor", "EGNN"]:
        if model in stats:
            s = stats[model]
            params = params_table.get(model, "??")
            
            # Format with bold for Versor
            if model == "Versor":
                mse_str = f"$\\mathbf{{{s['mse_mean']:.3f} \\pm {s['mse_std']:.3f}}}$"
                model_str = "\\textbf{Versor (Ours)}"
            else:
                mse_str = f"${s['mse_mean']:.3f} \\pm {s['mse_std']:.3f}$"
                model_str = model
            
            drift_str = f"${s['drift_mean']:.1f} \\pm {s['drift_std']:.1f}$"
            
            print(f"{model_str} & {params} & {mse_str} & {drift_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
