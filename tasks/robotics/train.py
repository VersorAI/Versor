import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Project imports
# This PoC uses SIMULATED data to validate geometric integration properties.
# No real-world robotics dataset is used in this PoC; it is a controlled physics verification.
from data_gen import generate_odometry_data
from models import BaselineGRU, VersorOdometry, measure_manifold_drift

def train_robotics_benchmark(epochs=40, batch_size=32, n_steps=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Identify project root for paths
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_base = os.path.join(root_dir, "results", "robotics")
    os.makedirs(results_base, exist_ok=True)
    
    print(f"Robotics Benchmark | Device: {device} | Roots: {root_dir}")
    
    # 1. Dataset Generation 
    # (Synthetic IMU-like data for controlled validation of SE(3) constraints)
    print("Generating simulated IMU-like data (noisy velocities -> rotor accumulation)...")
    train_in, train_target = generate_odometry_data(n_samples=500, n_steps=n_steps, device=device)
    val_in, val_target = generate_odometry_data(n_samples=100, n_steps=n_steps, device=device)
    
    # 2. Model Initialization
    baseline = BaselineGRU().to(device)
    versor = VersorOdometry().to(device)
    
    models = {"GRU_Baseline": baseline, "Versor_RRA": versor}
    optimizers = {name: optim.Adam(m.parameters(), lr=1e-3) for name, m in models.items()}
    loss_fn = nn.MSELoss()
    
    # 3. Training Loop
    history = {name: [] for name in models.keys()}
    t_start = time.time()
    
    for epoch in range(epochs):
        for name, model in models.items():
            model.train()
            optimizer = optimizers[name]
            
            perm = torch.randperm(train_in.shape[0])
            epoch_loss = 0
            
            for i in range(0, train_in.shape[0], batch_size):
                idx = perm[i:i+batch_size]
                x = train_in[idx]
                y = train_target[idx]
                
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (train_in.shape[0] // batch_size)
            history[name].append(avg_loss)
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d} | GRU Loss: {history['GRU_Baseline'][-1]:.6f} | Versor Loss: {history['Versor_RRA'][-1]:.6f}")

    # 4. Evaluation 
    results_summary = {}
    print("\nEvaluating Final Performance (calculated on validation set outputs)...")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            preds = model(val_in)
            mse = loss_fn(preds, val_target).item()
            drift = measure_manifold_drift(preds)
            
            results_summary[name] = {
                "final_mse": mse, 
                "manifold_drift": drift,
                "history": history[name]
            }
            print(f"[{name}] Validation MSE: {mse:.6f} | Manifold Drift: {drift:.6f}")

    # 5. Audit Log (Transparency File)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_base, f"odometry_audit_{timestamp}.json")
    
    audit_data = {
        "metadata": {
            "timestamp": timestamp,
            "device": device,
            "data_type": "SimulatedDeadReckoning",
            "hyperparams": {"epochs": epochs, "batch_size": batch_size, "n_steps": n_steps},
            "system": str(sys.version)
        },
        "metrics": {name: {"mse": d["final_mse"], "drift": d["manifold_drift"]} for name, d in results_summary.items()},
        "training_history": history
    }
    
    with open(log_file, 'w') as f:
        json.dump(audit_data, f, indent=4)
    print(f"✓ Audit log (JSON) saved: {log_file}")

    # 6. Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for name, losses in history.items():
        plt.plot(losses, label=name)
    plt.title("Path Estimation Training Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    names = list(results_summary.keys())
    drifts = [results_summary[n]["manifold_drift"] for n in names]
    plt.bar(names, drifts, color=['#e74c3c', '#2ecc71'])
    plt.title("Final Manifold Deviation")
    plt.ylabel("||R*rev(R) - 1||")
    
    plot_path = os.path.join(results_base, "odometry_benchmark.png")
    plt.savefig(plot_path)
    print(f"✓ Visualization saved: {plot_path}")
    
    return audit_data

if __name__ == "__main__":
    train_robotics_benchmark(epochs=40)
