import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import time
import json
from datetime import datetime

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "Physics"))

from Physics.data_gen import generate_gravity_data
from Physics.models import VersorRotorRNN, MultiChannelVersor

def train_and_eval(name, model, train_data, test_data, device, epochs=30):
    params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Testing {name} ({params/1e3:.1f}k params) ---")
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, 1:]
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
    
    elapsed = time.time() - start_time
    model.eval()
    with torch.no_grad():
        mse = loss_fn(model(X_test), Y_test).item()
    print(f"  Final MSE: {mse:.6f} (Time: {elapsed:.1f}s)")
    return {"mse": mse, "params": params, "time": elapsed}

def main():
    device = "cpu"
    n_particles = 15
    epochs = 30
    
    print(f"🚀 MULTI-CHANNEL SCALING STUDY (N={n_particles})")
    
    # Data Generation
    train_data = generate_gravity_data(n_samples=200, n_steps=100, n_particles=n_particles, device=device)
    test_data = generate_gravity_data(n_samples=20, n_steps=100, n_particles=n_particles, device=device)
    
    results = {}

    # 1. Baseline: Single-Channel RNN
    rnn_model = VersorRotorRNN(n_particles=n_particles, hidden_channels=16).to(device)
    results["1-ch RNN (Baseline)"] = train_and_eval("1-ch RNN", rnn_model, train_data, test_data, device, epochs)
    baseline_params = results["1-ch RNN (Baseline)"]["params"]

    # 2. Multi-Channel: Disadvantaged (Original 4-ch)
    mc_disadv = MultiChannelVersor(n_particles=n_particles, n_channels=4, n_layers=2).to(device)
    results["Multi-Channel (Disadvantaged)"] = train_and_eval("Multi-Channel (4-ch, Small)", mc_disadv, train_data, test_data, device, epochs)

    # 3. Multi-Channel: Parameter-Matched
    # Standard 4-ch is ~48k. RNN is ~109k. 
    # Let's increase layers or hidden projections to match.
    # hidden_channels=12 approx matches the budget
    mc_matched = MultiChannelVersor(n_particles=n_particles, n_channels=12, n_layers=2).to(device)
    results["Multi-Channel (Param-Matched)"] = train_and_eval("Multi-Channel (12-ch, Matched)", mc_matched, train_data, test_data, device, epochs)

    # 4. Multi-Channel: Unleashed
    mc_unleashed = MultiChannelVersor(n_particles=n_particles, n_channels=16, n_layers=4).to(device)
    results["Multi-Channel (Unleashed)"] = train_and_eval("Multi-Channel (16-ch, 4-layer)", mc_unleashed, train_data, test_data, device, epochs)

    # Save Results
    os.makedirs('results', exist_ok=True)
    filename = f"results/multi_channel_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("💎 SCALING STUDY COMPLETE")
    print("="*60)
    for name, data in results.items():
        print(f"{name:30} | MSE: {data['mse']:.6f} | Params: {data['params']/1e3:6.1f}k")
    print("="*60)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
