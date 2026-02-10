import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
import sys
import os

# Ensure paths are correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Physics.data_gen import generate_gravity_data
from Physics.models import MultiChannelVersor

def train_and_evaluate(model, X_train, Y_train, test_data, epochs=30, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test = test_data[:, :-1]
        Y_test = test_data[:, 1:]
        pred = model(X_test)
        mse = loss_fn(pred, Y_test).item()
    return mse

def main():
    device = "cpu"
    seeds = [42, 123, 456, 789, 1011]
    
    print("="*60)
    print("RUNNING LARGE SCALE MULTI-CHANNEL VERSOR EXPERIMENT")
    print("Configuration: 16 Channels, 4 Layers (Approx 1.14M params)")
    print("="*60)

    all_mses = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Larger model configuration: n_channels=16, n_layers=4
        model = MultiChannelVersor(n_particles=5, n_channels=16, n_layers=4).to(device)
        params = sum(p.numel() for p in model.parameters())
        
        train_data = generate_gravity_data(n_samples=200, n_steps=100, device=device)
        test_data = generate_gravity_data(n_samples=20, n_steps=100, device=device)
        
        X_train = train_data[:, :-1]
        Y_train = train_data[:, 1:]
        
        print(f"Seed {seed}: Params={params/1e6:.2f}M... training...", end=' ', flush=True)
        mse = train_and_evaluate(model, X_train, Y_train, test_data)
        all_mses.append(mse)
        print(f"Done. MSE: {mse:.4f}")

    mean_mse = np.mean(all_mses)
    std_mse = np.std(all_mses)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_channels": 16,
            "n_layers": 4,
            "n_particles": 5,
            "device": device
        },
        "seeds": seeds,
        "all_mses": all_mses,
        "mean_mse": mean_mse,
        "std_mse": std_mse
    }

    # Save to JSON
    os.makedirs('results', exist_ok=True)
    filename = f"results/large_scale_versor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print(f"FINAL RESULT (5 SEEDS):")
    print(f"MSE: {mean_mse:.4f} \u00b1 {std_mse:.4f}")
    print(f"✓ Results saved to {filename}")
    print("="*60)

if __name__ == "__main__":
    main()
