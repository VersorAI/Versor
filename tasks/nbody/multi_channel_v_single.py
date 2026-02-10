import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import time

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "Physics"))

from Physics.data_gen import generate_gravity_data
from Physics.models import VersorRotorRNN, MultiChannelVersor

def benchmark_density(n_particles=15, epochs=30):
    device = "cpu"
    print(f"\n" + "="*50)
    print(f"🚀 DENSITY STRESS TEST: N={n_particles} BODIES")
    print(f"Goal: Show Multi-Channel vs Single-Channel advantage")
    print("="*50)
    
    # Generate dense data
    print(f"Generating {n_particles}-body data...")
    train_data = generate_gravity_data(n_samples=200, n_steps=100, n_particles=n_particles, device=device)
    test_data = generate_gravity_data(n_samples=20, n_steps=100, n_particles=n_particles, device=device)
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    X_test = test_data[:, :-1]
    Y_test = test_data[:, 1:]

    # Models
    # 1. Single Channel RNN (The baseline)
    model_rnn = VersorRotorRNN(n_particles=n_particles, hidden_channels=16).to(device)
    
    # 2. Multi-Channel Transformer (The "4-ch" from the paper)
    # We increase the layers slightly to match the RNN's depth-like recurrence
    model_multi = MultiChannelVersor(n_particles=n_particles, n_channels=4, n_layers=2).to(device)
    
    models = {
        "Versor (1-ch RNN)": model_rnn,
        "Versor (4-ch Transformer)": model_multi
    }
    
    results = {}
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"\n--- Training {name} ({params/1e3:.1f}k params) ---")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        start_time = time.time()
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.6f}")
        
        train_time = time.time() - start_time
        
        model.eval()
        with torch.no_grad():
            mse = loss_fn(model(X_test), Y_test).item()
            print(f"  Final MSE: {mse:.6f} (Time: {train_time:.1f}s)")
            results[name] = mse

    print("\n" + "="*50)
    print(f"🏆 FINAL VERDICT (N={n_particles})")
    print("="*50)
    diff = (results["Versor (1-ch RNN)"] - results["Versor (4-ch Transformer)"]) / results["Versor (1-ch RNN)"] * 100
    print(f"1-ch RNN MSE: {results['Versor (1-ch RNN)']:.6f}")
    print(f"4-ch Trans MSE: {results['Versor (4-ch Transformer)']:.6f}")
    
    if diff > 0:
        print(f"✅ Multi-Channel is {diff:.1f}% BETTER than Single-Channel.")
    else:
        print(f"❌ Multi-Channel is {abs(diff):.1f}% WORSE (Saturation still active).")
    print("="*50)

if __name__ == "__main__":
    benchmark_density(n_particles=15) # Increase n to 15 to stress the manifold
