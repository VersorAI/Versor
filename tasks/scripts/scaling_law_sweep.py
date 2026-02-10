import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path to import Model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.model import VersorTransformer
from Physics.data_gen import generate_gravity_data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(config, n_samples=2000, n_epochs=20):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training config: {config}, Device: {device}")
    
    # Data Setup
    data = generate_gravity_data(n_samples=n_samples, n_steps=50, n_particles=5, device=device)
    
    train_size = int(0.8 * n_samples)
    data_train = data[:train_size]
    data_val = data[train_size:]
    
    # Model
    model = VersorTransformer(
        embed_dim=config["embed_dim"],
        n_heads=max(2, config["embed_dim"] // 8),
        n_layers=config["n_layers"],
        n_classes=6,
        use_rotor_pool=False
    ).to(device)
    
    params = count_parameters(model)
    print(f"Parameters: {params:,}")
    
    # Adjusted Learning Rate for deeper models
    lr = 1e-3 if config["n_layers"] < 8 else 5e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Preprocessing to multivector input
    input_proj = nn.Linear(6, config["embed_dim"] * 32).to(device)
    optimizer.add_param_group({'params': input_proj.parameters()})

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    batch_size = 32
    
    for epoch in range(n_epochs):
        model.train()
        input_proj.train()
        epoch_loss = 0
        
        # Shuffle train data
        indices = torch.randperm(train_size)
        train_data = data_train[indices]
        
        for i in range(0, train_size, batch_size):
            batch = train_data[i:i+batch_size]
            B, S, P, D = batch.shape
            
            x = batch[:, :-1].transpose(1, 2).reshape(B * P, S-1, D)
            y = batch[:, 1:].transpose(1, 2).reshape(B * P, S-1, D)
            
            optimizer.zero_grad()
            h = input_proj(x).view(B * P, S-1, config["embed_dim"], 32)
            
            for block in model.blocks:
                h = block(h)
            
            out = model.classifier(h.reshape(B * P, S-1, -1))
            loss = criterion(out, y)
            
            if torch.isnan(loss):
                print(f"NAN Loss detected at Epoch {epoch+1}")
                return {"params": params, "final_loss": 999, "val_loss": 999, "status": "NaN"}

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / (train_size // batch_size)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        input_proj.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(data_val), batch_size):
                batch = data_val[i:i+batch_size]
                B, S, P, D = batch.shape
                x = batch[:, :-1].transpose(1, 2).reshape(B * P, S-1, D)
                y = batch[:, 1:].transpose(1, 2).reshape(B * P, S-1, D)
                
                h = input_proj(x).view(B * P, S-1, config["embed_dim"], 32)
                for block in model.blocks:
                    h = block(h)
                out = model.classifier(h.reshape(B * P, S-1, -1))
                val_loss += criterion(out, y).item()
        
        avg_val_loss = val_loss / (len(data_val) // batch_size)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
    return {
        "params": params, 
        "final_loss": train_losses[-1], 
        "val_loss": val_losses[-1],
        "config": config
    }

def run_sweep():
    configs = [
        {"embed_dim": 8, "n_layers": 2},   # ~52k
        {"embed_dim": 16, "n_layers": 4},  # ~404k
        {"embed_dim": 24, "n_layers": 4},  # ~901k
    ]
    
    # Check for existing results to skip
    results = []
    if os.path.exists("results/scaling_law_results.json"):
        with open("results/scaling_law_results.json", "r") as f:
            existing_results = json.load(f)
            # Only keep the ones that actually converged well (not the stuck 3.2M one)
            results = [r for r in existing_results if r.get("final_loss", 999) < 20]
            print(f"Resuming sweep. Loaded {len(results)} previous results.")

    already_done = [r["config"] for r in results]
    
    for config in configs:
        if config in already_done:
            print(f"Skipping {config} as it's already done.")
            continue
            
        res = train_model(config, n_samples=1000, n_epochs=20)
        results.append(res)
        
        # Intermediate save
        os.makedirs("results", exist_ok=True)
        with open("results/scaling_law_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"✓ Completed: {config}, Params: {res['params']:,}, Loss: {res['final_loss']:.6f}, Val: {res['val_loss']:.6f}")
        
    print("\nScaling Law Sweep Complete!")

if __name__ == "__main__":
    run_sweep()
