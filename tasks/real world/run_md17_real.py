import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Model.model import VersorTransformer

# Config
DATA_PATH = "/Users/mac/Desktop/Versor/data/md17/rmd17/npz_data/rmd17_salicylic.npz"
RESULTS_PATH = "results/real_md17_results.json"
DEVICE = torch.device("cpu") # Clifford kernels are stable on CPU/MPS
BATCH_SIZE = 32
TRAIN_SAMPLES = 50000 
VAL_SAMPLES = 10000
EMBED_DIM = 24
N_HEADS = 4
N_LAYERS = 2
LR = 5e-4
EPOCHS = 15

class MD17Dataset(torch.utils.data.Dataset):
    def __init__(self, coords, energies, mean=None, std=None):
        self.coords = torch.tensor(coords, dtype=torch.float32)
        # Use provided stats or compute from current data (for train set)
        self.energy_mean = mean if mean is not None else energies.mean()
        self.energy_std = std if std is not None else energies.std()
        self.energies = torch.tensor((energies - self.energy_mean) / self.energy_std, dtype=torch.float32)
        
    def __len__(self):
        return len(self.coords)
        
    def __getitem__(self, idx):
        return self.coords[idx], self.energies[idx]

class MD17VersorWrapper(nn.Module):
    def __init__(self, embed_dim=16, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(3, embed_dim * 32)
        # We use use_rotor_pool=True for MD17 because we are predicting a global 
        # scalar (energy) from a set of interacting particles. The RRA provides 
        # a permutation-invariant summary of the system.
        self.transformer = VersorTransformer(embed_dim, n_heads, n_layers, n_classes=1, use_rotor_pool=True)
        
    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        # Lifting coordinates to multivector space
        h = self.input_proj(x).view(B, N, -1, 32)
        # Transformer with global rotor pooling
        energy_out = self.transformer(h) # (B, 1)
        return energy_out.squeeze(-1)

def train_md17():
    print(f"Loading MD17 (Salicylic) from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    data = np.load(DATA_PATH)
    coords = data['coords']
    energies = data['energies']
    
    # Shuffle and Split
    perm = np.random.permutation(len(coords))
    train_idx = perm[:TRAIN_SAMPLES]
    val_idx = perm[TRAIN_SAMPLES:TRAIN_SAMPLES+VAL_SAMPLES]
    
    dataset_train = MD17Dataset(coords[train_idx], energies[train_idx])
    dataset_val = MD17Dataset(coords[val_idx], energies[val_idx], 
                              mean=dataset_train.energy_mean, 
                              std=dataset_train.energy_std)
    
    # Track scale for final MSE calculation
    energy_std = dataset_train.energy_std
    
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)
    
    model = MD17VersorWrapper(embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"Training on {len(train_idx)} samples... (STD: {energy_std:.4f})")
    
    start_time = time.time()
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        steps = 0
        for x, y in loader_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
            if steps % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {steps} | MSE: {loss.item():.6f}")
        
        avg_train_mse = (epoch_loss / steps) * (energy_std ** 2)
        
        # Validation
        model.eval()
        val_mse = 0
        val_steps = 0
        with torch.no_grad():
            for x, y in loader_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_mse += F.mse_loss(pred, y).item()
                val_steps += 1
        
        real_val_mse = (val_mse / val_steps) * (energy_std ** 2)
        print(f"Epoch {epoch+1} Result | Train MSE: {avg_train_mse:.4f} | Val MSE: {real_val_mse:.4f}")
        history.append({"epoch": epoch+1, "train_mse": avg_train_mse, "val_mse": real_val_mse})
        
        if real_val_mse < 1.80: # Early stop if we hit paper targets
             print("Hit paper target MSE (approx 1.76)!")
             # break

    # Final report
    results = {
        "final_val_mse": history[-1]["val_mse"],
        "training_time": time.time() - start_time,
        "history": history
    }
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    import json
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    train_md17()
