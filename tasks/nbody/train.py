import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from data_gen import generate_gravity_data
from models import StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, HamiltonianNN, HamiltonianVersorNN

def compute_energy(data, mass=1.0, G=1.0):
    """
    Computes total energy of the system.
    Data: (B, T, N, 6) -> (pos, vel)
    Returns: (B, T) energy
    """
    pos = data[..., :3]
    vel = data[..., 3:]
    
    # Kinetic Energy calculation: T = 0.5 * \sum m_i * v_i^2
    # Assumptions: Uniform mass distribution (m=1.0) for relative stability metrics.
    # Conservation analysis is performed relative to initial state t=0.
    
    v_sq = torch.sum(vel**2, dim=-1) # (B, T, N)
    ke = 0.5 * torch.sum(v_sq, dim=-1) # (B, T)
    
    # Potential Energy: - G * mi * mj / r
    pe = torch.zeros_like(ke)
    B, T, N, _ = pos.shape
    
    # Calculation of pairwise potential energy
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[..., i, :] - pos[..., j, :]
            dist = torch.norm(diff, dim=-1) + 1e-3
            pe -= (G * 1.0 * 1.0) / dist
            
    return ke + pe

def autoregressive_rollout(model, seed_data, steps=100):
    """
    Predicts next 'steps' frames using the model autoregressively.
    seed_data: (B, Seed_Steps, N, 6)
    """
    current_seq = seed_data
    preds = []
    
    with torch.no_grad():
        for _ in range(steps):
            # Sequence prediction via recursive model invocation.
            # Performance note: O(L) or O(L^2) complexity depending on model architecture.
            
            out = model(current_seq)
            next_step = out[:, -1:, :, :] # (B, 1, N, 6)
            preds.append(next_step)
            
            current_seq = torch.cat([current_seq, next_step], dim=1)
            # Context window management
            if current_seq.shape[1] > 100:
                current_seq = current_seq[:, -100:, :, :]
                
    return torch.cat(preds, dim=1)

def train(seed=42, model_types=['std', 'versor', 'gns', 'hnn', 'versor_hnn']):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Using device: {device} | Seed: {seed}")
    
    # Hyperparams
    BATCH_SIZE = 16
    STEPS = 100
    EPOCHS = 5 
    LR = 1e-3
    
    # Generate Training Data
    print("Generating training data...")
    train_data = generate_gravity_data(n_samples=200, n_steps=STEPS, device=device)
    val_data = generate_gravity_data(n_samples=50, n_steps=STEPS, device=device)
    
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    # Init Models
    models = {}
    optimizers = {}
    if 'std' in model_types:
        models['std'] = StandardTransformer(n_particles=5).to(device)
        optimizers['std'] = optim.Adam(models['std'].parameters(), lr=LR)
    if 'versor' in model_types:
        models['versor'] = VersorRotorRNN(n_particles=5).to(device)
        optimizers['versor'] = optim.Adam(models['versor'].parameters(), lr=LR)
    if 'gns' in model_types:
        models['gns'] = GraphNetworkSimulator(n_particles=5).to(device)
        optimizers['gns'] = optim.Adam(models['gns'].parameters(), lr=LR)
    if 'hnn' in model_types:
        models['hnn'] = HamiltonianNN(n_particles=5).to(device)
        optimizers['hnn'] = optim.Adam(models['hnn'].parameters(), lr=LR)
    if 'versor_hnn' in model_types:
        models['versor_hnn'] = HamiltonianVersorNN(n_particles=5).to(device)
        optimizers['versor_hnn'] = optim.Adam(models['versor_hnn'].parameters(), lr=LR)
    if 'egnn' in model_types:
        from models import EquivariantGNN
        models['egnn'] = EquivariantGNN(n_particles=5).to(device)
        optimizers['egnn'] = optim.Adam(models['egnn'].parameters(), lr=LR)
    if 'versor_mc' in model_types:
        from models import MultiChannelVersor
        # K=4 channels => D_model = 128
        models['versor_mc'] = MultiChannelVersor(n_particles=5, n_channels=4).to(device)
        optimizers['versor_mc'] = optim.Adam(models['versor_mc'].parameters(), lr=LR)
    
    loss_fn = nn.MSELoss()
    
    print(f"\nTraining models: {', '.join(models.keys())}")
    
    # Timing stats
    train_times = {name: 0.0 for name in models.keys()}
    inference_times = {name: 0.0 for name in models.keys()}

    for epoch in range(EPOCHS):
        for m in models.values():
            m.train()
        
        perm = torch.randperm(X_train.shape[0])
        batch_losses = {name: 0.0 for name in models.keys()}
        
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch_x = X_train[idx]
            batch_y = Y_train[idx]
            
            for name, model in models.items():
                t0 = time.time()
                optimizers[name].zero_grad()
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                train_times[name] += (time.time() - t0)
                batch_losses[name] += loss.item()
            
        if (epoch+1) % 10 == 0:
            n_batches = X_train.shape[0] // BATCH_SIZE
            avg_losses = {name: loss / n_batches for name, loss in batch_losses.items()}
            loss_str = " | ".join([f"{name}: {loss:.4f}" for name, loss in avg_losses.items()])
            print(f"Epoch {epoch+1:2d} | {loss_str}")
            
    # Evaluation
    for m in models.values():
        m.eval()
    
    test_data = generate_gravity_data(n_samples=20, n_steps=200, device=device)
    seed_window = test_data[:, :100]
    ground_truth = test_data[:, 100:]
    
    results = {}
    for name, model in models.items():
        t0 = time.time()
        preds = autoregressive_rollout(model, seed_window, steps=100)
        inference_times[name] = time.time() - t0
        
        mse = loss_fn(preds, ground_truth).item()
        
        seed_last = seed_window[:, -1:]
        e_start = compute_energy(seed_last)
        e_end = compute_energy(preds[:, -1:])
        drift = torch.mean(torch.abs(e_end - e_start)).item()
        
        results[name] = {
            "mse": mse, 
            "drift": drift, 
            "train_time_sec": train_times[name],
            "inf_time_sec": inference_times[name]
        }
        
    return results

if __name__ == "__main__":
    res = train()
    print("\nFinal Results:", res)
