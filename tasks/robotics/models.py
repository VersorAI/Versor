import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Append parent directory and library directory to system path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "library"))

from gacore.kernel import manifold_normalization, geometric_product

class BaselineGRU(nn.Module):
    """
    Standard GRU baseline for Odometry.
    Learns to accumulate pose in a flat space.
    """
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (B, S, 6)
        out, _ = self.gru(x)
        # Predict absolute rotor: (B, S, 32)
        pred = self.fc(out)
        return pred

class VersorOdometry(nn.Module):
    """
    Versor-based Odometry model.
    Uses Recursive Rotor Accumulator (RRA) for manifold-constrained integration.
    """
    def __init__(self, input_dim=6, hidden_channels=8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.signature = [1, 1, 1, 1, -1] # Cl(4,1) for Versor
        
        # Project noisy velocities to incremental rotor parameters
        # (B, S, 6) -> (B, S, H, 32)
        self.proj_in = nn.Linear(input_dim, hidden_channels * 32)
        
        # Projection to final absolute rotor (32D multivector)
        self.proj_out = nn.Linear(hidden_channels * 32, 32)

    def forward(self, x):
        B, S, D = x.shape
        
        # 1. Project to GA incremental units
        # u_t: (B, S, H, 32)
        u = self.proj_in(x).reshape(B, S, self.hidden_channels, 32)
        
        # 2. Initial State (Identity Rotor)
        psi = torch.zeros(B, self.hidden_channels, 32, device=x.device)
        psi[..., 0] = 1.0 
        
        outputs = []
        
        # Recursive integration (RRA)
        for t in range(S):
            # Incremental rotor generator
            u_t = u[:, t]
            
            # Map to manifold (Identity + epsilon)
            delta_b = u_t.clone()
            
            # Apply safe generator squashing to non-compact components (Rotor Clamping)
            boost_mask = torch.zeros_like(delta_b)
            boost_mask[..., [17, 18, 20, 24]] = 1.0
            
            delta_b_compact = delta_b * (1.0 - boost_mask)
            delta_b_boost = delta_b * boost_mask
            
            boost_norm = torch.linalg.norm(delta_b_boost, dim=-1, keepdim=True) + 1e-6
            delta_b_boost_safe = delta_b_boost * (1.99 * torch.tanh(boost_norm) / boost_norm)
            
            delta_b = delta_b_compact + delta_b_boost_safe
            
            delta_r = delta_b
            delta_r[..., 0] += 1.0 
            delta_r = manifold_normalization(delta_r, self.signature)
            
            # Group action: Multiplicative accumulation
            psi = geometric_product(delta_r, psi, self.signature, method='bitmasked')
            psi = manifold_normalization(psi, self.signature)
            
            # Project high-dim hidden state to target 32D rotor 
            out_rotor = self.proj_out(psi.reshape(B, -1))
            out_rotor = manifold_normalization(out_rotor, self.signature)
            outputs.append(out_rotor)
            
        return torch.stack(outputs, dim=1)

def measure_manifold_drift(rotor_batch):
    """
    Rigorously measure how far predicted rotors are from the Spin manifold.
    In Cl(4,1), valid rotors satisfy R * reverse(R) = 1.
    """
    B, S, D = rotor_batch.shape
    # Sign mapping for Cl(4,1) squares
    sig_arr = np.array([1, 1, 1, 1, -1])
    sig32 = np.ones(D)
    for i in range(D):
        for b in range(5):
            if (i >> b) & 1:
                sig32[i] *= sig_arr[b]
    cl_metric_sig = torch.tensor(sig32, device=rotor_batch.device, dtype=rotor_batch.dtype)
    
    # Square norm 
    norm_sq = torch.sum(rotor_batch * rotor_batch * cl_metric_sig, dim=-1)
    drift = torch.abs(torch.sqrt(torch.abs(norm_sq) + 1e-6) - 1.0).mean()
    return drift.item()
