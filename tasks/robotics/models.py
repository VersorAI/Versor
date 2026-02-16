import torch
import torch.nn as nn
import sys
import os

# Append parent directory and library directory to system path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "library"))

from Model.model import normalize_cl41
from tasks.nbody import algebra # Reusing the GA kernels from nbody task

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
        # Using the Python fallback logic for visibility in this benchmark
        for t in range(S):
            # Incremental rotor generator
            u_t = u[:, t]
            
            # Map to manifold (Identity + epsilon)
            delta_r = u_t.clone()
            delta_r[..., 0] += 1.0 
            delta_r = normalize_cl41(delta_r)
            
            # Group action: Multiplicative accumulation
            # psi_{t+1} = delta_r * psi_t
            psi = algebra.geometric_product(delta_r, psi)
            psi = normalize_cl41(psi)
            
            # Project high-dim hidden state to target 32D rotor 
            # and project back to manifold
            out_rotor = self.proj_out(psi.reshape(B, -1))
            out_rotor = normalize_cl41(out_rotor)
            outputs.append(out_rotor)
            
        return torch.stack(outputs, dim=1)

def measure_manifold_drift(rotor_batch):
    """
    Measures how far the predicted rotors are from a valid Spin(4,1) element.
    In Cl(4,1), rotors satisfy R * reverse(R) = 1.
    """
    # For SO(3) sub-group, this measures deviation from orthogonality
    # Here we'll use a simplified check on the scalar part of (R * rev(R))
    # This is simplified for the PoC.
    B, S, D = rotor_batch.shape
    # (Simplified metric for this script)
    # We want the norm of the multivector to stay near 1
    norms = torch.norm(rotor_batch, dim=-1) # (B, S)
    drift = torch.abs(norms - 1.0).mean()
    return drift.item()
