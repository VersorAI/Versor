import torch
import numpy as np
import sys
import os

# Append parent directory and library directory to system path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "library"))

from library.gacore import cga
from Model.core import normalize_cl41

def generate_odometry_data(n_samples=500, n_steps=100, dt=0.1, noise_std=0.05, device='cpu'):
    """
    Generates 3D odometry data (position and orientation trajectories).
    Input: Noisy linear and angular velocities (v, omega).
    Target: Ground truth pose (Absolute Rotor in Cl(4,1)).
    """
    # 1. Initialize Pose: Identity Rotor 
    # In CGA, points are lifted to null vectors, but the pose is a rotor R.
    # Here we'll work with the 32D multivector representing the rotor in Cl(4,1).
    B = n_samples
    
    # Ground truth states
    gt_rotors = []
    
    # Current pose rotor (starts at Identity)
    # Shape: (B, 32)
    curr_rotor = torch.zeros(B, 32, device=device)
    curr_rotor[:, 0] = 1.0 # Scalar part = 1
    
    # Generate random smooth velocities
    # Linear velocity in local frame
    v_base = torch.randn(B, n_steps, 3, device=device) * 0.2
    # Angular velocity (bivectors in 3D, represented as 3 scalars for e12, e23, e13)
    omega_base = torch.randn(B, n_steps, 3, device=device) * 0.1
    
    # Low-pass filter to make trajectories smoother
    # (Optional, but makes it look more like real motion)
    for t in range(1, n_steps):
        v_base[:, t] = 0.8 * v_base[:, t-1] + 0.2 * v_base[:, t]
        omega_base[:, t] = 0.8 * omega_base[:, t-1] + 0.2 * omega_base[:, t]

    inputs = []
    targets = []
    
    for t in range(n_steps):
        # 1. Get current true velocities
        v_t = v_base[:, t]
        w_t = omega_base[:, t] 
        
        # 2. Add noise to observations (Inputs) for the model
        v_noisy = v_t + torch.randn_like(v_t) * noise_std
        w_noisy = w_t + torch.randn_like(w_t) * noise_std
        inputs.append(torch.cat([v_noisy, w_noisy], dim=-1))

        # 3. Update Ground Truth Rotor (Accumulate movement)
        # Construct bivector for current movement: v * e_inf + w_bivector
        # R_{t+1} = Delta_R * R_t
        # For this PoC, we'll use a simplified Rotor update: 
        # Delta_R = 1 + 0.5 * (v * dt * e_inf + w_bivector * dt)
        
        dr = torch.zeros(B, 32, device=device)
        dr[:, 0] = 1.0 # Scalar part
        # Simplified: put v in e1, e2, e3 for now to see accumulation
        # and w in e12, e23, e13. 
        # (A more rigorous CGA setup would use e_inf)
        dr[:, 1:4] = v_t * dt * 0.5 # Vector part
        dr[:, 5:8] = w_t * dt * 0.5 # Bivector part
        
        # Proper GA product for accumulation
        from tasks.nbody import algebra
        curr_rotor = algebra.geometric_product(dr, curr_rotor)
        curr_rotor = normalize_cl41(curr_rotor) # Use Model.core stabilization
        
        gt_rotors.append(curr_rotor.clone())
        
    inputs = torch.stack(inputs, dim=1) # (B, S, 6)
    targets = torch.stack(gt_rotors, dim=1) # (B, S, 32)
    
    return inputs, targets

if __name__ == "__main__":
    inputs, targets = generate_odometry_data(n_samples=5, n_steps=20)
    print(f"Inputs: {inputs.shape}")
    print(f"Targets: {targets.shape}")
