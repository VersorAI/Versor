import torch
import torch.nn as nn
import numpy as np
import gacore.kernel as kernel

# --- Physical Constants (AMBER ff14SB derived) ---
BOND_LENGTHS = {
    "N-CA": 1.458,
    "CA-C": 1.525,
    "C-N": 1.329,
    "C-O": 1.231,
    "CA-CB": 1.520,
}

# VDW Radii (Angstroms)
VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "H": 1.20,
}

class DifferentiablePhysicsEngine(nn.Module):
    """
    Generalized Differentiable Potential Energy Function.
    Works with any Clifford manifold.
    """
    def __init__(self, device="cuda", signature=None):
        super().__init__()
        self.device = device
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        self.dielectric = 80.0 
        
    def forward(self, coords, motors, mask=None):
        if coords.dim() == 5:
            B, S, T, A, D = coords.shape
            coords = coords.transpose(1, 2).reshape(B*T, S, A, D)
            motors = motors.transpose(1, 2).reshape(B*T, S, -1)
            mask = mask.unsqueeze(1).expand(B, T, S).reshape(B*T, S)
        
        B_curr, S_curr, _, _ = coords.shape
        if mask is None:
            mask = torch.ones((B_curr, S_curr), device=coords.device)
            
        lj_energy = self._compute_vdw_repulsion(coords, mask)
        hbond_energy = self._compute_hbond_energy(coords, mask)
        motor_smoothness = self._compute_motor_regularization(motors, mask)
        rg_loss = self._compute_rg_loss(coords[:, :, 1, :], mask) 
        
        total_energy = (
            1.0 * lj_energy + 
            0.5 * hbond_energy + 
            0.1 * motor_smoothness + 
            0.05 * rg_loss
        )
        
        return total_energy

    def _compute_vdw_repulsion(self, coords, mask, sigma=3.0, epsilon=1.0):
        B, S, A, _ = coords.shape
        flat_coords = coords.view(B, S*A, 3)
        flat_mask = mask.repeat_interleave(A, dim=1)
        
        dist_sq = torch.sum((flat_coords.unsqueeze(1) - flat_coords.unsqueeze(2))**2, dim=-1)
        r = torch.sqrt(dist_sq + 1e-6)
        overlap = torch.clamp(sigma - r, min=0.0)
        clash_energy = (overlap**4)
        
        pair_mask = flat_mask.unsqueeze(1) * flat_mask.unsqueeze(2)
        eye = torch.eye(S*A, device=coords.device).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - eye)
        
        return torch.sum(clash_energy * pair_mask) / (pair_mask.sum() + 1e-6)

    def _compute_hbond_energy(self, coords, mask):
        N = coords[:, :, 0, :]
        O = coords[:, :, 3, :]
        dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), dim=-1)
        h_energy = torch.exp(-0.5 * (dist - 2.9)**2 / (0.2**2))
        S = dist.shape[1]
        range_mask = torch.abs(torch.arange(S, device=self.device).view(1, -1) - 
                             torch.arange(S, device=self.device).view(-1, 1)) > 2
        pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2) * range_mask.unsqueeze(0)
        return -torch.sum(h_energy * pair_mask) / (mask.sum() + 1e-6)

    def _compute_motor_regularization(self, motors, mask):
        """Generalized manifold smoothness."""
        # For small updates, L2 distance between normalized multivectors 
        # is a good proxy for the manifold geodesic distance.
        diff = torch.sum((motors[:, 1:] - motors[:, :-1])**2, dim=-1)
        m_mask = mask[:, 1:] * mask[:, :-1]
        return torch.sum(diff * m_mask) / (m_mask.sum() + 1e-6)

    def _compute_rg_loss(self, ca_coords, mask):
        B, S, _ = ca_coords.shape
        mean_pos = torch.sum(ca_coords * mask.unsqueeze(-1), dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
        dist_sq = torch.sum((ca_coords - mean_pos.unsqueeze(1))**2, dim=-1)
        rg_sq = torch.sum(dist_sq * mask, dim=1) / (mask.sum(dim=1) + 1e-6)
        return torch.mean(rg_sq)

def apply_physics_constraints(coords, motors, mask, signature=None):
    engine = DifferentiablePhysicsEngine(device=coords.device, signature=signature)
    return engine(coords, motors, mask)

