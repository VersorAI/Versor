import torch
import torch.nn as nn
import gacore.kernel as kernel

class CliffordKinematicChain(nn.Module):
    """
    Generalized Kinematic Chain.
    Works with any Clifford signature provided in the constructor.
    """
    def __init__(self, signature=None):
        super().__init__()
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        self.ga_dim = 1 << len(self.signature)
        
        # Local relative positions for backbone atoms (Ideal peptide geometry)
        backbone_local = torch.tensor([
            [-0.529, -1.359, 1e-6], # N
            [0.000, 0.000, 0.000],  # CA
            [1.525, 0.000, 0.000],   # C
            [2.153, 1.062, 1e-6],   # O
        ], dtype=torch.float32)
        
        # Lift atoms to GA points
        self.register_buffer("backbone_templates", self._lift_to_ga(backbone_local))

    def _lift_to_ga(self, x):
        """ Lifts coordinates to the multivector basis. """
        # We assume standard bitmask indexing (2^i)
        B, A, _ = x.shape if x.dim() == 3 else (1, *x.shape)
        device = x.device
        p = torch.zeros(B, A, self.ga_dim, device=device)
        
        # Identify "spatial" indices (signature == 1)
        spatial_indices = [1 << i for i, s in enumerate(self.signature) if s == 1]
        # Identify "conformal" indices if they exist (usually the last two)
        # e+ (8) and e- (16) in Cl(4,1)
        # For generality, we just map the first 3 spatial indices to x, y, z
        for i in range(min(len(spatial_indices), 3)):
            p[..., spatial_indices[i]] = x[..., i]
            
        # If it's a conformal-like model (has negative signature)
        # We perform a standard CGA lift if we have room
        neg_indices = [1 << i for i, s in enumerate(self.signature) if s == -1]
        pos_indices = [1 << i for i, s in enumerate(self.signature) if s == 1]
        
        if len(neg_indices) >= 1 and len(pos_indices) >= 4:
            # Assume Cl(n, 1) or similar conformal setup
            # e_inf = e+ + e-; n_o = 0.5 * (e- - e+)
            # n_inf index: pos_indices[3], n_o index: neg_indices[0]
            # This is specifically for Cl(4,1) style
            r2 = torch.sum(x**2, dim=-1, keepdim=True)
            p[..., pos_indices[3]] = 0.5 * r2.squeeze(-1) - 0.5 # e+
            p[..., neg_indices[0]] = 0.5 * r2.squeeze(-1) + 0.5 # e-
            
        return p

    def forward(self, motors):
        B, S, ga_dim = motors.shape
        device = motors.device
        
        # 1. Cumulative Motor Product
        # Note: In generalized mode, we use the recursive doubling for O(log S) 
        # or a simple loop for clarity in forward kinematics.
        curr_motor = torch.zeros(B, ga_dim, device=device)
        curr_motor[:, 0] = 1.0 
        
        chain_motors = []
        for i in range(S):
            curr_motor = kernel.geometric_product(motors[:, i], curr_motor, self.signature)
            curr_motor = kernel.manifold_normalization(curr_motor, self.signature)
            chain_motors.append(curr_motor)
            
        chain_motors = torch.stack(chain_motors, dim=1) 
        
        # 2. Map Multi-Atom Template through the Chain
        templates = self.backbone_templates.view(1, 1, 4, ga_dim).expand(B, S, 4, ga_dim)
        m_stack = chain_motors.unsqueeze(2).expand(B, S, 4, ga_dim)
        
        # M * P * ~M
        # kernel.reverse(m) is the adjoint
        m_rev = kernel.get_sign_matrix(self.signature, "numpy")[:, 0] # Proxy for reverse signs
        # Actually kernel has a reverse? No, let's use the property
        # For rotors, ~M is just sign flips on certain grades.
        # But gacore.kernel handles basic products.
        
        # Let's use a simpler path: ~M in GA
        grade_mask = torch.ones(ga_dim, device=device)
        for i in range(ga_dim):
            g = bin(i).count('1')
            if (g * (g - 1) // 2) % 2 == 1:
                grade_mask[i] = -1.0
        m_rev_stack = m_stack * grade_mask
        
        p_cga = kernel.geometric_product(kernel.geometric_product(m_stack, templates, self.signature), 
                                         m_rev_stack, self.signature)
        
        # Extract Euclidean (Standard spatial components)
        spatial_indices = [1 << i for i, s in enumerate(self.signature) if s == 1]
        coords = p_cga[..., spatial_indices[:3]]
        return coords, chain_motors

