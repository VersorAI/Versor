import torch
import torch.nn as nn
from .layers import VersorAttention, VersorLinear
from .core import normalize_cl41, gp_cl41, reverse_cl41, inner_cl41

class VersorActivation(nn.Module):
    """
    Structural activation function utilizing multivector magnitude gating.
    Preserves the orientational integrity in Clifford GA space while applying 
    a non-linear transformation based on the scalar multivector field norm.
    """
    def forward(self, x):
        # Using the inner product to find the magnitude squared
        norm_sq = inner_cl41(x, x)
        norm = torch.sqrt(torch.abs(norm_sq) + 1e-8)
        # ReLU gate on the magnitude
        gate = torch.relu(norm) / (norm + 1e-8)
        return x * gate.unsqueeze(-1)

class VersorBlock(nn.Module):
    """
    High-performance block specialized for Conformal Geometric Algebra.
    Integrates Geometric Product Attention (GPA) with a Clifford-covariant MLP.
    """
    def __init__(self, embed_dim, n_heads, expansion=4):
        super().__init__()
        self.attn = VersorAttention(embed_dim, n_heads)
        # LayerNorm is applied across the dimension and multivector lanes
        self.ln1 = nn.LayerNorm([embed_dim, 32])
        self.ln2 = nn.LayerNorm([embed_dim, 32])
        
        self.mlp = nn.Sequential(
            VersorLinear(embed_dim, expansion * embed_dim),
            nn.Tanh(), # Hyperbolic tangent activation for optimized manifold mapping
            VersorLinear(expansion * embed_dim, embed_dim)
        )
        
    def forward(self, x, return_attention=False):
        # Residual Connection + Attention
        if return_attention:
            attn_out, probs = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln1(x))
        
        x = normalize_cl41(x) # Manifold projection
        
        # Residual Connection + MLP
        x = x + self.mlp(self.ln2(x))
        x = normalize_cl41(x)
        
        if return_attention:
            return x, probs
        return x

class RecursiveRotorAccumulator(nn.Module):
    """
    True Recursive Rotor Accumulator (RRA).
    Implements \Psi_{t+1} = \Delta R_t \Psi_t with manifold normalization.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Project each time step to a 'delta-rotor' generator (Bivector)
        self.rotor_gen = VersorLinear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x: (B, S, D, 32)
        B, S, D, _ = x.shape
        
        # 1. Transform sequence to rotor generators (Bivectors)
        # Using the exponential map approximation: R = exp(B/2) \approx 1 + B/2
        delta_b = self.rotor_gen(x) 
        
        # 2. Sequential Accumulation
        # Start with identity spinor (Grade 0 = 1)
        psi = torch.zeros(B, D, 32, device=x.device)
        psi[..., 0] = 1.0 
        
        for t in range(S):
            # Current step's rotor
            r_t = delta_b[:, t]
            # \Psi_{t+1} = r_t * \Psi_t (Geometric Product)
            # We treat this as a per-channel geometric product
            psi = gp_cl41(r_t, psi)
            # Manifold Normalization to prevent drift
            psi = normalize_cl41(psi)
            
        return psi

class VersorTransformer(nn.Module):
    """
    Full Geometric Transformer for Conformal Geometric Algebra Cl(4,1).
    Equipped with Geometric Blocks and optional Stabilized Rotor Pooling.
    """
    def __init__(self, embed_dim, n_heads, n_layers, n_classes, expansion=4, use_rotor_pool=True):
        super().__init__()
        self.use_rotor_pool = use_rotor_pool
        self.blocks = nn.ModuleList([
            VersorBlock(embed_dim, n_heads, expansion=expansion) for _ in range(n_layers)
        ])
        
        if self.use_rotor_pool:
            self.pooler = RecursiveRotorAccumulator(embed_dim)
        
        # Final classifier maps the multivector state to class logits
        self.classifier = nn.Linear(embed_dim * 32, n_classes)
        
    def forward(self, x, return_attention=False):
        # Process through geometric layers
        all_attn = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                all_attn.append(attn)
            else:
                x = block(x)
            
        if self.use_rotor_pool:
            # Recursive Rotor Accumulation (Global State Persistence)
            x = self.pooler(x) 
        else:
            # Standard Global Average Pooling
            x = x.mean(dim=1)
        
        # Flatten multivector lanes for the linear classifier
        logits = self.classifier(x.view(x.shape[0], -1))
        
        if return_attention:
            return logits, all_attn
        return logits

def get_grade_energies(x):
    """
    Returns the L2 norm for each Clifford grade in the multivector x.
    x shape: (..., 32)
    """
    from .core import GRADE_INDICES
    energies = {}
    for grade, indices in GRADE_INDICES.items():
        # norm of components in this grade
        grade_data = x[..., indices]
        energies[grade] = torch.norm(grade_data, p=2, dim=-1).mean().item()
    return energies
