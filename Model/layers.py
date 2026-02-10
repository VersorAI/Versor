import torch
import torch.nn as nn
from .core import gp_cl41, wedge_cl41, inner_cl41, normalize_cl41, GRADE_INDICES, get_gp_map

class VersorLinear(nn.Module):
    """
    In-features: d_in, Out-features: d_out
    Parameters are represented as multivectors in the Clifford basis. 
    The operation implements a linear contraction via the Geometric Product (GP), 
    preserving geometric covariance through the transformation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialization using grade-aware variance scaling
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 32))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Versor Initialization (Paper Appendix G).
        Initializes weights to preserve variance across the Geometric Product.
        Formula: sigma = sqrt(2 / (fan_in * 32))
        """
        with torch.no_grad():
            # Factor of 2 for He initialization (ReLU)
            # Factor of 1/32 for the Cl(4,1) algebra dimension
            std = (2.0 / (self.in_features * 32)) ** 0.5
            self.weight.data.normal_(0.0, std)

    def forward(self, x):
        """
        Efficient forward pass utilizing the pre-computed Cayley transformation.
        Optimization reduces the contraction complexity from O(32^3) per 
        element to a linearized manifold application.
        """
        # Try to use the optimized kernel from gacore library
        try:
            import sys
            import os
            # Ensure library is in path to find gacore
            # Assuming file layout: Versor/Model/layers.py
            # library is at Versor/library
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            lib_dir = os.path.join(root_dir, "library")
            if lib_dir not in sys.path:
                sys.path.append(lib_dir)
            import gacore.kernel as kernel
            # Default signature for Cl(4,1)
            signature = [1, 1, 1, 1, -1]
            out = kernel.geometric_linear_layer(x, self.weight, signature)
            return normalize_cl41(out)
        except ImportError as e:
            # Fallback to local logic if gacore not found
            pass

        device = x.device
        batch, seq, _, _ = x.shape
        
        # 1. Computation of the 'Cayley-Baked' Linear Operator Matrix (O, I, 32, 32)
        gp_map = get_gp_map(device, x.dtype)
        # (O, I, J) * (J, L, K) -> (O, I, L, K)
        # J is the weight lane, L is the input lane, K is the output lane
        W_op = torch.einsum('o i j, j l k -> o i l k', self.weight, gp_map)
        
        # 2. Application of the manifold operator via optimized Einstein summation
        out = torch.einsum('b s i l, o i l k -> b s o k', x, W_op)
        
        return normalize_cl41(out)

    def __repr__(self):
        return f"VersorLinear(in_features={self.in_features}, out_features={self.out_features})"


class VersorAttention(nn.Module):
    """
    Geometric Product Attention (GPA).
    
    Instead of standard dot-product attention, GPA uses the full Geometric
    Product Q * K to compute attention scores. This incorporates:
    1.  The Scalar Projection (Standard Attention).
    2.  The Bivector Rotation (Geometric Coupling).
    
    This allows the model to attend to "orientational" features in GA space.
    """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.q_proj = VersorLinear(embed_dim, embed_dim)
        self.k_proj = VersorLinear(embed_dim, embed_dim)
        self.v_proj = VersorLinear(embed_dim, embed_dim)
        self.o_proj = VersorLinear(embed_dim, embed_dim)
        
        # Scaling parameter for the bivector influence
        self.attn_lambda = nn.Parameter(torch.tensor(0.1))
        self.bivector_indices = GRADE_INDICES[2]
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x (Tensor): Multivector sequence (B, S, D, 32)
            return_attention (bool): Whether to return the attention weights.
        """
        batch, seq, embed_dim, _ = x.shape
        
        # Project and restructure for Multi-Head Attention
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        
        # Optimized High-Speed Scoring
        # 1. Scalar Score: <Q * K>_0 = sum(Q * signature * K)
        from .core import get_signature
        sig = get_signature(q.device)
        
        # Reshape for high-speed matrix multiply
        # (B, H, S, D, 32) -> (B, H, S, D*32)
        q_flat = (q * sig).reshape(batch, self.n_heads, seq, -1)
        k_flat = k.reshape(batch, self.n_heads, seq, -1)
        
        # Acceleration via standard matrix factorization over weighted components
        scalar_score = torch.matmul(q_flat, k_flat.transpose(-1, -2))
        
        # GPA Score: Incorporates both scalar similarity and orientational torque.
        # Torque is represented by the magnitude of the bivector components 
        # in the Geometric Product Q * K.
        
        # 2. Bivector Score (Torque/Orientation)
        # For physics, this is critical to capture orthogonal interactions.
        # We use the relation ||Q ^ K||^2 = ||Q||^2 ||K||^2 - (Q · K)^2
        
        score = scalar_score / (self.head_dim ** 0.5)
        
        if self.attn_lambda != 0:
            q_norm_sq = torch.sum(q_flat**2, dim=-1, keepdim=True)
            k_norm_sq = torch.sum(k_flat**2, dim=-1, keepdim=True)
            dot_sq = scalar_score**2
            # ||Q ^ K|| is the magnitude of the bivector, representing the 'torque'
            torque_sq = torch.relu(q_norm_sq * k_norm_sq.transpose(-1, -2) - dot_sq)
            torque_score = torch.sqrt(torque_sq + 1e-6) / (self.head_dim ** 0.5)
            
            score = score + self.attn_lambda * torque_score
        
        attn_probs = torch.softmax(score, dim=-1)
        
        # Weighted accumulation of Value multivectors
        out = torch.einsum('b h s i , b h i d l -> b h s d l', attn_probs, v)
        
        # Recombine heads and final projection
        out = out.transpose(1, 2).contiguous().view(batch, seq, embed_dim, 32)
        out = self.o_proj(out)
        
        if return_attention:
            # Return both the standard attention probability and the raw bivector intensity
            # for the "Qualitative Proof" plot.
            return out, (attn_probs, torque_score if self.attn_lambda != 0 else torch.zeros_like(attn_probs))
        return out

    def __repr__(self):
        return f"VersorAttention(embed_dim={self.embed_dim}, heads={self.n_heads})"

