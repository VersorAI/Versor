import torch
import torch.nn as nn
import os
import sys

# Ensure library is in path to find gacore
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Prefer the "library new" if it exists, otherwise use "library"
if os.path.exists(os.path.join(root_dir, "library new")):
    lib_dir = os.path.join(root_dir, "library new")
else:
    lib_dir = os.path.join(root_dir, "library")

if lib_dir not in sys.path:
    sys.path.append(lib_dir)

import gacore.kernel as kernel

class VersorLinear(nn.Module):
    """
    General Clifford Linear Layer.
    - method: 'triton'/'bitmasked' for custom kernels, 'matrix' for matrix isomorphism turbo path.
    """
    def __init__(self, in_features, out_features, signature=None, method=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        self.ga_dim = 1 << len(self.signature)
        
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, self.ga_dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            std = (2.0 / (self.in_features * self.ga_dim)) ** 0.5
            self.weight.data.normal_(0.0, std)

    def forward(self, x):
        out = kernel.geometric_linear_layer(x, self.weight, self.signature, method=self.method)
        return kernel.manifold_normalization(out, self.signature)

    def __repr__(self):
        sig_str = "".join(['+' if s > 0 else '-' if s < 0 else '0' for s in self.signature])
        meth_str = f", method={self.method}" if self.method else ""
        return f"VersorLinear(in={self.in_features}, out={self.out_features}, Cl({sig_str}){meth_str})"


class VersorAttention(nn.Module):
    """
    Generalized Geometric-Aware Attention (GAA).
    """
    def __init__(self, embed_dim, n_heads, signature=None, method=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        self.method = method
        self.ga_dim = 1 << len(self.signature)
        
        self.q_proj = VersorLinear(embed_dim, embed_dim, signature=self.signature, method=self.method)
        self.k_proj = VersorLinear(embed_dim, embed_dim, signature=self.signature, method=self.method)
        self.v_proj = VersorLinear(embed_dim, embed_dim, signature=self.signature, method=self.method)
        self.o_proj = VersorLinear(embed_dim, embed_dim, signature=self.signature, method=self.method)
        
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x, mask=None, return_attention=False):
        batch, seq, embed_dim, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim, self.ga_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim, self.ga_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim, self.ga_dim).transpose(1, 2)
        
        q_flat = q.reshape(batch * self.n_heads, seq, self.head_dim, self.ga_dim)
        k_flat = k.reshape(batch * self.n_heads, seq, self.head_dim, self.ga_dim)
        
        # Scalar part contraction
        sig_diag = kernel.get_sign_matrix(self.signature, "numpy")[:, 0]
        sig_diag = torch.from_numpy(sig_diag).to(q.device, dtype=q.dtype)
        
        q_sig = q_flat * sig_diag.view(1, 1, 1, self.ga_dim)
        scalar_score = torch.einsum('b i d l, b j d l -> b i j', q_sig, k_flat)
        
        q_norm_sq = torch.sum(q_flat**2, dim=(-1, -2)).view(batch * self.n_heads, seq, 1)
        k_norm_sq = torch.sum(k_flat**2, dim=(-1, -2)).view(batch * self.n_heads, 1, seq)
        dot_sq = (scalar_score)**2
        bivector_norm = torch.sqrt(torch.relu(q_norm_sq * k_norm_sq - dot_sq) + 1e-6)
        
        score = (scalar_score + self.gamma * bivector_norm) / (self.head_dim ** 0.5)
        
        if seq > 1:
            triangle_update = torch.matmul(torch.softmax(score, dim=-1), score)
            score = score + 0.1 * triangle_update
        
        score = score.view(batch, self.n_heads, seq, seq)
        
        if mask is not None:
            score = score.masked_fill(mask.view(batch, 1, 1, seq) == 0, -1e9)
            
        attn_probs = torch.softmax(score, dim=-1)
        out = torch.einsum('b h i j, b h j d l -> b h i d l', attn_probs, v)
        
        out = out.transpose(1, 2).contiguous().view(batch, seq, embed_dim, self.ga_dim)
        out = self.o_proj(out)
        
        if return_attention:
            return out, attn_probs
        return out

    def __repr__(self):
        sig_str = "".join(['+' if s > 0 else '-' if s < 0 else '0' for s in self.signature])
        meth_str = f", method={self.method}" if self.method else ""
        return f"VersorAttention(heads={self.n_heads}, Cl({sig_str}){meth_str})"



