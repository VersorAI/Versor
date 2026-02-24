import torch
import torch.nn as nn
from .layers import VersorAttention, VersorLinear
import gacore.kernel as kernel

class VersorActivation(nn.Module):
    """
    Generalized Structural activation function.
    """
    def __init__(self, signature=None):
        super().__init__()
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        
    def forward(self, x):
        # Using the inner product to find the magnitude squared
        # <x * ~x>_0
        n_dims = x.shape[-1]
        sig_diag = kernel.get_sign_matrix(self.signature, "numpy")[:, 0]
        sig_diag = torch.from_numpy(sig_diag).to(x.device, dtype=x.dtype)
        
        norm_sq = torch.sum(x * x * sig_diag, dim=-1, keepdim=True)
        norm = torch.sqrt(torch.abs(norm_sq) + 1e-8)
        # ReLU gate on the magnitude
        gate = torch.relu(norm) / (norm + 1e-8)
        return x * gate

class VersorBlock(nn.Module):
    """
    General Geometric Block.
    """
    def __init__(self, embed_dim, n_heads, signature=None, expansion=4, init_values=1e-5, method=None):
        super().__init__()
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        self.ga_dim = 1 << len(self.signature)
        self.method = method
        
        self.attn = VersorAttention(embed_dim, n_heads, signature=self.signature, method=self.method)
        self.ln1 = nn.LayerNorm([embed_dim, self.ga_dim])
        self.ln2 = nn.LayerNorm([embed_dim, self.ga_dim])
        
        # LayerScale parameters
        self.gamma1 = nn.Parameter(init_values * torch.ones((embed_dim, self.ga_dim)))
        self.gamma2 = nn.Parameter(init_values * torch.ones((embed_dim, self.ga_dim)))
        
        self.mlp = nn.Sequential(
            VersorLinear(embed_dim, expansion * embed_dim, signature=self.signature, method=self.method),
            nn.Tanh(),
            VersorLinear(expansion * embed_dim, embed_dim, signature=self.signature, method=self.method)
        )
        
    def forward(self, x, mask=None, return_attention=False):
        if return_attention:
            attn_out, probs = self.attn(self.ln1(x), mask=mask, return_attention=True)
            x = x + self.gamma1 * attn_out
        else:
            x = x + self.gamma1 * self.attn(self.ln1(x), mask=mask)
        
        x = kernel.manifold_normalization(x, self.signature)
        
        x = x + self.gamma2 * self.mlp(self.ln2(x))
        x = kernel.manifold_normalization(x, self.signature)
        
        if return_attention:
            return x, probs
        return x

class RecursiveRotorAccumulator(nn.Module):
    """
    Generalized Parallel Recursive Rotor Accumulator.
    """
    def __init__(self, embed_dim, signature=None, method=None):
        super().__init__()
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        self.method = method
        self.rotor_gen = VersorLinear(embed_dim, embed_dim, signature=self.signature, method=self.method)
        
    def forward(self, x, mask=None):
        B, S, D, ga_dim = x.shape
        
        delta_b = self.rotor_gen(x) 
        rotors = torch.zeros_like(delta_b)
        rotors[..., 0] = 1.0 # Identity rotor scalar part
        rotors = rotors + 0.5 * delta_b
        rotors = kernel.manifold_normalization(rotors, self.signature)
        
        curr_rotors = rotors
        step = 1
        while step < S:
            left = curr_rotors[:, :-step]
            right = curr_rotors[:, step:]
            
            combined = kernel.geometric_product(right, left, self.signature, method=self.method)
            combined = kernel.manifold_normalization(combined, self.signature)
            
            new_rotors = curr_rotors.clone()
            new_rotors[:, step:] = combined
            curr_rotors = new_rotors
            step *= 2
            
        return curr_rotors

class VersorTransformer(nn.Module):
    """
    Fully Generalized Geometric Transformer.
    """
    def __init__(self, embed_dim, n_heads, n_layers, n_classes, signature=None, expansion=4, use_rotor_pool=True, method=None):
        super().__init__()
        self.signature = signature if signature is not None else [1, 1, 1, 1, -1]
        self.ga_dim = 1 << len(self.signature)
        self.use_rotor_pool = use_rotor_pool
        self.method = method
        
        self.blocks = nn.ModuleList([
            VersorBlock(embed_dim, n_heads, signature=self.signature, expansion=expansion, method=self.method) for _ in range(n_layers)
        ])
        
        if self.use_rotor_pool:
            self.pooler = RecursiveRotorAccumulator(embed_dim, signature=self.signature, method=self.method)
        
        self.classifier = nn.Linear(embed_dim * self.ga_dim, n_classes)
        
    def forward(self, x, return_attention=False):
        all_attn = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                all_attn.append(attn)
            else:
                x = block(x)
            
        if self.use_rotor_pool:
            x = self.pooler(x)[:, -1] # Final accumulated rotor
        else:
            x = x.mean(dim=1)
        
        logits = self.classifier(x.reshape(x.shape[0], -1))
        
        if return_attention:
            return logits, all_attn
        return logits


def get_grade_energies(x, signature):
    """
    Generalized grade energy analysis.
    """
    n_dims = x.shape[-1]
    energies = {}
    for i in range(n_dims):
        grade = bin(i).count('1')
        if grade not in energies:
            energies[grade] = 0.0
        # Sum of squares for components of this grade
        energies[grade] += torch.sum(x[..., i]**2).item()
    
    # Normalize by total and take mean across batch/seq
    total = sum(energies.values()) + 1e-8
    return {k: v/total for k, v in energies.items()}

