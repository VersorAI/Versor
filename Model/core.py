import torch
import numpy as np

# Grade Indices for Cl(4,1) - maps grade to list of basis blade indices
# Grade 0: scalar (1 element)
# Grade 1: vectors (5 elements: e1, e2, e3, e+, e-)
# Grade 2: bivectors (10 elements)
# Grade 3: trivectors (10 elements)
# Grade 4: quadvectors (5 elements)
# Grade 5: pseudoscalar (1 element)
GRADE_INDICES = {
    0: [0],  # Scalar
    1: [1, 2, 4, 8, 16],  # e1, e2, e3, e+, e-
    2: [3, 5, 6, 9, 10, 12, 17, 18, 20, 24],  # Bivectors
    3: [7, 11, 13, 14, 19, 21, 22, 25, 26, 28],  # Trivectors
    4: [15, 23, 27, 29, 30],  # Quadvectors
    5: [31]  # Pseudoscalar
}

def basis_product_cl41(a, b):
    """
    Computes the geometric product of two basis blades in Cl(4,1).
    Basis blades are represented by bitmasks (0-31).
    Returns (sign, resulting_blade_mask).
    
    Signature: (++,+,-) for e1, e2, e3, e+, e-
    Indices: e1=1, e2=2, e3=4, e+=8, e-=16 (2^0, 2^1, 2^2, 2^3, 2^4)
    """
    sign = 1.0
    a_bits = a
    for i in range(5):
        if (b >> i) & 1:
            # For each set bit in b, we move it past bits in a
            for j in range(i + 1, 5):
                if (a_bits >> j) & 1:
                    sign *= -1.0
            if (a_bits >> i) & 1:
                # Handle the square of the basis vector
                if i == 4: # e- squared is -1
                    sign *= -1.0
                a_bits &= ~(1 << i)
            else:
                a_bits |= (1 << i)
    return sign, a_bits

# Shared Cayley table for vectorized operations
_GP_MAP = None

def get_gp_map(device, dtype=torch.float32):
    """
    Returns the precomputed Cayley table for Geometric Product in Cl(4,1).
    Shape: (32, 32, 32)
    """
    global _GP_MAP
    if _GP_MAP is None:
        table = torch.zeros((32, 32, 32), device=device, dtype=dtype)
        for a in range(32):
            for b in range(32):
                sign, res = basis_product_cl41(a, b)
                table[a, b, res] = sign
        _GP_MAP = table
    return _GP_MAP.to(device)

def gp_cl41(A, B):
    """
    Vectorized Geometric Product A * B.
    Handles broadcasting for (..., 32) tensors.
    """
    device = A.device
    gp_map = get_gp_map(device, A.dtype)
    return torch.einsum('...i, ...j, ijk -> ...k', A, B, gp_map)


def wedge_cl41(A, B):
    """
    Vectorized Outer Product A ^ B.
    Filter: Only keeps components where grades add up (grade(A*B) == grade(A) + grade(B)).
    """
    device = A.device
    gp_map = get_gp_map(device, A.dtype).clone()
    for a in range(32):
        for b in range(32):
            ga = bin(a).count('1')
            v_bar_versor = bin(b).count('1')
            _, res = basis_product_cl41(a, b)
            gres = bin(res).count('1')
            if gres != (ga + v_bar_versor):
                gp_map[a, b, res] = 0.0
    return torch.einsum('...i, ...j, ijk -> ...k', A, B, gp_map)


def reverse_cl41(A):
    """
    Clifford Reverse (tilde operator): ~A.
    Flips grades k where k*(k-1)/2 is odd (grades 2, 3).
    """
    device = A.device
    mask = torch.ones(32, device=device)
    for i in range(32):
        grade = bin(i).count('1')
        if (grade * (grade - 1) // 2) % 2 == 1:
            mask[i] = -1.0
    return A * mask

# Precomputed Metric Signature for inner product efficiency
_SIGNATURE = None

def get_signature(device):
    global _SIGNATURE
    if _SIGNATURE is None:
        # Indices: e1=1, e2=2, e3=4, e+=8, e-=16
        # Signature: (1, 1, 1, 1, -1)
        sig = torch.ones(32, device=device)
        for i in range(32):
            # Check e- (bit 4)
            if (i >> 4) & 1:
                sig[i] *= -1.0
            # Also need to account for Clifford reverse logic in inner product
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1:
                sig[i] *= -1.0
        _SIGNATURE = sig
    return _SIGNATURE.to(device)

def inner_cl41(A, B):
    """
    Optimized Scalar product <A * ~B>_0.
    Uses the diagonal nature of the Cl(4,1) signature to avoid 3D contractions.
    """
    sig = get_signature(A.device)
    return torch.sum(A * B * sig, dim=-1)


def normalize_cl41(A, eps=1e-8):
    """
    Project multivectors onto the unit manifold.
    Calculation: A / sqrt(|A * ~A|_0)
    
    Stabilization: Adds a Frobenius Norm (total lane energy) guard to 
    prevent 'Null-Vector Explosion' where lanes grow massive while 
    the scalar product stays small.
    """
    # 1. Standard Geometric Norm (for manifold projection)
    norm_sq = inner_cl41(A, A)
    g_norm = torch.sqrt(torch.abs(norm_sq) + eps)
    
    # 2. Frobenius Norm (for numerical lane stability)
    f_norm = torch.norm(A, p=2, dim=-1, keepdim=True) + eps
    
    # Use the max of the two to ensure stability
    denom = torch.max(g_norm.unsqueeze(-1), f_norm / 4.0).clamp(min=1.0)
    
    return A / denom


def conformal_lift(spins):
    """
    Lifts Ising spins (+1, -1) to the Conformal Null Basis in Cl(4,1).
    +1 -> n_o (origin) = 0.5 * (e- - e+)
    -1 -> n_inf (infinity) = e- + e+
    
    Basis Indices: e+=8, e-=16
    """
    n_o = torch.zeros(32)
    n_o[16] = 0.5  # e-
    n_o[8] = -0.5  # e+
    
    n_inf = torch.zeros(32)
    n_inf[16] = 1.0
    n_inf[8] = 1.0
    
    batch_size, seq_len = spins.shape
    out = torch.zeros((batch_size, seq_len, 32), device=spins.device, dtype=spins.dtype)
    
    mask1 = (spins == 1).unsqueeze(-1)
    mask2 = (spins == -1).unsqueeze(-1)
    
    out += mask1 * n_o.to(spins.device)
    out += mask2 * n_inf.to(spins.device)
    return out
