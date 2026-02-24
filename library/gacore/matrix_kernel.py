import torch
import torch.nn as nn
import numpy as np
import os

# Load Precomputed Mapping (Shared across processes)
# We embed the mapping generation logic here for self-containment
_MAPPING_CACHE = {}

def get_cl41_matrix_mapping(device, dtype=torch.float32):
    key = (device.type, dtype)
    if key in _MAPPING_CACHE:
        return _MAPPING_CACHE[key]
        
    print(f"Initializing Matrix mapping for {device}...")
    
    # Pauli Matrices
    s0 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    e = [None] * 5
    e[0] = np.kron(sx, s0)
    e[1] = np.kron(sy, s0)
    e[2] = np.kron(sz, sx)
    e[3] = np.kron(sz, sy)
    e[4] = np.kron(sz, sz) * 1j
    
    m_basis = np.zeros((32, 4, 4), dtype=complex)
    m_basis[0] = np.eye(4, dtype=complex)
    for i in range(1, 32):
        mat = np.eye(4, dtype=complex)
        for b in range(5):
            if (i >> b) & 1:
                mat = mat @ e[b]
        m_basis[i] = mat
        
    # Convert to real format (32, 4, 4, 2)
    mapping_real = np.stack([m_basis.real, m_basis.imag], axis=-1).astype(np.float32)
    mapping_torch = torch.from_numpy(mapping_real).to(device=device, dtype=dtype)
    
    _MAPPING_CACHE[key] = mapping_torch
    return mapping_torch

def ga_to_matrix(x, mapping):
    # x: (..., 32)
    # mapping: (32, 4, 4, 2)
    # res: (..., 4, 4, 2)
    return torch.einsum('...i, ijkr -> ...jkr', x, mapping)

def matrix_to_ga(m, mapping):
    # m: (..., 4, 4, 2)
    # mapping: (32, 4, 4, 2)
    # Basis is orthogonal under trace inner product: Re(Tr(A B*))
    # For our basis: a_i = 1/4 * Re(Tr(M * M_i_inv))
    # Since M_i are unitary and M_i^2 = +/- 1, M_i_inv = +/- M_i
    # e1-e4 square to 1, e5 squares to -1.
    
    # Precompute inverse/signatures for back-projection
    # Actually, we can just use einsum and find the projection
    # Because it's an isomorphism, M = sum a_i M_i is unique.
    # We can solve it as a least-squares or just dot product.
    # Result: (..., 32)
    return torch.einsum('...jkr, ijkr -> ...i', m, mapping) / 4.0

def complex_matmul_broadcast(A_real, B_real):
    # A, B are (..., 4, 4, 2)
    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    ac = torch.matmul(A_real[..., 0], B_real[..., 0])
    bd = torch.matmul(A_real[..., 1], B_real[..., 1])
    ad = torch.matmul(A_real[..., 0], B_real[..., 1])
    bc = torch.matmul(A_real[..., 1], B_real[..., 0])
    
    return torch.stack([ac - bd, ad + bc], dim=-1)

def geometric_product_matrix(a, b):
    """Vectorized Geometric Product A * B using Matrix Representation."""
    device = a.device
    mapping = get_cl41_matrix_mapping(device, a.dtype)
    
    # 1. Map to Matrix
    ma = ga_to_matrix(a, mapping)
    mb = ga_to_matrix(b, mapping)
    
    # 2. Complex MatMul
    mres = complex_matmul_broadcast(ma, mb)
    
    # 3. Map back
    return matrix_to_ga(mres, mapping)

def complex_matmul_fast(A_real, B_real):
    # A, B are (M, K, 2) complex
    # Standard complex multiplication: (ac-bd) + (ad+bc)i
    # Optimization: Use grouped GEMM or just 4 matmuls
    ac = torch.matmul(A_real[..., 0], B_real[..., 0])
    bd = torch.matmul(A_real[..., 1], B_real[..., 1])
    ad = torch.matmul(A_real[..., 0], B_real[..., 1])
    bc = torch.matmul(A_real[..., 1], B_real[..., 0])
    
    return torch.stack([ac - bd, ad + bc], dim=-1)

def matrix_geometric_product(ma, mb):
    """Geometric Product in Matrix Space: A * B."""
    return complex_matmul_broadcast(ma, mb)

def matrix_manifold_normalization(m, eps=1e-6):
    """
    Project multivectors onto the unit manifold while in matrix space.
    Equivalent to normalize_cl41 in GA space.
    """
    # 1. Standard Geometric Norm: ||A||² = <A * ~A>_0
    # In M4(C), <A * B>_0 = Re(Tr(A @ B)) / 4
    # Reverse ~A in matrix space is M^\dagger (adjoint) for some bases, 
    # but for Cl(4,1) with our mapping, it's specific.
    # However, for Sp(4,1) rotors, we mainly need to preserve det(M) = 1 or similar.
    # A more robust way in matrix space:
    # normalize such that Re(Tr(M @ M_rev)) = 4
    
    # Simpler: Projection back to Sp(4,1) can be approximate:
    # m / sqrt(|det(m)|) if it were a simple rotation.
    # To match normalize_cl41 exactly, we use the trace of product with identity representation.
    # But wait, we can just use the Frobenius norm of the matrix as a proxy for stability.
    
    # For RRA, we'll use a trace-based norm for exactness:
    # <A*~A>_0 is the first component of the GA vector.
    # We can extract it via m @ mapping.transpose(...)
    # But simpler: mapping[0] is Identity. 
    # So <A*~B>_0 = Re(Tr(MA @ MB_rev)) / 4
    
    # For now, let's use the Frobenius norm / 4 as a stable proxy, 
    # or just use ga_to_matrix(normalize_cl41(matrix_to_ga(m))) if we want identity.
    # To be "Fused", we should avoid going back.
    
    # Trace-based scalar part: <M>_0 = Re(Tr(M)) / 4
    # Since Mapping[0] is Identity.
    trace_real = m[..., 0, 0, 0] + m[..., 1, 1, 0] + m[..., 2, 2, 0] + m[..., 3, 3, 0]
    scalar_part = trace_real / 4.0
    
    # This is <M>_0. We need <M * ~M>_0.
    # Let's just use Frobenius norm for RRA stability, matching the 'f_norm' in normalize_cl41.
    f_norm = torch.sqrt(torch.sum(m ** 2, dim=(-1, -2, -3)) / 4.0 + eps)
    
    return m / f_norm[..., None, None, None]

def geometric_linear_layer_matrix(x, weight):
    """
    Hyper-Optimized Geometric Linear Layer via Matrix Representation.
    - Path: Cl(4,1) -> M4(C) -> Flattened GEMM -> Cl(4,1)
    - Flop Reduction: 4x (256 vs 1024 basis products)
    - Hardware Utilization: Converts sparse-ish GA to massive dense GEMM.
    """
    device = x.device
    mapping = get_cl41_matrix_mapping(device, x.dtype)
    
    M_orig_shape = x.shape[:-2]
    K = x.shape[-2]
    N = weight.shape[0]
    
    # 0. Flatten batch dims
    x_flat = x.view(-1, K, 32)
    M = x_flat.shape[0]
    
    # 1. Map to Matrix
    x_mat = ga_to_matrix(x_flat, mapping) # (M, K, 4, 4, 2)
    w_mat = ga_to_matrix(weight, mapping) # (N, K, 4, 4, 2)
    
    # 2. Reshape to Massive GEMM
    # X: (M, K, 4, 4) -> (M, 4, K, 4) -> (M*4, K*4)
    x_gemm = x_mat.permute(0, 2, 1, 3, 4).reshape(M*4, K*4, 2)
    # W: (N, K, 4, 4) -> (K, 4, N, 4) -> (K*4, N*4)
    w_gemm = w_mat.permute(1, 2, 0, 3, 4).reshape(K*4, N*4, 2)
    
    # 3. Perform Single Large Complex GEMM
    # out = X @ W
    y_gemm = complex_matmul_fast(x_gemm, w_gemm) # (M*4, N*4, 2)
    
    # 4. Map Back
    y_mat = y_gemm.reshape(M, 4, N, 4, 2).permute(0, 2, 1, 3, 4) # (M, N, 4, 4, 2)
    y_ga = matrix_to_ga(y_mat, mapping) # (M, N, 32)
    
    return y_ga.view(*M_orig_shape, N, 32)
