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
        
    # Compute Basis Square Signs (Metric)
    # i-th basis blade squares to +/- 1.
    # In GA: e_i^2 = sign. 
    # For matrix: M_i^2 = sign * I.
    sig_np = np.zeros(32, dtype=np.float32)
    for i in range(32):
        # We can just check the matrix property M_i @ M_i
        res = m_basis[i] @ m_basis[i]
        # Check if it is +I or -I
        if np.allclose(res, np.eye(4)):
            sig_np[i] = 1.0
        elif np.allclose(res, -np.eye(4)):
            sig_np[i] = -1.0
        else:
            # Should not happen for Cl(4,1) unit bases
            sig_np[i] = 0.0

    # Convert to real format (32, 4, 4, 2)
    mapping_real = np.stack([m_basis.real, m_basis.imag], axis=-1).astype(np.float32)
    mapping_torch = torch.from_numpy(mapping_real).to(device=device, dtype=dtype)
    sig_torch = torch.from_numpy(sig_np).to(device=device, dtype=dtype)
    
    _MAPPING_CACHE[key] = (mapping_torch, sig_torch)
    return mapping_torch, sig_torch

def ga_to_matrix(x, mapping):
    # x: (..., 32)
    # mapping: (32, 4, 4, 2)
    # res: (..., 4, 4, 2)
    return torch.einsum('...i, ijkr -> ...jkr', x.to(mapping.dtype), mapping)

def matrix_to_ga(m, mapping, sig):
    # m: (..., 4, 4, 2)
    # mapping: (32, 4, 4, 2)
    # sig: (32,) [Used in normalization, not in basic projection]
    # Basis is orthogonal under complex inner product: Re(Tr(M * M_i_dagger)) / 4
    # Our einsum calculates: sum(m_real * mapping_real + m_imag * mapping_imag)
    # which is exactly Re(Tr(M * M_i_dagger)).
    # Thus a_i = Re(Tr(M * M_i_dagger)) / 4.
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
    common_dtype = torch.promote_types(a.dtype, b.dtype)
    mapping, sig = get_cl41_matrix_mapping(device, common_dtype)
    
    # 1. Map to Matrix
    ma = ga_to_matrix(a.to(common_dtype), mapping)
    mb = ga_to_matrix(b.to(common_dtype), mapping)
    
    # 2. Complex MatMul
    mres = complex_matmul_broadcast(ma, mb)
    
    # 3. Map back
    return matrix_to_ga(mres, mapping, sig)

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
    Strictly enforces <A * ~A>_0 = 1.
    """
    # 1. Extract the scalar part of the product with the reverse.
    # For unitary representations of Spin(4,1), the adjoint matrix is usually 
    # related to the reverse. However, we can just project to GA space and back, 
    # but that's slow.
    # Optimized: <A * ~A>_0 = sum(a_i^2 * sign(e_i * ~e_i))
    # We use the GA coefficients directly for the norm calculation.
    device = m.device
    mapping, sig = get_cl41_matrix_mapping(device, m.dtype)
    
    # Map back to GA to get coefficients
    a = matrix_to_ga(m, mapping, sig)
    
    # Clifford Norm Sq: sum(a_i^2 * metric_sign[i])
    # For Cl(4,1), metric_sign is exactly the sign of e_i * ~e_i.
    # Note: ~e_i = reverse(e_i). sign(e_i * ~e_i) is S[i, 0] in kernel.py.
    # Let's derive it from our sig (basis square signs).
    # reverse(e_i) = (-1)^(g(g-1)/2) e_i. 
    # So e_i * ~e_i = (-1)^(g(g-1)/2) e_i^2.
    
    basis_indices = np.arange(32)
    grades = np.array([bin(i).count('1') for i in basis_indices])
    rev_signs = ((-1)**(grades * (grades - 1) // 2)).astype(np.float32)
    rev_signs_torch = torch.from_numpy(rev_signs).to(device)
    
    # Full metric sig including revision
    cl_metric_sig = sig * rev_signs_torch
    
    norm_sq_abs = torch.abs(torch.sum(a * a * cl_metric_sig, dim=-1, keepdim=True))
    denom = torch.sqrt(norm_sq_abs + eps)
    
    return m / denom[..., None, None]

def geometric_linear_layer_matrix(x, weight):
    """
    Hyper-Optimized Geometric Linear Layer via Matrix Representation.
    - Path: Cl(4,1) -> M4(C) -> Flattened GEMM -> Cl(4,1)
    - Flop Reduction: 4x (256 vs 1024 basis products)
    - Hardware Utilization: Converts sparse-ish GA to massive dense GEMM.
    """
    device = x.device
    mapping, sig = get_cl41_matrix_mapping(device, x.dtype)
    
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
    y_ga = matrix_to_ga(y_mat, mapping, sig) # (M, N, 32)
    
    return y_ga.view(*M_orig_shape, N, 32)
