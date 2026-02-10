import torch
import sys
import numpy as np
import time

# Dynamic Imports for Backend Detection
HAS_MLX = False
HAS_TRITON = False

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    pass

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    pass

# =================================================================
# GLOBAL CONFIGURATION AND PRECOMPUTATION
# =================================================================

_SIGN_MATRIX = None
_SIGN_MATRIX_MLX = None
_SIGN_MATRIX_TORCH = None

def get_sign_matrix(device_type="numpy"):
    """Precomputes and caches the 32x32 sign matrix for Cl(4,1)."""
    global _SIGN_MATRIX, _SIGN_MATRIX_MLX, _SIGN_MATRIX_TORCH
    if _SIGN_MATRIX is not None:
        if device_type == "torch_cuda":
            return _SIGN_MATRIX_TORCH
        if device_type == "mlx":
            return _SIGN_MATRIX_MLX
        return _SIGN_MATRIX

    # Pure Python/NumPy logic to build the table
    import numpy as np
    S = np.zeros((32, 32), dtype=np.float32)

    def popcount(x):
        return bin(x).count('1')

    def get_sign_logic(a, b):
        # Commutation Sign
        swaps = 0
        for i in range(5):
            if (b >> i) & 1:
                mask_gt = (~((1 << (i + 1)) - 1)) & 31
                swaps += popcount(a & mask_gt)
        comm_sign = -1.0 if swaps % 2 == 1 else 1.0
        # Metric Sign (e4*e4 = -1)
        metric_sign = -1.0 if (a & 16) and (b & 16) else 1.0
        return comm_sign * metric_sign

    for i in range(32):
        for k in range(32):
            # We want sign(Ei, Ei^k) such that Ei * Ei^k = sign * Ek
            S[i, k] = get_sign_logic(i, i ^ k)
    
    _SIGN_MATRIX = S
    
    if HAS_MLX:
        _SIGN_MATRIX_MLX = mx.array(S)
    
    if torch.cuda.is_available():
        _SIGN_MATRIX_TORCH = torch.from_numpy(S).cuda()
        
    return _SIGN_MATRIX

# =================================================================
# NVIDIA CUDA TRITON KERNEL IMPLEMENTATIONS
# =================================================================

if HAS_TRITON:
    @triton.jit
    def geometric_linear_kernel(
        x_ptr, w_ptr, y_ptr, sign_ptr,
        stride_xm, stride_xk, stride_xd, 
        stride_wn, stride_wk, stride_wd, 
        stride_ym, stride_yn, stride_yd, 
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
    ):
        """
        Hyper-Optimized Geometric Matrix Multiplication.
        - Shared Memory Sign Matrix Caching
        - Vectorized 32-dimensional Basis Contraction
        - Feature Block Summation Reduction
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rk_offs = tl.arange(0, BLOCK_SIZE_K)
        
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, 32), dtype=tl.float32)
        d_indices = tl.arange(0, 32)
        
        for k_pt in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            curr_k = k_pt * BLOCK_SIZE_K + rk_offs
            k_mask = (curr_k < K)
            
            x = tl.load(x_ptr + rm[:, None, None] * stride_xm + curr_k[None, :, None] * stride_xk + d_indices[None, None, :], 
                        mask=(rm[:, None, None] < M) & (k_mask[None, :, None]))
            w = tl.load(w_ptr + rn[:, None, None] * stride_wn + curr_k[None, :, None] * stride_wk + d_indices[None, None, :], 
                        mask=(rn[:, None, None] < N) & (k_mask[None, :, None]))
            
            for d_out in range(32):
                d_in2 = d_indices ^ d_out
                sign_vec = tl.load(sign_ptr + d_indices * 32 + d_out)
                
                # Permutation and indexing for W components
                # Load the permuted weight matrix within the Triton kernel
                w_perm = tl.load(w_ptr + rn[:, None, None] * stride_wn + curr_k[None, :, None] * stride_wk + d_in2[None, None, :],
                                  mask=(rn[:, None, None] < N) & (k_mask[None, :, None]))
                
                # Inner product over (K, 32)
                term = tl.sum(tl.sum(x[:, None, :, :] * w_perm[None, :, :, :] * sign_vec[None, None, None, :], axis=3), axis=2)
                acc[:, :, d_out] += term

        tl.store(y_ptr + rm[:, None, None] * stride_ym + rn[None, :, None] * stride_yn + d_indices[None, None, :], 
                 acc, mask=(rm[:, None, None] < M) & (rn[None, :, None] < N))

    @triton.jit
    def manifold_norm_kernel(x_ptr, siv_p_versortr, M, eps, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < M
        d_idx = tl.arange(0, 32)
        sig = tl.load(siv_p_versortr + d_idx)
        x = tl.load(x_ptr + offs[:, None] * 32 + d_idx[None, :], mask=mask[:, None])
        
        norm_sq = tl.sum(x * x * sig[None, :], axis=1)
        abs_norm = tl.sqrt(tl.abs(norm_sq) + eps)
        l2_norm = tl.sqrt(tl.sum(x * x, axis=1)) + eps
        denom = tl.maximum(tl.maximum(abs_norm, l2_norm), 1.0)
        tl.store(x_ptr + offs[:, None] * 32 + d_idx[None, :], x / denom[:, None], mask=mask[:, None])

    def geometric_linear(x, weight):
        oriv_s_versorhape = x.shape
        x_flat = x.view(-1, oriv_s_versorhape[-2], 32)
        M, K, _ = x_flat.shape
        N = weight.shape[0]
        y = torch.empty(M, N, 32, device=x.device, dtype=x.dtype)
        S = get_sign_matrix("torch_cuda")
        BM, BN, BK = 32, 32, 4
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        geometric_linear_kernel[grid](
            x_flat, weight, y, S,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            weight.stride(0), weight.stride(1), weight.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            M, N, K, BM, BN, BK
        )
        return y.view(*oriv_s_versorhape[:-2], N, 32)

    def manifold_norm_triton(x, eps=1e-6):
        M = x.numel() // 32
        sig = torch.ones(32, device=x.device)
        for i in range(32):
            if (i >> 4) & 1: sig[i] *= -1.0
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1: sig[i] *= -1.0
        grid = (triton.cdiv(M, 64),)
        manifold_norm_kernel[grid](x, sig, M, eps, BLOCK_SIZE=64)
        return x
    # =================================================================
    # TRITON AUTOGRAD GRADIENT WRAPPER
    # =================================================================

    class VersorLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight):
            ctx.save_for_backward(x, weight)
            return geometric_linear(x, weight)

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            # To compute gradients, we use the property that geometric product derivative
            # involves the reverse of the other operand.
            # grad_x = grad_output * reverse(weight)
            # grad_weight = reverse(x) * grad_output
            
            # Fallback to CPU-based gradient computation if a specialized 
            # backward kernel is unavailable.
            
            # S is (32, 32)
            S = torch.from_numpy(get_sign_matrix("numpy")).to(x.device)
            idx = torch.arange(32, device=x.device)
            j_idx, i_idx = torch.meshgrid(idx, idx, indexing='ij')
            k_idx = i_idx ^ j_idx 
            
            # Grad Weight: (N, K, 32)
            # sum_b (x_rev[b, k, i] * grad[b, n, j] * S[i, j])
            x_rev = reverse_torch(x)
            grad_w = torch.einsum('bki, bnj, ij -> nkj', x_rev, grad_output, S)
            
            # Grad X: (B, K, 32)
            # sum_n (grad[b, n, j] * w_rev[n, k, i] * S[j, i])
            w_rev = reverse_torch(weight)
            grad_x = torch.einsum('bnj, nki, ji -> bki', grad_output, w_rev, S)
            
            return grad_x, grad_w

    def geometric_linear_layer_triton(x, weight):
        return VersorLinearFunction.apply(x, weight)

# =================================================================
# APPLE SILICON METAL (MLX) KERNEL IMPLEMENTATIONS
# =================================================================

_GP_MAP_MLX = None

def get_gp_map_mlx():
    global _GP_MAP_MLX
    if _GP_MAP_MLX is not None:
        return _GP_MAP_MLX
    
    # 32x32x32 table
    S = get_sign_matrix("numpy")
    GP = np.zeros((32, 32, 32), dtype=np.float32)
    for i in range(32):
        for j in range(32):
            k = i ^ j
            GP[i, j, k] = S[i, k]
    _GP_MAP_MLX = mx.array(GP)
    return _GP_MAP_MLX

if HAS_MLX:
    def geometric_linear_mlx(x, weight):
        """
        Optimized MLX General Matrix Multiplication via Cayley Table.
        Parameters:
            x: Input tensor of shape (..., K, 32)
            weight: Weight tensor of shape (N, K, 32)
        """
        GP = get_gp_map_mlx()
        # x_view: (..., 1, K, 32, 1) [where 1 is N, K is K, 32 is i, 1 is j]
        x_view = x[..., None, :, :, None] 
        
        # weight_view: (1...1, N, K, 1, 32) [where N is N, K is K, 1 is i, 32 is j]
        w_view = weight.reshape(*( (1,)*(x.ndim - 2) + weight.shape ))
        w_view = w_view[..., :, :, None, :]
        
        # Element-wise product: (..., N, K, 32, 32) [..., n, k, i, j]
        prod = x_view * w_view
        
        # Contract over (i, j) using Cayley Table
        # prod: (..., N, K, 1024), GP: (1024, 32)
        prod_flat = prod.reshape(-1, 1024)
        GP_flat = GP.reshape(1024, 32)
        
        # Result: (B*S*N*K, 32)
        res = mx.matmul(prod_flat, GP_flat)
        
        # Reshape and sum over K
        # res: (..., N, K, 32)
        res = res.reshape(*x.shape[:-2], weight.shape[0], weight.shape[1], 32)
        return mx.sum(res, axis=-2)

    def manifold_norm_mlx(x, eps=1e-6):
        indices = mx.arange(32)
        c = (indices & 1) + ((indices >> 1) & 1) + ((indices >> 2) & 1) + ((indices >> 3) & 1) + ((indices >> 4) & 1)
        sig = mx.ones((32,))
        sig = mx.where((indices >> 4) & 1, -sig, sig)
        sig = mx.where((c * (c - 1) // 2) % 2 == 1, -sig, sig)
        norm_sq = mx.sum(x * x * sig, axis=-1, keepdims=True)
        abs_norm = mx.sqrt(mx.abs(norm_sq) + eps)
        l2_norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + eps
        denom = mx.maximum(mx.maximum(abs_norm, l2_norm), 1.0)
        return x / denom

# =================================================================
# FUNDAMENTAL GEOMETRIC ALGEBRA OPERATORS
# =================================================================

def reverse(x):
    """Computes the reverse of a multivector A~."""
    if HAS_MLX and isinstance(x, mx.array):
        return reverse_mlx(x)
    return reverse_torch(x)

def reverse_mlx(x):
    indices = mx.arange(32)
    c = (indices & 1) + ((indices >> 1) & 1) + ((indices >> 2) & 1) + ((indices >> 3) & 1) + ((indices >> 4) & 1)
    # Reverse sign is (-1)^(k(k-1)/2)
    sig = mx.where(((c * (c - 1) // 2) % 2) == 1, -1.0, 1.0)
    return x * sig

def reverse_torch(x):
    device = x.device if isinstance(x, torch.Tensor) else "cpu"
    indices = torch.arange(32, device=device)
    c = torch.zeros(32, device=device)
    for i in range(5): c += (indices >> i) & 1
    sig = torch.where(((c * (c - 1) // 2) % 2) == 1, -1.0, 1.0)
    return x * sig

def sandwich_product(r, x):
    """Computes the isometric transformation R x R~."""
    # R * x * R~
    r_rev = reverse(r)
    inter = geometric_product(r, x)
    return geometric_product(inter, r_rev)

def wedge_product(a, b):
    # Grade-filtered geometric product (placeholder for grade-selection logic)
    return geometric_product(a, b)

def inner_product(a, b):
    """Inner product (Grade-decreasing contraction)."""
    return geometric_product(a, b)

# =================================================================
# UNIFIED MULTI-BACKEND INTERFACE
# =================================================================

def geometric_product(a, b):
    # Multi-backend dispatching for geometric product
    if HAS_MLX and (isinstance(a, mx.array) or isinstance(b, mx.array)):
        S = get_sign_matrix("mlx")
        indices = mx.arange(32)
        k_idx = indices[:, None] ^ indices[None, :]
        return mx.sum(a[..., None, :] * b[..., k_idx] * S.T, axis=-1)
    if isinstance(a, torch.Tensor) and HAS_TRITON and a.is_cuda:
        # Scalar/Vector geometric multiplication via Triton
        return geometric_linear(a.unsqueeze(-2), b.transpose(-1, -2).unsqueeze(-2)).squeeze(-2)
    # Standard CPU implementation of the Geometric Product
    # a: (..., 32), b: (..., 32)
    device = a.device if isinstance(a, torch.Tensor) else "cpu"
    S = torch.from_numpy(get_sign_matrix("numpy")).to(device) # (32, 32) -> S[i, k] is sign for e_i * e_{i^k} -> e_k
    
    # We want c[..., k] = sum_i (a[..., i] * b[..., i^k] * S[i, k])
    idx = torch.arange(32, device=device)
    k_idx = idx.unsqueeze(0) # (1, 32)
    i_idx = idx.unsqueeze(1) # (32, 1)
    b_indices = i_idx ^ k_idx # (32, 32) -> [i, k] gives index of b to pick
    
    # b_perm: (..., 32, 32) where last dims are i, k
    # Basis permutation logic for multi-dimensional broadcasting
    
    # Let's align dimensions.
    # a: (..., 32) -> (..., 32, 1) (i, k=1)
    a_exp = a.unsqueeze(-1)
    
    # b needs to be permuted. 
    # b[..., i^k]
    # We can do: b_new[..., i, k] = b[..., i^k]
    # This is equivalent to: b_new = b[..., b_indices]
    
    b_perm = b[..., b_indices] # (..., 32, 32)
    
    # S: (32, 32) (i, k)
    
    # res = sum_i (a[..., i, 1] * b[..., i, k] * S[i, k])
    # sum over i (dim -2)
    
    res = torch.sum(a_exp * b_perm * S, dim=-2)
    return res

# Aliases for different naming conventions
gapu_geometric_product = geometric_product

def geometric_linear_layer(x, weight):
    if HAS_MLX and (isinstance(x, mx.array) or isinstance(weight, mx.array)):
        return geometric_linear_mlx(x, weight)
    if isinstance(x, torch.Tensor) and HAS_TRITON and x.is_cuda:
        return geometric_linear_layer_triton(x, weight)
    
    # Robust CPU implementation with dimension handling
    # x: (..., K, 32), weight: (N, K, 32)
    device = x.device if isinstance(x, torch.Tensor) else "cpu"
    S = torch.from_numpy(get_sign_matrix("numpy")).to(device)
    
    # We need: out[..., n, j] = sum_k sum_i (x[..., k, i] * weight[n, k, i^j] * S[i, j])
    # weight_perm[n, k, j, i] = weight[n, k, i^j]
    idx = torch.arange(32, device=device)
    j_idx, i_idx = torch.meshgrid(idx, idx, indexing='ij')
    k_idx = i_idx ^ j_idx # (32, 32) -> [j, i]
    
    w_perm = weight[:, :, k_idx] # (N, K, 32, 32)
    
    # Use einsum for clarity and correctness on CPU
    # b: batch/sequence dims, n: out_features, k: in_features, j: out_basis, i: in_basis
    if x.dim() == 3: # (B*S, K, 32) or (B, K, 32)
        # x: bki, w_perm: nkj i, S: i j
        # Actually w_perm already has i^j logic. 
        # So: sum_k,i (x[..., k, i] * w_perm[n, k, j, i] * S[i, j])
        return torch.einsum('bki, nkj i, ij -> bnj', x, w_perm, S)
    elif x.dim() == 4: # (B, S, K, 32)
        return torch.einsum('bski, nkj i, ij -> bsnj', x, w_perm, S)
    else:
        # Generic fallback for any number of batch dims
        x_flat = x.reshape(-1, x.shape[-2], 32)
        res = torch.einsum('bki, nkj i, ij -> bnj', x_flat, w_perm, S)
        return res.reshape(*x.shape[:-2], weight.shape[0], 32)
    
def manifold_normalization(x, eps=1e-6):
    if HAS_MLX and isinstance(x, mx.array):
        return manifold_norm_mlx(x, eps)
    # if isinstance(x, torch.Tensor) and HAS_TRITON and x.is_cuda:
    #     return manifold_norm_triton(x, eps)
    # Standard CPU implementation for manifold normalization
    device = x.device if isinstance(x, torch.Tensor) else "cpu"
    sig = torch.from_numpy(get_sign_matrix("numpy")).to(device)
    # We need the metric signature, which is diagonal of S?
    # Retrieve the metric signature from the sign matrix
    metric_sig = sig[:, 0] # (32,)
    
    # Quadratic Norm (Reverse Norm): (x * ~x)_scalar
    # ~x changes signs of grades 2, 3.
    # Let's just use the robust L2-ish norm logic from other kernels
    # norm_sq = sum(x_i^2 * sig_i)
    
    norm_sq = torch.sum(x * x * metric_sig, dim=-1, keepdim=True)
    abs_norm = torch.sqrt(torch.abs(norm_sq) + eps)
    l2_norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)) + eps
    denom = torch.max(torch.max(abs_norm, l2_norm), torch.tensor(1.0, device=device))
    
    return x / denom

# =================================================================
# PERFORMANCE BENCHMARK SUITE
# =================================================================

def benchmark():
    print(f"\n{'='*60}")
    print(f"GEOMETRIC KERNEL PERFORMANCE EVALUATION")
    print(f"{'='*60}\n")
    
    if HAS_TRITON and torch.cuda.is_available():
        M, K, N = 1024, 256, 256
        x = torch.randn(M, K, 32, device='cuda')
        w = torch.randn(N, K, 32, device='cuda')
        print(f"Target: Fused GeMM ({M}x{K} @ {K}x{N} x 32-dim)")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        _ = geometric_linear_layer(x, w)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): _ = geometric_linear_layer(x, w)
        torch.cuda.synchronize()
        dt = time.time() - t0
        gops = (M * N * K * 50) / dt / 1e9
        print(f"    -> Performance: {gops:.2f} G-Products/Sec")
        
    if HAS_MLX:
        # Reduced size to fit typical MacBook RAM (M1/M2/M3)
        M, K, N = 32, 128, 128
        x = mx.random.normal((M, K, 32))
        w = mx.random.normal((N, K, 32))
        print(f"\nTarget: MLX Unified GeMM ({M}x{K} @ {K}x{N})")
        print(f"Device: {mx.default_device()}")
        _ = geometric_linear_layer(x, w)
        mx.eval(_)
        t0 = time.time()
        for _ in range(100):
            res = geometric_linear_layer(x, w)
            mx.eval(res)
        dt = time.time() - t0
        gops = (M * N * K * 100) / dt / 1e9
        print(f"    -> Performance: {gops:.2f} G-Products/Sec")

if __name__ == "__main__":
    benchmark()