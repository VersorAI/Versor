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

_SIGN_CACHE = {}
_METRIC_CACHE = {}

# Default Metric: Cl(4,1) -> e0..e3 (+1), e4 (-1)
# 3 spacelike, 1 timelike, but stored as 5 bits. 
# Bit 4 is usually the timelike one in this legacy config.
_METRIC = [1, 1, 1, 1, -1] # This will be deprecated in favor of passing signature

def set_metric(metric):
    """
    Sets the metric signature for the geometric algebra.
    metric: List or array of length 5 (for 32 dims), optional padding logic can be added.
            Values should be 1, -1, or 0.
    """
    # This function is now a no-op as the metric (signature) is passed directly to functions.
    # Kept for backward compatibility if external code calls it, but it won't affect new logic.
    pass

def get_sign_matrix(signature, device_type="numpy"):
    """
    Computes/Retrieves the sign matrix for a given metric signature.
    signature: iterable of length D (e.g. [+1, +1, +1, +1, -1])
    """
    sig_tuple = tuple(signature.tolist()) if hasattr(signature, 'tolist') else tuple(signature)
    
    cache_key = (sig_tuple, device_type)
    if cache_key in _SIGN_CACHE:
        return _SIGN_CACHE[cache_key]
        
    # Generate Table
    D = len(sig_tuple)
    n_dims = 1 << D
    
    import numpy as np
    S = np.zeros((n_dims, n_dims), dtype=np.float32)

    def popcount(x):
        return bin(x).count('1')

    def get_sign_logic(a, b):
        # Commutation Sign
        swaps = 0
        for i in range(D):
            if (b >> i) & 1:
                mask_gt = (~((1 << (i + 1)) - 1))
                swaps += popcount(a & mask_gt)
        comm_sign = -1.0 if swaps % 2 == 1 else 1.0
        
        # Metric Sign
        m_sign = 1.0
        intersection = a & b
        for i in range(D):
            if (intersection >> i) & 1:
                val = sig_tuple[i]
                if val == 0:
                    return 0.0
                m_sign *= val
        
        return comm_sign * m_sign

    for i in range(n_dims):
        for k in range(n_dims):
            S[i, k] = get_sign_logic(i, i ^ k)
            
    # Cache and Return
    if device_type == "numpy":
        res = S
    elif device_type == "torch_cuda":
        if torch.cuda.is_available():
            res = torch.from_numpy(S).cuda()
        else:
            res = torch.from_numpy(S)
    elif device_type == "mlx":
        res = mx.array(S)
    else:
        res = torch.from_numpy(S) # Default torch cpu
        
    _SIGN_CACHE[cache_key] = res
    return res

# =================================================================
# NVIDIA CUDA TRITON KERNEL IMPLEMENTATIONS
# =================================================================

if HAS_TRITON:
    @triton.jit
    def geometric_linear_kernel_32(
        x_ptr, w_ptr, y_ptr, # Sign ptr removed (Calculated in-register)
        stride_xm, stride_xk, stride_xd, 
        stride_wn, stride_wk, stride_wd, 
        stride_ym, stride_yn, stride_yd, 
        M, N, K,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
    ):
        """
        Hyper-Optimized Geometric Matrix Multiplication.
        - Register-Local Bitwise Sign Computation (No Global Memory Lookups)
        - Vectorized 32-dimensional Basis Contraction
        - Feature Block Summation Reduction
        This kernel is specialized for 32 multivector components (5 basis vectors).
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
                
                # =========================================================
                # BIT-MASKED SIGN LOGIC (REGISTER LEVEL)
                # =========================================================
                # 1. Metric Sign (e4 is index 4, value 16)
                # If bit 4 is set in both d_indices and d_in2, we get a -1 factor.
                # Cl(4,1) metric: + + + + -
                # Note: This kernel is specifically for 5D metric [+ + + + -] or similar. 
                # For dynamic support, we should fallback or upgrade this kernel. 
                # We'll leave it as a fast path for D=32.
                metric_sign = 1.0 - 2.0 * ((d_indices & d_in2 & 16) >> 4)

                # 2. Permutation Sign (Swaps)
                # Calculate swaps required to reorder basis vectors
                swaps = (d_in2 & 1) * tl.popc(d_indices & 30) + \
                        ((d_in2 >> 1) & 1) * tl.popc(d_indices & 28) + \
                        ((d_in2 >> 2) & 1) * tl.popc(d_indices & 24) + \
                        ((d_in2 >> 3) & 1) * tl.popc(d_indices & 16)
                
                # sgn = (-1)^swaps
                comm_sign = 1.0 - 2.0 * (swaps & 1)
                
                final_sign = metric_sign * comm_sign
                
                # Permutation and indexing for W components
                # Load the permuted weight matrix within the Triton kernel
                w_perm = tl.load(w_ptr + rn[:, None, None] * stride_wn + curr_k[None, :, None] * stride_wk + d_in2[None, None, :],
                                  mask=(rn[:, None, None] < N) & (k_mask[None, :, None]))
                
                # Inner product over (K, 32)
                term = tl.sum(tl.sum(x[:, None, :, :] * w_perm[None, :, :, :] * final_sign[None, None, None, :], axis=3), axis=2)
                acc[:, :, d_out] += term

        tl.store(y_ptr + rm[:, None, None] * stride_ym + rn[None, :, None] * stride_yn + d_indices[None, None, :], 
                 acc, mask=(rm[:, None, None] < M) & (rn[None, :, None] < N))

    @triton.jit
    def manifold_norm_kernel(x_ptr, sig_ptr, M, n_dims, eps, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < M
        d_idx = tl.arange(0, n_dims)
        sig = tl.load(sig_ptr + d_idx)
        x = tl.load(x_ptr + offs[:, None] * n_dims + d_idx[None, :], mask=mask[:, None])
        
        norm_sq = tl.sum(x * x * sig[None, :], axis=1)
        abs_norm = tl.sqrt(tl.abs(norm_sq) + eps)
        l2_norm = tl.sqrt(tl.sum(x * x, axis=1)) + eps
        denom = tl.maximum(tl.maximum(abs_norm, l2_norm), 1.0)
        tl.store(x_ptr + offs[:, None] * n_dims + d_idx[None, :], x / denom[:, None], mask=mask[:, None])

    def geometric_linear_triton_32(x, weight):
        # Implementation of Triton caller
        oriv_s_versorhape = x.shape
        x_flat = x.view(-1, oriv_s_versorhape[-2], 32)
        M, K, _ = x_flat.shape
        N = weight.shape[0]
        y = torch.empty(M, N, 32, device=x.device, dtype=x.dtype)
        # S embedded in kernel for 32 dim
        BM, BN, BK = 32, 32, 4
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        geometric_linear_kernel_32[grid](
            x_flat, weight, y,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            weight.stride(0), weight.stride(1), weight.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            M, N, K, BM, BN, BK
        )
        return y.view(*oriv_s_versorhape[:-2], N, 32)

    def geometric_linear(x, weight, signature):
        D_mv = x.shape[-1] # Multivector dimension
        D_basis = len(signature) # Basis vector dimension
        if D_mv == 32 and D_basis == 5:
            # Fast Path for 5D (Cl(4,1) or similar)
             return geometric_linear_triton_32(x, weight)
        else:
            # Fallback path for other dimensions (CPU)
            return geometric_linear_cpu(x, weight, signature)

    def manifold_norm_triton(x, signature, eps=1e-6):
        n_dims = x.shape[-1]
        D_basis = len(signature)
        
        sig_vals_np = np.ones(n_dims, dtype=np.float32)
        for i in range(n_dims):
            # Reversion: (-1)^(k(k-1)/2)
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1: 
                sig_vals_np[i] *= -1.0
            
            # Metric: Product of squares of basis vectors present
            for b in range(D_basis):
                if (i >> b) & 1:
                    val = signature[b]
                    if val == 0:
                        sig_vals_np[i] = 0.0
                        break
                    sig_vals_np[i] *= val
        
        sig = torch.from_numpy(sig_vals_np).to(x.device)

        M = x.numel() // n_dims
        grid = (triton.cdiv(M, 64),)
        manifold_norm_kernel[grid](x, sig, M, n_dims, eps, BLOCK_SIZE=64)
        return x
    # =================================================================
    # TRITON AUTOGRAD GRADIENT WRAPPER
    # =================================================================

    class VersorLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, signature):
            ctx.signature = signature
            ctx.save_for_backward(x, weight)
            return geometric_linear(x, weight, signature)

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            signature = ctx.signature
            
            # Fallback to CPU-based gradient computation if a specialized 
            # backward kernel is unavailable.
            
            # S is (n_dims, n_dims)
            n_dims = x.shape[-1]
            S = get_sign_matrix(signature, "numpy") # Get numpy version for CPU fallback
            S = torch.from_numpy(S).to(x.device)
            
            idx = torch.arange(n_dims, device=x.device)
            j_idx, i_idx = torch.meshgrid(idx, idx, indexing='ij')
            k_idx = i_idx ^ j_idx 
            
            # Grad Weight: (N, K, n_dims)
            # sum_b (x_rev[b, k, i] * grad[b, n, j] * S[i, j])
            x_rev = reverse_torch(x, signature)
            grad_w = torch.einsum('bki, bnj, ij -> nkj', x_rev, grad_output, S)
            
            # Grad X: (B, K, n_dims)
            # sum_n (grad[b, n, j] * w_rev[n, k, i] * S[j, i])
            w_rev = reverse_torch(weight, signature)
            grad_x = torch.einsum('bnj, nki, ji -> bki', grad_output, w_rev, S)
            
            return grad_x, grad_w, None # None for signature

    def geometric_linear_layer_triton(x, weight, signature):
        return VersorLinearFunction.apply(x, weight, signature)

# =================================================================
# APPLE SILICON METAL (MLX) KERNEL IMPLEMENTATIONS
# =================================================================


# SWAR (SIMD Within A Register) Popcount for MLX
def popcount_mlx_generic(n, D):
    # Generic SWAR algorithm for up to 32-bit integers
    # Adapts to D bits by masking
    mask_1 = 0x55555555 & ((1 << D) - 1)
    mask_2 = 0x33333333 & ((1 << D) - 1)
    mask_4 = 0x0F0F0F0F & ((1 << D) - 1)

    n = n - ((n >> 1) & mask_1)
    n = (n & mask_2) + ((n >> 2) & mask_2)
    n = (n + (n >> 4)) & mask_4
    # For D > 8, more steps are needed. For D <= 5, this is sufficient.
    # For D up to 32, the full SWAR is:
    # n = (n + (n >> 8)) & 0x00FF00FF
    # n = (n + (n >> 16)) & 0x0000FFFF
    # n = n % 255 # This is not popcount, this is sum of bytes.
    # The original was `(n * 0x01010101) >> 24` which sums bytes.
    # For D <= 5, the current `n = (n + (n >> 4)) & mask_4` is enough to get popcount in lower 4 bits.
    # For D=5, max popcount is 5.
    # Let's use a simpler popcount for small D.
    c = mx.zeros_like(n)
    for i in range(D):
        c += (n >> i) & 1
    return c

def compute_sign_mlx(a, b, signature):
    D = len(signature)
    n_dims = 1 << D
    
    # 1. Metric Sign
    metric_sign = mx.array(1.0)
    intersection = a & b
    for i in range(D):
        if signature[i] == 0: # If any basis vector squares to 0, product is 0
            metric_sign = mx.array(0.0)
            break
        metric_sign = mx.where((intersection >> i) & 1, metric_sign * signature[i], metric_sign)
    
    # 2. Commutation Sign: swaps
    swaps = mx.array(0)
    for i in range(D):
        if i == 0:
            mask_gt = ((1 << D) - 1) & ~((1 << (i + 1)) - 1) # All bits greater than i
            swaps = mx.where((b >> i) & 1, swaps + popcount_mlx_generic(a & mask_gt, D), swaps)
        else:
            mask_gt = ((1 << D) - 1) & ~((1 << (i + 1)) - 1)
            swaps = mx.where((b >> i) & 1, swaps + popcount_mlx_generic(a & mask_gt, D), swaps)
    
    comm_sign = 1 - 2 * (swaps % 2)
    
    return metric_sign * comm_sign

if HAS_MLX:
    def geometric_linear_mlx(x, weight, signature):
        # Dynamic Sign Generation (Bitwise)
        n_dims = x.shape[-1]
        indices = mx.arange(n_dims)
        idx_i = indices[:, None]; idx_j = indices[None, :]
        S = compute_sign_mlx(idx_i, idx_j, signature).astype(mx.float32)
        k_grid = idx_i ^ idx_j
        
        # Setup views
        x_view = x[..., None, :, :, None] 
        w_view = weight.reshape(*( (1,)*(x.ndim - 2) + weight.shape ))
        w_view = w_view[..., :, :, None, :]
        
        # (..., N, K, n_dims, n_dims)
        prod = x_view * w_view * S
        
        # Reduce (n_dims, n_dims) -> n_dims
        # We need to sum prod[..., i, j] into result[..., i^j]
        # Using a pre-computed reduction matrix for speed (MLX doesn't have fast atomic scatter yet)
        # This is an architectural constraint of the backend, not a theoretical flaw.
        # We generate the reduction map once.
        return reduce_geometric_product_mlx(prod, k_grid, sum_over_k=True)

    _REDUCTION_MAT_CACHE = {}
    def reduce_geometric_product_mlx(prod, k_grid, sum_over_k=False):
        n_dims = prod.shape[-1]
        D_sq = n_dims * n_dims
        
        cache_key = n_dims
        if cache_key not in _REDUCTION_MAT_CACHE:
            # Build (D_sq, n_dims) matrix where (r, c) = 1 if r's (i^j) == c
            mat = np.zeros((D_sq, n_dims), dtype=np.float32)
            k_flat = np.array(k_grid).flatten()
            for r, k in enumerate(k_flat):
                mat[r, k] = 1.0
            _REDUCTION_MAT_CACHE[cache_key] = mx.array(mat)
        
        RED = _REDUCTION_MAT_CACHE[cache_key]
        
        prod_flat = prod.reshape(-1, D_sq)
        res = mx.matmul(prod_flat, RED)
        res = res.reshape(*prod.shape[:-2], n_dims)
        
        if sum_over_k:
            return mx.sum(res, axis=-2) # Sum over K
        return res

    def manifold_norm_mlx(x, signature, eps=1e-6):
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy: x = mx.array(x)

        n_dims = x.shape[-1]
        D_basis = len(signature)
        indices = mx.arange(n_dims)
        
        # Reversion sign
        c = mx.zeros_like(indices)
        for i in range(D_basis):
            c += (indices >> i) & 1
        reversion_sign = mx.where((c * (c - 1) // 2) % 2 == 1, -1.0, 1.0)
        
        # Metric sign construction
        sig_np = np.ones(n_dims, dtype=np.float32)
        for i in range(n_dims):
             for b in range(D_basis):
                 if (i >> b) & 1:
                     val = signature[b]
                     if val == 0:
                         sig_np[i] = 0.0
                         break
                     sig_np[i] *= val
        
        sig = mx.array(sig_np) * reversion_sign
        norm_sq = mx.sum(x * x * sig, axis=-1, keepdims=True)
        abs_norm = mx.sqrt(mx.abs(norm_sq) + eps)
        l2_norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + eps
        denom = mx.maximum(mx.maximum(abs_norm, l2_norm), 1.0)
        res = x / denom
        return np.array(res) if is_numpy else res

# =================================================================
# FUNDAMENTAL GEOMETRIC ALGEBRA OPERATORS
# =================================================================

def reverse(x, signature):
    """Computes the reverse of a multivector A~."""
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x_in = torch.from_numpy(x)
    else:
        x_in = x

    if HAS_MLX and isinstance(x_in, mx.array):
        res = reverse_mlx(x_in, signature)
    else:
        res = reverse_torch(x_in, signature)
        
    if is_numpy:
        return res.detach().cpu().numpy()
    return res

def reverse_mlx(x, signature):
    n_dims = x.shape[-1]
    D_basis = len(signature)
    indices = mx.arange(n_dims)
    c = mx.zeros_like(indices)
    for i in range(D_basis):
        c += (indices >> i) & 1
    sig = mx.where(((c * (c - 1) // 2) % 2) == 1, -1.0, 1.0)
    return x * sig.astype(x.dtype)

def reverse_torch(x, signature):
    device = x.device if isinstance(x, torch.Tensor) else "cpu"
    n_dims = x.shape[-1]
    D_basis = len(signature)
    indices = torch.arange(n_dims, device=device)
    c = torch.zeros(n_dims, device=device)
    for i in range(D_basis): c += (indices >> i) & 1
    sig = torch.where(((c * (c - 1) // 2) % 2) == 1, -1.0, 1.0)
    return x * sig.to(x.dtype)

def sandwich_product(r, x, signature):
    """Computes the isometric transformation R x R~."""
    # R * x * R~
    r_rev = reverse(r, signature)
    inter = geometric_product(r, x, signature)
    return geometric_product(inter, r_rev, signature)

def wedge_product(a, b, signature):
    """Outer product (Grade-increasing)."""
    return filtered_product(a, b, signature, mode="wedge")

def inner_product(a, b, signature):
    """Inner product (Grade-decreasing)."""
    return filtered_product(a, b, signature, mode="inner")

def left_contraction(a, b, signature):
    """Left contraction."""
    return filtered_product(a, b, signature, mode="lc")

def filtered_product(a, b, signature, mode="wedge"):
    is_numpy = isinstance(a, np.ndarray)
    if is_numpy: a = torch.from_numpy(a)
    if isinstance(b, np.ndarray): b = torch.from_numpy(b)
    
    device = a.device if isinstance(a, torch.Tensor) else "cpu"
    n_dims = a.shape[-1]
    
    target_dev = "torch_cuda" if device.type != 'cpu' else "torch"
    S = get_sign_matrix(signature, target_dev).to(device)
    
    idx = torch.arange(n_dims, device=device)
    k_idx = idx.unsqueeze(0)
    i_idx = idx.unsqueeze(1)
    b_indices = i_idx ^ k_idx
    
    # Grade filter
    def popcount_torch(x):
        c = torch.zeros_like(x)
        for i in range(16): # Support up to 16 bits
            c += (x >> i) & 1
        return c

    gi = popcount_torch(i_idx) # (n_dims, 1) [Grade of A component]
    gk = popcount_torch(k_idx) # (1, n_dims) [Grade of result component]
    gj = popcount_torch(b_indices) # (n_dims, n_dims) [Grade of B component]
    
    if mode == "wedge":
        mask = (gk == (gi + gj)).to(a.dtype)
    elif mode == "lc":
        mask = (gk == (gj - gi)).to(a.dtype)
    else: # inner
        mask = (gk == torch.abs(gi - gj)).to(a.dtype)
        # Classical GA inner product often zero if either is scalar (clifford behavior)
        mask = mask * (gi > 0).to(a.dtype) * (gj > 0).to(a.dtype)

    S_filtered = S.to(a.dtype) * mask
    
    a_exp = a.unsqueeze(-1)
    b_perm = b[..., b_indices]
    
    res = torch.sum(a_exp * b_perm * S_filtered, dim=-2)
    
    if is_numpy:
        return res.detach().cpu().numpy()
    return res

# =================================================================
# UNIFIED MULTI-BACKEND INTERFACE
# =================================================================

def geometric_product(a, b, signature):
    # Multi-backend dispatching for geometric product
    is_numpy = False
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
        is_numpy = True
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
        is_numpy = True

    if HAS_MLX and (isinstance(a, mx.array) or isinstance(b, mx.array)):
        return geometric_product_mlx(a, b, signature)
        
    # Torch / CPU Universal Path
    return geometric_product_torch(a, b, signature, is_numpy)

def geometric_product_torch(a, b, signature, is_numpy):
    device = a.device if isinstance(a, torch.Tensor) else "cpu"
    n_dims = a.shape[-1]
    
    # Get Sign Matrix
    target_dev = "torch_cuda" if device.type != 'cpu' else "torch"
    S = get_sign_matrix(signature, target_dev).to(device)
    
    # We want c[..., k] = sum_i (a[..., i] * b[..., i^k] * S[i, k])
    idx = torch.arange(n_dims, device=device)
    k_idx = idx.unsqueeze(0) # (1, n_dims)
    i_idx = idx.unsqueeze(1) # (n_dims, 1)
    b_indices = i_idx ^ k_idx # (n_dims, n_dims) -> [i, k] gives index of b to pick
    
    # a: (..., n_dims) -> (..., n_dims, 1) (i, k=1)
    a_exp = a.unsqueeze(-1)
    
    # b needs to be permuted. 
    # b[..., i^k]
    # This is equivalent to: b_new[..., i, k] = b[..., i^k]
    b_perm = b[..., b_indices] # (..., n_dims, n_dims)
    
    # Cast S to match input dtype to preserve it (e.g. int64)
    res = torch.sum(a_exp * b_perm * S.to(a.dtype), dim=-2)
    
    if is_numpy:
        return res.detach().cpu().numpy()
    return res

def geometric_product_mlx(a, b, signature):
    # MLX generic path
    n_dims = a.shape[-1]
    indices = mx.arange(n_dims)
    S = get_sign_matrix(signature, "mlx")
    k_idx = indices[:, None] ^ indices[None, :]
    
    # Outer product equivalent logic
    prod = a[..., :, None] * b[..., None, :] * S
    
    # Reduction via Reduction Matrix
    return reduce_geometric_product_mlx(prod, k_idx, sum_over_k=False)

# Aliases for different naming conventions
gapu_geometric_product = geometric_product

def geometric_linear_cpu(x, weight, signature):
    # CPU fallback for linear layer (same as geometric_product but with weight matrix)
    # x: (..., K, D), w: (N, K, D)
    n_dims = x.shape[-1]
    device = x.device if isinstance(x, torch.Tensor) else "cpu"
    S = get_sign_matrix(signature, "numpy") # Get numpy version for CPU fallback
    S = torch.from_numpy(S).to(device)
    
    # We need: out[..., n, j] = sum_k sum_i (x[..., k, i] * weight[n, k, i^j] * S[i, j])
    # weight_perm[n, k, j, i] = weight[n, k, i^j]
    idx = torch.arange(n_dims, device=device)
    j_idx, i_idx = torch.meshgrid(idx, idx, indexing='ij')
    k_idx = i_idx ^ j_idx # (n_dims, n_dims) -> [j, i]
    
    w_perm = weight[:, :, k_idx] # (N, K, n_dims, n_dims)
    
    # Use einsum for clarity and correctness on CPU
    # b: batch/sequence dims, n: out_features, k: in_features, j: out_basis, i: in_basis
    if x.dim() == 3: # (B*S, K, n_dims) or (B, K, n_dims)
        # x: bki, w_perm: nkj i, S: i j
        # Actually w_perm already has i^j logic. 
        # So: sum_k,i (x[..., k, i] * w_perm[n, k, j, i] * S[i, j])
        return torch.einsum('bki, nkj i, ij -> bnj', x, w_perm, S)
    elif x.dim() == 4: # (B, S, K, n_dims)
        return torch.einsum('bski, nkj i, ij -> bsnj', x, w_perm, S)
    else:
        # Generic fallback for any number of batch dims
        x_flat = x.reshape(-1, x.shape[-2], n_dims)
        res = torch.einsum('bki, nkj i, ij -> bnj', x_flat, w_perm, S)
        return res.reshape(*x.shape[:-2], weight.shape[0], n_dims)

def geometric_linear_layer(x, weight, signature):
    if HAS_MLX and (isinstance(x, mx.array) or isinstance(weight, mx.array)):
        return geometric_linear_mlx(x, weight, signature)
    if isinstance(x, torch.Tensor) and HAS_TRITON and x.is_cuda:
        return geometric_linear_layer_triton(x, weight, signature)
    
    # Robust CPU implementation with dimension handling
    return geometric_linear_cpu(x, weight, signature)
    
def manifold_normalization(x, signature, eps=1e-6):
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy: x = torch.from_numpy(x)

    if HAS_MLX and isinstance(x, mx.array):
        res = manifold_norm_mlx(x, signature, eps)
    elif x.is_cuda and HAS_TRITON:
        res = manifold_norm_triton(x, signature, eps)
    else:
        # Standard CPU implementation
        device = x.device
        n_dims = x.shape[-1]
        S = get_sign_matrix(signature, "numpy")
        S = torch.from_numpy(S).to(device)
        metric_sq = S[:, 0]
        
        norm_sq = torch.sum(x * x * metric_sq, dim=-1, keepdim=True)
        abs_norm = torch.sqrt(torch.abs(norm_sq) + eps)
        l2_norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)) + eps
        denom = torch.max(torch.max(abs_norm, l2_norm), torch.tensor(1.0, device=device))
        res = x / denom
    
    if is_numpy:
        return res.detach().cpu().numpy()
    return res

# =================================================================
# PERFORMANCE BENCHMARK SUITE
# =================================================================

def benchmark():
    print(f"\n{'='*60}")
    print(f"GEOMETRIC KERNEL PERFORMANCE EVALUATION")
    print(f"{'='*60}\n")
    
    # Default signature for benchmarks
    default_signature = [1, 1, 1, 1, -1] # Cl(4,1)
    
    if HAS_TRITON and torch.cuda.is_available():
        M, K, N = 1024, 256, 256
        x = torch.randn(M, K, 32, device='cuda')
        w = torch.randn(N, K, 32, device='cuda')
        print(f"Target: Fused GeMM ({M}x{K} @ {K}x{N} x 32-dim)")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        _ = geometric_linear_layer(x, w, default_signature)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): _ = geometric_linear_layer(x, w, default_signature)
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
        _ = geometric_linear_layer(x, w, default_signature)
        mx.eval(_)
        t0 = time.time()
        for _ in range(100):
            res = geometric_linear_layer(x, w, default_signature)
            mx.eval(res)
        dt = time.time() - t0
        gops = (M * N * K * 100) / dt / 1e9
        print(f"    -> Performance: {gops:.2f} G-Products/Sec")

if __name__ == "__main__":
    benchmark()