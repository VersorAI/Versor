import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import gc
import sys
import os
import time
import math
import warnings
import datetime

# Optional imports for plotting and metrics
try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.metrics import matthews_corrcoef
except ImportError:
    # Fallback implementation if sklearn is not installed
    def matthews_corrcoef(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0: return 0
        return numerator / denominator

# =================================================================
# 0. EXPERIMENTAL CONFIGURATION
# =================================================================

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def print_header():
    print(f"\n{'='*80}")
    print(f"BENCHMARK REPORT: TOPOLOGICAL CONNECTIVITY ESTIMATION")
    print(f"{'='*80}")
    print(f"Date:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware:     {GPU_NAME}")
    print(f"Task:         Path Continuity ('Broken Snake' Dataset)")
    print(f"Config:       Batch 64 | FP32 (Versor) | High-Gain Attention")
    print(f"{'='*80}\n")

# =================================================================
# 1. ALGEBRA KERNEL (Cl(4,1))
# =================================================================

GP_MAP_CACHE = {}
SIG_CACHE = {}

def compute_basis_product_cl41(a: int, b: int):
    """Sign and index for Cl(4,1) basis multiplication."""
    sign, a_bits = 1.0, a
    for i in range(5):
        if (b >> i) & 1:
            for j in range(i + 1, 5):
                if (a_bits >> j) & 1: 
                    sign *= -1.0
            if (a_bits >> i) & 1:
                if i == 4: sign *= -1.0
                a_bits &= ~(1 << i)
            else: 
                a_bits |= (1 << i)
    return sign, a_bits

def get_gp_map(device: torch.device):
    idx = device.index if device.index is not None else 0
    if idx not in GP_MAP_CACHE:
        table = torch.zeros((32, 32, 32))
        for a in range(32):
            for b in range(32):
                s, r = compute_basis_product_cl41(a, b)
                table[a, b, r] = s
        GP_MAP_CACHE[idx] = table.to(device)
    return GP_MAP_CACHE[idx]

def get_metric_signature(device: torch.device):
    idx = device.index if device.index is not None else 0
    if idx not in SIG_CACHE:
        sig = torch.ones(32, device=device)
        for i in range(32):
            if (i >> 4) & 1: sig[i] *= -1.0
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1: sig[i] *= -1.0
        SIG_CACHE[idx] = sig
    return SIG_CACHE[idx]

def manifold_normalization(A: torch.Tensor, eps: float = 1e-6):
    """Keeps activations on the manifold to prevent gradient explosion."""
    sig = get_metric_signature(A.device)
    norm_sq = torch.sum(A * A * sig, dim=-1)
    denom = torch.max(
        torch.sqrt(torch.abs(norm_sq) + eps).unsqueeze(-1),
        torch.norm(A, p=2, dim=-1, keepdim=True) / 4.0 + eps
    ).clamp(min=1.0)
    return A / denom

def conformal_projection(grid: torch.Tensor):
    """
    Project Grid -> Vectors.
    Geometry is handled by Learnable Positional Embeddings downstream.
    """
    b, seq_len = grid.shape
    n_o = torch.zeros(32, device=DEVICE)
    n_o[16], n_o[8] = 0.5, -0.5
    
    n_inf = torch.zeros(32, device=DEVICE)
    n_inf[16], n_inf[8] = 1.0, 1.0

    out = torch.zeros((b, seq_len, 32), device=DEVICE)
    out += (grid == 1).unsqueeze(-1) * n_o 
    out += (grid == 0).unsqueeze(-1) * n_inf
    return out

# =================================================================
# 2. MODEL ARCHITECTURES
# =================================================================

class VersorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 32))
        with torch.no_grad():
            # Conservative init for stability
            std = 0.5 / (in_features * 32)**0.5
            self.weight.normal_(0, std)

    def forward(self, x: torch.Tensor):
        gp = get_gp_map(x.device)
        W_op = torch.einsum('oij,jlk->oilk', self.weight, gp)
        out = torch.einsum('bsil,oilk->bsok', x, W_op)
        return manifold_normalization(out)

class VersorAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = VersorLinear(d_model, d_model)
        self.k_proj = VersorLinear(d_model, d_model)
        self.v_proj = VersorLinear(d_model, d_model)
        self.o_proj = VersorLinear(d_model, d_model)
        
        # FIX: Higher initial scale for long sequences (32x32=1024)
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, x: torch.Tensor):
        b, s, d, _ = x.shape
        
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        
        sig = get_metric_signature(x.device)
        q_flat = (q * sig).flatten(3) 
        k_flat = k.flatten(3)
        
        # RESTORED STABILITY: Divisor + Learnable Multiplier
        score = torch.matmul(q_flat, k_flat.transpose(-1, -2))
        score = score / (self.d_head * 32)**0.5
        score = score * self.scale
        
        attn_weights = torch.softmax(score, dim=-1)
        out = torch.einsum('bhsi,bhidl->bhsdl', attn_weights, v)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, d, 32))

class CGA_Transformer(nn.Module):
    def __init__(self, d_vectors: int = 4, n_layers: int = 4, seq_len: int = 1024):
        super().__init__()
        # Learnable Geometric Positional Embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_vectors, 32) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': VersorAttention(d_vectors),
                'mlp': nn.Sequential(
                    VersorLinear(d_vectors, d_vectors*4),
                    nn.Tanh(),
                    VersorLinear(d_vectors*4, d_vectors)
                )
            }) for _ in range(n_layers)
        ])
        self.pool = VersorLinear(d_vectors, d_vectors)
        self.head = nn.Linear(d_vectors*32, 2)

    def forward(self, x: torch.Tensor):
        x = x + self.pos_emb[:, :x.shape[1], :, :]
        for layer in self.layers:
            x = manifold_normalization(x + layer['attn'](x))
            x = manifold_normalization(x + layer['mlp'](x))
        pooled = self.pool(x).mean(dim=1) 
        pooled = torch.tanh(pooled).view(x.shape[0], -1)
        return self.head(pooled)

class Standard_Transformer(nn.Module):
    def __init__(self, d_input: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor):
        # x starts as (B, S, d_vectors, 32)
        x = x.view(x.shape[0], x.shape[1], -1) 
        # x is now (B, S, D_in)
        
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x.mean(dim=1))

# =================================================================
# 3. DATASET GENERATION
# =================================================================

def generate_dataset(size: int, n_samples: int, d_vectors: int = 4):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Data] Generating {n_samples} samples ({size}x{size}, d={d_vectors})...", end=" ", flush=True)
    X_list, Y_list = [], []
    for _ in range(n_samples):
        grid = np.zeros((size, size), dtype=np.float32)
        path_coords = [(0,0)]
        curr = (0,0)
        grid[0,0] = 1

        while curr != (size-1, size-1):
            cx, cy = curr
            moves = []
            if cx < size-1: moves.append((cx+1, cy))
            if cy < size-1: moves.append((cx, cy+1))
            if not moves: break
            
            next_step = moves[np.random.randint(len(moves))]
            
            # Logic to occasionally skip path addition (no-op in original code, kept for consistency)
            if np.random.rand() < 0.3 and cx > 0 and (cx-1, cy) not in path_coords: 
                pass 
            
            path_coords.append(next_step)
            grid[next_step] = 1
            curr = next_step
            
        is_solvable = (np.random.rand() > 0.5)
        if is_solvable: 
            label = 1
        else:
            label = 0
            if len(path_coords) > 5:
                cut_idx = np.random.randint(2, len(path_coords)-2)
                grid[path_coords[cut_idx]] = 0 
        
        X_list.append(grid.flatten())
        Y_list.append(label)
        
    X_t = torch.tensor(np.array(X_list), device=DEVICE)
    Y_t = torch.tensor(np.array(Y_list), device=DEVICE)
    # Project and repeat for d_vectors
    X_proj = conformal_projection(X_t).unsqueeze(2).repeat(1, 1, d_vectors, 1)
    print("Completed.")
    return X_proj, Y_t

# =================================================================
# 4. EXPERIMENTAL ROUTINE
# =================================================================

def run_traininv_c_versorycle(model_name: str, model: nn.Module, size: int, d_vectors: int):
    # Dynamic sample size: More samples for harder (larger) grids
    n_samples = 3000
    if size >= 32: n_samples = 6000

    X, Y = generate_dataset(size, n_samples, d_vectors)
    dataset = TensorDataset(X, Y)
    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset)-train_len])

    # CRITICAL MEMORY FIX: BATCH SIZE MANAGEMENT
    if size <= 16:   
        BATCH_SIZE = 64
        ACCUM_STEPS = 1
    elif size == 32:            
        BATCH_SIZE = 16
        ACCUM_STEPS = 4  # 16 * 4 = 64 Effective Batch
    else: # size >= 64
        BATCH_SIZE = 4
        ACCUM_STEPS = 16 # 4 * 16 = 64 Effective Batch

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.to(DEVICE)

    # CRITICAL: LOWER LR FOR 32x32 TO PREVENT OVERSHOOT
    current_lr = 0.001 if size <= 16 else 0.0005
    optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.CrossEntropyLoss()

    use_amp = "Standard" in model_name
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Train] Model: {model_name} | Batch: {BATCH_SIZE} (x{ACCUM_STEPS} acc) | AMP: {'ON' if use_amp else 'OFF'}")
    print(f"           Progress: ", end="", flush=True)

    history_mcc = []
    start_time = time.time()
    EPOCHS = 25

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        for i, (xb, yb) in enumerate(train_loader):
            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(xb)
                loss = criterion(preds, yb)
                loss = loss / ACCUM_STEPS 
            
            if use_amp:
                scaler.scale(loss).backward()
                if (i + 1) % ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (i + 1) % ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(xb)
                preds_all.extend(outputs.argmax(1).cpu().numpy())
                targets_all.extend(yb.cpu().numpy())
        mcc = matthews_corrcoef(targets_all, preds_all)
        history_mcc.append(mcc)
        if epoch % 5 == 0: print(".", end="", flush=True)
            
    total_time = time.time() - start_time
    print(f" Completed.")
    print(f"           Result:   Final MCC = {mcc:.3f} | Total Time = {total_time:.1f}s")

    del X, Y, dataset, train_loader, val_loader; gc.collect(); torch.cuda.empty_cache()
    return history_mcc, mcc

# =================================================================
# 5. MAIN ENTRY POINT
# =================================================================

if __name__ == "__main__":
    print_header()
    
    # MODIFIED: Reverted to reaching 32x32 only
    SIZES = [8, 16, 32] 
    results_summary = {}

    print(f"{'Grid Size':<12} | {'Geometric (MCC)':<18} | {'Standard (MCC)':<18} | {'Delta':<10}")
    print("-" * 65)

    for size in SIZES:
        # GEOMETRIC MODEL SETUP
        # Give more capacity for larger grids
        d_vec = 8 if size >= 32 else 4
        model_versor = CGA_Transformer(d_vectors=d_vec, n_layers=4, seq_len=size*size)
        hist_geo, mcc_geo = run_traininv_c_versorycle(f"CGA-Transformer ({size}x{size})", model_versor, size, d_vec)
        
        # STANDARD MODEL SETUP
        # Scale standard model to match input dimensionality (d_vec * 32)
        d_std_input = d_vec * 32
        # d_model = d_std_input // 2 if size < 32 else d_std_input # (Logic from notes)
        d_std_model = 128 if size < 32 else 256
        
        model_std = Standard_Transformer(d_input=d_std_input, d_model=d_std_model, n_heads=4, n_layers=4)
        
        hist_std, mcc_std = run_traininv_c_versorycle(f"Standard-Attention ({size}x{size})", model_std, size, d_vec)
        
        delta = mcc_geo - mcc_std
        print(f"\r{size}x{size:<9} | {mcc_geo:.3f}              | {mcc_std:.3f}              | {delta:+.3f}")
        results_summary[size] = {'versor': hist_geo, 'std': hist_std}
        print("-" * 65)
        
    print("\nBenchmark sequence complete.")
    
    # Plotting results
    if sns:
        sns.set_theme(style="whitegrid", context="paper")
    else:
        plt.style.use('ggplot')
        
    plt.figure(figsize=(10, 6))
    for idx, size in enumerate(SIZES):
        plt.plot(results_summary[size]['versor'], label=f"CGA ({size}x{size})", linewidth=2.5, alpha=0.8)
        plt.plot(results_summary[size]['std'], label=f"Std ({size}x{size})", linewidth=2.5, linestyle='--', alpha=0.5)
    
    plt.title("Convergence Analysis: Topological Connectivity")
    plt.ylabel("MCC")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("topology_benchmark_results.png", dpi=300)
    print(f"[Output] Performance graph saved to: topology_benchmark_results.png")

    # =================================================================
    # PART D: COMPUTE KERNEL BENCHMARK (MLX vs PYTORCH)
    # =================================================================
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import mlx.core as mx
        import kernel
        
        print("\n" + "="*80)
        print("PART D: COMPUTE KERNEL BENCHMARK (Geometric Product)")
        print("="*80)
        
        # Test Case: Matrix Multiplication of Multivectors
        # (Batch, 256, 32) * (256, 512, 32) via Geometric Product
        # Note: kernel.gapu_geometric_product is element-wise broadcastable (A * B)
        # To match a Linear Layer (MatMul), we need broadcasting or simple element-wise throughput.
        # Let's benchmark Element-Wise Product throughput (A * B) as a proxy for raw arithmetic power.
        
        B_SIZE = 4096 * 4
        DIM = 32
        
        # 1. PyTorch
        print(f"[PyTorch] Geometric Product (Element-wise) {B_SIZE} vectors...")
        pt_a = torch.randn(B_SIZE, DIM, device=DEVICE)
        pt_b = torch.randn(B_SIZE, DIM, device=DEVICE)
        
        # Using the helper from this file (compute_basis_product_cl41 logic is cached in GP_MAP)
        # Note: 'VersorLinear' uses einsum with GP_MAP.
        gp_map = get_gp_map(DEVICE)
        
        # Element-wise product logic in PyTorch using the table:
        # A_i * B_j * Table_ijk -> C_k
        # Element-wise: (B, 32), (B, 32) -> (B, 32)
        # We can simulate this with einsum:
        
        def pt_gp_elementwise(a, b):
            # bs: batch, i: input, j: input, k: output
            return torch.einsum('bi, bj, ijk -> bk', a, b, gp_map)
            
        # Warmup
        for _ in range(5): _ = pt_gp_elementwise(pt_a, pt_b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        t0 = time.time()
        for _ in range(50):
            _ = pt_gp_elementwise(pt_a, pt_b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt_pt = time.time() - t0
        ops_pt = (B_SIZE * 50) / dt_pt
        print(f"    -> PyTorch Speed: {ops_pt:,.0f} Products/sec")
        
        # 2. MLX
        print(f"[MLX] GAPU Kernel (Metal)...")
        mx_a = mx.array(pt_a.cpu().numpy())
        mx_b = mx.array(pt_b.cpu().numpy())
        
        # Warmup
        _ = kernel.gapu_geometric_product(mx_a, mx_b)
        mx.eval(_)
        
        t0 = time.time()
        for _ in range(50):
            out = kernel.gapu_geometric_product(mx_a, mx_b)
            mx.eval(out)
        dt_mx = time.time() - t0
        ops_mx = (B_SIZE * 50) / dt_mx
        print(f"    -> MLX Speed:     {ops_mx:,.0f} Products/sec")
        
        print(f"    -> Speedup: {ops_mx / ops_pt:.2f}x")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"[Warning] MLX Benchmark skipped: {e}")