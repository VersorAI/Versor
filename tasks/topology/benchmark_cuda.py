import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
import sys
import time
import math
import warnings
import datetime
import copy
import os

# =================================================================
# EXPERIMENTAL CONFIGURATION AND UTILITIES
# =================================================================

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def matthews_corrcoef(y_true, y_pred):
    """
    Manual MCC calculation (No sklearn dependency).
    Range: -1 (Total Disagreement) to +1 (Total Agreement)
    """
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0: return 0.0
    return numerator / denominator

def print_header():
    print(f"\n{'='*80}")
    print(f"BENCHMARK: GEOMETRIC ALGEBRA vs STANDARD ATTENTION")
    print(f"{'='*80}")
    print(f"Date:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware:     {GPU_NAME}")
    print(f"Strategy:     Curriculum Learning (8->16->32)")
    print(f"{'='*80}\n")

# =================================================================
# CONFORMAL GEOMETRIC ALGEBRA CORE (Cl(4,1))
# =================================================================

GP_MAP_CACHE = {}
SIG_CACHE = {}

def compute_basis_product_cl41(a: int, b: int):
    """
    Returns the metric scalar part and basis index for Cl(4,1) multiplication.
    """
    sign, a_bits = 1.0, a
    for i in range(5):
        if (b >> i) & 1:
            for j in range(i + 1, 5):
                if (a_bits >> j) & 1: sign *= -1.0
            if (a_bits >> i) & 1:
                if i == 4: sign *= -1.0 # Metric signature (-1 for e4)
                a_bits &= ~(1 << i)
            else: a_bits |= (1 << i)
    return sign, a_bits

def get_gp_map(device):
    idx = device.index if device.index is not None else 0
    if idx not in GP_MAP_CACHE:
        # Precompute Cayley Table
        table = torch.zeros((32, 32, 32))
        for a in range(32):
            for b in range(32):
                s, r = compute_basis_product_cl41(a, b)
                table[a, b, r] = s
        GP_MAP_CACHE[idx] = table.to(device)
    return GP_MAP_CACHE[idx]

def get_metric_signature(device):
    idx = device.index if device.index is not None else 0
    if idx not in SIG_CACHE:
        sig = torch.ones(32, device=device)
        for i in range(32):
            # Apply metric signature for Cl(4,1)
            if (i >> 4) & 1: sig[i] *= -1.0
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1: sig[i] *= -1.0
        SIG_CACHE[idx] = sig
    return SIG_CACHE[idx]

def manifold_normalization(A: torch.Tensor, eps: float = 1e-6):
    """
    Manifold-preserving normalization: ensures multivectors remain within 
    the transformation group, preventing numerical divergence in 
    isometric sequence modeling.
    """
    sig = get_metric_signature(A.device)
    norm_sq = torch.sum(A * A * sig, dim=-1)
    # Robust normalization handling null vectors
    denom = torch.max(
        torch.sqrt(torch.abs(norm_sq) + eps).unsqueeze(-1),
        torch.norm(A, p=2, dim=-1, keepdim=True) / 4.0 + eps
    ).clamp(min=1.0)
    return A / denom

def conformal_projection(grid: torch.Tensor):
    """
    Project Binary Grid (0/1) -> Conformal Vectors (Point / Infinity).
    """
    b, seq_len = grid.shape
    device = grid.device # Use the input tensor's device
    # n_o = 0.5(e4-e3), n_inf = e4+e3 in standard CGA basis mapping
    n_o = torch.zeros(32, device=device); n_o[16], n_o[8] = 0.5, -0.5
    n_inf = torch.zeros(32, device=device); n_inf[16], n_inf[8] = 1.0, 1.0
    
    out = torch.zeros((b, seq_len, 32), device=device)
    # 1 -> Point at Origin, 0 -> Point at Infinity (Simplified Topology)
    out += (grid == 1).unsqueeze(-1) * n_o 
    out += (grid == 0).unsqueeze(-1) * n_inf
    return out

# =================================================================
# ARCHITECTURAL IMPLEMENTATIONS: GEOMETRIC VS EUCLIDEAN
# =================================================================

class VersorLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Weight shape: (Out, In, 32) -> Multivector Weights
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 32))
        with torch.no_grad():
            std = 0.5 / (in_features * 32)**0.5
            self.weight.normal_(0, std)

    def forward(self, x):
        gp = get_gp_map(x.device)
        # Multivector linear contraction via Geometric Product
        W_op = torch.einsum('oij,jlk->oilk', self.weight, gp)
        out = torch.einsum('bsil,oilk->bsok', x, W_op)
        return manifold_normalization(out)

class VersorAttention(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = VersorLinear(d_model, d_model)
        self.k_proj = VersorLinear(d_model, d_model)
        self.v_proj = VersorLinear(d_model, d_model)
        self.o_proj = VersorLinear(d_model, d_model)
        
        # High initial scale to sharpen geometric interactions
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, x):
        b, s, d, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        
        sig = get_metric_signature(x.device)
        q_flat = (q * sig).flatten(3) 
        k_flat = k.flatten(3)
        
        # Rotational Similarity Score
        score = torch.matmul(q_flat, k_flat.transpose(-1, -2))
        score = score / (self.d_head * 32)**0.5
        score = score * self.scale
        
        attn_weights = torch.softmax(score, dim=-1)
        out = torch.einsum('bhsi,bhidl->bhsdl', attn_weights, v)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, d, 32))

class CGA_Transformer(nn.Module):
    def __init__(self, d_vectors=4, n_layers=4, seq_len=1024):
        super().__init__()
        # Positional embeddings (Specific to grid size)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_vectors, 32) * 0.02)
        
        # The "Brain" (Invariant to grid size)
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

    def forward(self, x):
        # Add Positional Embeddings
        x = x + self.pos_emb[:, :x.shape[1], :, :]
        for layer in self.layers:
            # Residual + Normalization
            x = manifold_normalization(x + layer['attn'](x))
            x = manifold_normalization(x + layer['mlp'](x))
        
        # Global Pooling
        pooled = self.pool(x).mean(dim=1) 
        pooled = torch.tanh(pooled).view(x.shape[0], -1)
        return self.head(pooled)

class Standard_Transformer(nn.Module):
    def __init__(self, d_input, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        # Input Projection: d_input (d_vec * 32) -> d_model
        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        # Flatten CGA input: (B, S, D, 32) -> (B, S, D*32)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x.mean(dim=1))

# =================================================================
# DATA SYNTHESIS AND EXPERIMENTAL UTILITIES
# =================================================================

def generate_dataset(size: int, n_samples: int, d_vectors: int = 4):
    """
    Generates 'Broken Snake' topological dataset.
    Label 1: Continuous path from TL to BR.
    Label 0: Path broken by a single gap.
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Data] Generating {n_samples} samples ({size}x{size})...", end=" ", flush=True)
    
    # Store large datasets on CPU to avoid OOM
    target_dev = torch.device('cpu') if size >= 64 else DEVICE
    
    X_list, Y_list = [], []
    
    for _ in range(n_samples):
        grid = np.zeros((size, size), dtype=np.float32)
        path_coords = [(0,0)]; curr = (0,0); grid[0,0] = 1
        
        # Generate random walk
        while curr != (size-1, size-1):
            cx, cy = curr
            moves = []
            if cx < size-1: moves.append((cx+1, cy))
            if cy < size-1: moves.append((cx, cy+1))
            if not moves: break
            
            # Simple heuristic to move towards goal
            next_step = moves[np.random.randint(len(moves))]
            path_coords.append(next_step); grid[next_step] = 1; curr = next_step
            
        is_solvable = (np.random.rand() > 0.5)
        
        if is_solvable:
            label = 1
        else:
            label = 0
            if len(path_coords) > 5:
                # Break the path
                cut_idx = np.random.randint(2, len(path_coords)-2)
                grid[path_coords[cut_idx]] = 0 
                
        X_list.append(grid.flatten()); Y_list.append(label)
        
    X_t = torch.tensor(np.array(X_list), device=target_dev)
    Y_t = torch.tensor(np.array(Y_list), device=target_dev)
    
    # Project to CGA space
    X_proj = conformal_projection(X_t).unsqueeze(2).repeat(1, 1, d_vectors, 1)
    print("Done.")
    return X_proj, Y_t

def transfer_weights(source_model, target_model):
    CURRICULUM LEARNING TRANSFORMATION:
    Weight transfer of isometric layers from low-complexity to high-complexity 
    topological manifolds.
    """
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    
    # Filter out positional embeddings
    pretrained_dict = {k: v for k, v in source_dict.items() if k in target_dict and 'pos_emb' not in k}
    
    # Load logic
    target_dict.update(pretrained_dict)
    target_model.load_state_dict(target_dict)
    print(f"    >>> Transfer Learning: Loaded {len(pretrained_dict)} geometric layers from previous stage.")
    return target_model

# =================================================================
# BENCHMARK EVALUATION ENGINE
# =================================================================

def run_cycle(name, model, size, d_vec, epochs=25, transfer_from=None):
    # Data Config
    n_samples = 3000 if size < 32 else 6000 # More samples for large grids
    X, Y = generate_dataset(size, n_samples, d_vec)
    dataset = TensorDataset(X, Y)
    train_len = int(0.9 * len(dataset))
    train, val = random_split(dataset, [train_len, len(dataset)-train_len])
    
    # Batch Scaling
    if size <= 16:   batch_size, accum = 64, 1
    elif size == 32: batch_size, accum = 16, 4
    else:            batch_size, accum = 4, 16 
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    
    # Transfer Learning
    if transfer_from:
        model = transfer_weights(transfer_from, model)
        
    model.to(DEVICE)
    
    # Optimizer
    lr = 0.001 if size <= 16 else 0.0005
    opt = optim.AdamW(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    use_amp = "Standard" in name
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    history = []
    history_acc = []
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Train] {name} | B:{batch_size}x{accum} | Epochs:{epochs}")
    
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(xb)
                loss = crit(out, yb) / accum
            
            if use_amp:
                scaler.scale(loss).backward()
                if (i+1) % accum == 0: scaler.step(opt); scaler.update(); opt.zero_grad()
            else:
                loss.backward()
                if (i+1) % accum == 0: opt.step(); opt.zero_grad()
        
        sched.step()
        
        # Validation
        if (ep+1) % 2 == 0 or ep == epochs-1:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        o = model(xb)
                    preds.extend(o.argmax(1).cpu().numpy())
                    trues.extend(yb.cpu().numpy())
            mcc = matthews_corrcoef(trues, preds)
            acc = np.mean(np.array(trues) == np.array(preds))
            history.append(mcc)
            history_acc.append(acc)
            print(f"    Ep {ep+1}/{epochs} | MCC: {mcc:.3f} | Acc: {acc:.2%}")
            
            # Early Exit for Curriculum
            if mcc > 0.99 and "CGA" in name:
                print(f" -> Converged! Stopping early.")
                break
    
    print(f"\n    Final MCC: {history[-1]:.3f} | Final Acc: {history_acc[-1]:.2%}")
    del X, Y, dataset, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return history, history[-1], history_acc[-1], model

# =================================================================
# EXPERIMENTAL EXECUTION PROTOCOL (CURRICULUM LADDER)
# =================================================================

if __name__ == "__main__":
    print_header()
    
    SIZES = [8, 16, 32]
    EPOCH_MAP = {8: 20, 16: 30, 32: 40}
    D_VEC = 8 # Constant vector dim to allow weight transfer
    results = {}
    
    # ---------------------------------------------------------
    # PART A: GEOMETRIC ALGEBRA (CURRICULUM LEARNING)
    # ---------------------------------------------------------
    print(">>> MODE: GEOMETRIC ALGEBRA (With Curriculum Transfer)")
    prev_model = None
    
    for size in SIZES:
        print(f"\n--- STAGE: {size}x{size} Grid ---")
        seq_len = size * size
        
        # Init Model
        cga_model = CGA_Transformer(d_vectors=D_VEC, n_layers=4, seq_len=seq_len)
        
        # Train (Transferring from previous size if available)
        hist, mcc, acc, trained_model = run_cycle(
            f"CGA-{size}", cga_model, size, D_VEC, 
            epochs=EPOCH_MAP[size], 
            transfer_from=prev_model
        )
        
        results[size] = {'versor_hist': hist, 'geo_mcc': mcc, 'cga_acc': acc}
        prev_model = trained_model # Save for next stage
        
    # ---------------------------------------------------------
    # PART B: STANDARD ATTENTION (BASELINE)
    # ---------------------------------------------------------
    print("\n\n>>> MODE: STANDARD TRANSFORMER (Baseline)")
    
    for size in SIZES:
        print(f"\n--- STAGE: {size}x{size} Grid ---")
        d_input = D_VEC * 32
        
        # INCREASED CAPACITY FOR FAIR COMPARISON AT LARGE SCALES
        # 32x32 and 64x64 get d_model=256 to avoid bottlenecking
        d_std_model = 256 if size >= 32 else 128
        
        # Standard models don't benefit much from transfer learning on this task
        # because the positional embeddings dominate the logic. We train fresh.
        std_model = Standard_Transformer(d_input=d_input, d_model=d_std_model)
        
        hist, mcc, acc, _ = run_cycle(
            f"Std-{size}", std_model, size, D_VEC, 
            epochs=EPOCH_MAP[size]
        )
        
        results[size]['std_hist'] = hist
        results[size]['std_mcc'] = mcc
        results[size]['std_acc'] = acc

    # ---------------------------------------------------------
    # PART C: VISUALIZATION
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(f"{'Grid':<6} | {'Geometric':<10} | {'Standard':<10} | {'Delta'}")
    print("-" * 50)
    for s in SIZES:
        versor = results[s]['geo_mcc']
        std = results[s]['std_mcc']
        versor_acc = results[s].get('cga_acc', 0) # Assuming stored if I added it (I missed adding cga_acc above, but its fine for display logic)
        std_acc = results[s]['std_acc']
        print(f"{s}x{s:<3} | {versor:.3f}      | {std:.3f}      | {versor-std:+.3f} | (Std Acc: {std_acc:.2%})")
        
    # Plot
    plt.figure(figsize=(12, 8))
    for i, s in enumerate(SIZES):
        # Normalize x-axis to 0-1 range for comparison
        h_g = results[s]['versor_hist']
        h_s = results[s]['std_hist']
        
        # Pad shorter histories to match plotting
        max_len = max(len(h_g), len(h_s))
        if len(h_g) < max_len: h_g += [h_g[-1]] * (max_len - len(h_g))
        if len(h_s) < max_len: h_s += [h_s[-1]] * (max_len - len(h_s))
            
        x_axis = np.linspace(0, 100, max_len)
        
        plt.subplot(2, 2, i+1)
        plt.plot(x_axis, h_g, label='CGA (Ladder)', color='green', linewidth=2)
        plt.plot(x_axis, h_s, label='Standard', color='red', linestyle='--', linewidth=2)
        plt.title(f"Grid Size: {s}x{s}")
        plt.ylim(-0.1, 1.1); plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    
    # SAVE PROGRESS
    with open('benchmark_results.json', 'w') as f:
        # Convert numpy types to native python for JSON
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {}
            for sk, sv in v.items():
                if isinstance(sv, list): serializable_results[k][sk] = sv
                elif isinstance(sv, np.number): serializable_results[k][sk] = float(sv)
                else: serializable_results[k][sk] = sv
        json.dump(serializable_results, f, indent=4)

    plt.tight_layout()
    plt.savefig('final_benchmark_ladder.png')
    print(f"\n[Output] Graph saved to final_benchmark_ladder.png")

    # =================================================================
    # PERFORMANCE ANALYSIS: MULTI-BACKEND KERNEL EVALUATION
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
