import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
import sys
import time
import math
import warnings
import datetime
import os
import argparse
from collections import defaultdict
import copy

# =================================================================
# ARCHITECTURAL CONFIGURATION AND HYPERPARAMETER TUNING
# =================================================================
# Set IS_CLUSTER to True for A100/H100/L40S clusters to maximize throughput.
IS_CLUSTER = True 

# 1. Computational Batch Scaling (Linearization relative to VRAM availability)
B_SMALL   = 256 if IS_CLUSTER else 64   # For grid sizes <= 16
B_MEDIUM  = 128 if IS_CLUSTER else 16   # For grid sizes <= 28
B_LARGE   = 64  if IS_CLUSTER else 4    # For grid sizes <= 64
B_XL      = 16  if IS_CLUSTER else 2    # For grid sizes > 64 (e.g. 128)

# 2. Dataset Cardinality (Optimized for statistical significance)
S_SMALL  = 10000 if IS_CLUSTER else 3000 # Samples for small grids
S_LARGE  = 10000 if IS_CLUSTER else 6000 # Samples for large grids (>= 32)

# 3. Statistical Confidence Parameters
N_REPEATS_DEFAULT = 5 if IS_CLUSTER else 3 # Curriculum is more stable, so 5 is plenty

# 4. Hardware Acceleration Directives
ALLOW_TF32 = True # Enable TensorFloat-32 for NVIDIA Ampere/Hopper GPUs

# =================================================================
# OPTIMIZED GEOMETRIC KERNEL ARCHITECTURE (Cl(4,1))
# =================================================================

parser = argparse.ArgumentParser(description='Curriculum Systematic Sweep: Geometric vs. Standard Architectures')
parser.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 32, 64, 128], help='Sequence lengths for complexity sweep')
parser.add_argument('--d_vecs', type=int, nargs='+', default=[2, 4, 6, 8, 10, 12], help='Hidden vectors for capacity sweep')
parser.add_argument('--repeats', type=int, default=N_REPEATS_DEFAULT, help='Independent curriculum trials')
parser.add_argument('--epochs', type=int, default=40, help='Max training epochs per stage')
parser.add_argument('--outfile', type=str, default='curriculum_sweep_maxpool_results.json', help='Output file')

if ('ipykernel' in sys.modules or any('ipykernel' in arg for arg in sys.argv)) and not IS_CLUSTER:
    print("[INFO] Jupyter environment detected. Initializing with test parameters.")
    args = parser.parse_args(args=['--sizes', '8', '12', '16', '--repeats', '1', '--epochs', '30'])
else:
    if 'ipykernel' in sys.modules:
        print("[INFO] Cluster Mode active. Using full sweep parameters in Jupyter.")
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32 
torch.backends.cudnn.allow_tf32 = ALLOW_TF32
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU Unavailable"

def matthews_corrcoef(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0: return 0.0
    return numerator / denominator

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =================================================================
# 2. CONFORMAL GEOMETRIC ALGEBRA (Cl(4,1)) KERNEL
# =================================================================

GP_MAP_CACHE = {}
SIG_CACHE = {}

def compute_basis_product_cl41(a: int, b: int):
    sign, a_bits = 1.0, a
    for i in range(5):
        if (b >> i) & 1:
            for j in range(i + 1, 5):
                if (a_bits >> j) & 1: sign *= -1.0
            if (a_bits >> i) & 1:
                if i == 4: sign *= -1.0
                a_bits &= ~(1 << i)
            else: a_bits |= (1 << i)
    return sign, a_bits

def get_gp_map(device):
    idx = device.index if device.index is not None else 0
    if idx not in GP_MAP_CACHE:
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
            if (i >> 4) & 1: sig[i] *= -1.0
            grade = bin(i).count('1')
            if (grade * (grade - 1) // 2) % 2 == 1: sig[i] *= -1.0
        SIG_CACHE[idx] = sig
    return SIG_CACHE[idx]

def manifold_normalization(A: torch.Tensor, eps: float = 1e-5):
    sig = get_metric_signature(A.device)
    norm_sq = torch.sum(A * A * sig, dim=-1)
    denom = torch.max(
        torch.sqrt(torch.abs(norm_sq) + eps).unsqueeze(-1),
        torch.norm(A, p=2, dim=-1, keepdim=True) / 4.0 + eps
    ).clamp(min=1.0)
    return A / denom

def conformal_projection(grid: torch.Tensor):
    b, seq_len = grid.shape
    device = grid.device
    n_o = torch.zeros(32, device=device); n_o[16], n_o[8] = 0.5, -0.5
    n_inf = torch.zeros(32, device=device); n_inf[16], n_inf[8] = 1.0, 1.0
    out = torch.zeros((b, seq_len, 32), device=device)
    out += (grid == 1).unsqueeze(-1) * n_o 
    out += (grid == 0).unsqueeze(-1) * n_inf
    return out

def vector_where_max_norm(x):
    # x: (B, S, D, 32)
    norms = torch.norm(x, p=2, dim=-1) # (B, S, D)
    idx = torch.argmax(norms, dim=1)   # (B, D)
    
    # Gather along sequence dimension (1)
    idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 32)
    return torch.gather(x, 1, idx_exp).squeeze(1)

# =================================================================
# MODEL ARCHITECTURES: GEOMETRIC VS EUCLIDEAN BASELINES
# =================================================================

class VersorLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 32))
        with torch.no_grad():
            std = 0.5 / (in_features * 32)**0.5
            self.weight.normal_(0, std)

    def forward(self, x):
        gp = get_gp_map(x.device).view(32, 1024)
        
        # Construction of the optimized geometric linear operator
        W_op = torch.matmul(self.weight.view(-1, 32), gp).view(self.weight.shape[0], self.weight.shape[1], 32, 32)
        
        # Prepare W_real for F.linear: (O*32, I*32)
        # Permute (o, i, l, k) -> (o, k, i, l) then flatten
        W_real = W_op.permute(0, 3, 1, 2).reshape(self.weight.shape[0] * 32, self.weight.shape[1] * 32)
        
        # Efficient application
        x_flat = x.reshape(x.shape[:-2] + (-1,))
        out_flat = F.linear(x_flat, W_real)
        out = out_flat.view(x.shape[:-2] + (self.weight.shape[0], 32))
        
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
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, x):
        b, s, d, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        sig = get_metric_signature(x.device)
        q_flat = (q * sig).reshape(b, self.n_heads, s, -1)
        k_flat = k.reshape(b, self.n_heads, s, -1)
        score = torch.matmul(q_flat, k_flat.transpose(-1, -2))
        score = score / (self.d_head * 32)**0.5
        score = score * self.scale
        attn_weights = torch.softmax(score, dim=-1)
        
        # Optimize output projection: einsum -> matmul
        # Optimize output projection: einsum -> matmul
        v_flat = v.reshape(b, self.n_heads, s, -1) # (b, h, s, d_head*32)
        out_flat = torch.matmul(attn_weights, v_flat) # (b, h, s, d_head*32)
        out = out_flat.reshape(b, self.n_heads, s, self.d_head, 32)
        
        return self.o_proj(out.transpose(1, 2).reshape(b, s, d, 32))

class CGA_Transformer(nn.Module):
    def __init__(self, d_vectors=8, n_layers=4, seq_len=1024):
        super().__init__()
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

    def forward(self, x):
        x = x + self.pos_emb[:, :x.shape[1], :, :]
        for layer in self.layers:
            x = manifold_normalization(x + layer['attn'](x))
            x = manifold_normalization(x + layer['mlp'](x))
        pooled = vector_where_max_norm(self.pool(x))
        pooled = torch.tanh(pooled).view(x.shape[0], -1)
        return self.head(pooled)

class Standard_Transformer(nn.Module):
    def __init__(self, d_input, d_model=256, n_heads=4, n_layers=4, seq_len=1024):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x.max(dim=1)[0])

# =================================================================
# CURRICULUM LEARNING AND DATA GENERATION UTILITIES
# =================================================================

def transfer_weights(source_model, target_model):
    """Transfer weights between models, ignoring size-dependent positional embeddings."""
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    
    # Filter out positional embeddings (reset body, keep brain)
    pretrained_dict = {k: v for k, v in source_dict.items() if k in target_dict and 'pos_emb' not in k}
    
    target_dict.update(pretrained_dict)
    target_model.load_state_dict(target_dict)
    return target_model

def generate_dataset(size: int, n_samples: int, d_vectors: int = 8):
    # Offload large grids to CPU during generation to save VRAM
    target_dev = torch.device('cpu') 
    X_list, Y_list = [], []
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing='ij')
    coords_flat = np.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)
    coords_t = torch.tensor(coords_flat, dtype=torch.float32, device=target_dev).unsqueeze(0).repeat(n_samples, 1, 1)

    for _ in range(n_samples):
        grid = np.zeros((size, size), dtype=np.float32)
        path_coords = [(0,0)]; curr = (0,0); grid[0,0] = 1
        while curr != (size-1, size-1):
            cx, cy = curr
            moves = []
            if cx < size-1: moves.append((cx+1, cy))
            if cy < size-1: moves.append((cx, cy+1))
            if not moves: break
            next_step = moves[np.random.randint(len(moves))]
            path_coords.append(next_step); grid[next_step] = 1; curr = next_step
        label = 1 if (np.random.rand() > 0.5) else 0
        if label == 0 and len(path_coords) > 5:
            cut_idx = np.random.randint(2, len(path_coords)-2)
            grid[path_coords[cut_idx]] = 0 
        X_list.append(grid.flatten()); Y_list.append(label)
    
    X_t = torch.tensor(np.array(X_list), device=target_dev)
    Y_t = torch.tensor(np.array(Y_list), device=target_dev)
    X_proj = conformal_projection(X_t.to(DEVICE)).unsqueeze(2).repeat(1, 1, d_vectors, 1)
    return X_proj.cpu(), Y_t.cpu(), coords_t.cpu()

# =================================================================
# BENCHMARK EVALUATION ENGINE
# =================================================================

def train_one_config(model, size, d_vec, epochs=40, seed=42, model_type='cga'):
    # Scale epochs substantially for larger grids to ensure convergence
    if size == 16:    epochs = max(epochs, 60)
    elif size >= 32:  epochs = max(epochs, 100)
    
    torch.manual_seed(seed); np.random.seed(seed)
    if size < 32: n_samples = S_SMALL
    else: n_samples = S_LARGE
    X, Y, Coords = generate_dataset(size, n_samples, d_vec)
    dataset = TensorDataset(X, Y, Coords)
    train_len = int(0.9 * len(dataset))
    train, val = random_split(dataset, [train_len, len(dataset)-train_len])
    
    if size <= 16:   batch_size = B_SMALL
    elif size <= 28: batch_size = B_MEDIUM
    elif size <= 64: batch_size = B_LARGE
    else:            batch_size = B_XL
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    
    model.to(DEVICE)
    lr = 0.001 if size <= 16 else 0.0005
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine Scheduler with Warmup
    warmup_iters = 5
    def lr_lambda(epoch):
        if epoch < warmup_iters: return (epoch + 1) / warmup_iters
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_iters) / (epochs - warmup_iters)))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    crit = nn.CrossEntropyLoss()
    
    # Use BF16 on H100 for better stability
    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=not use_bf16)
    
    best_mcc, best_model_state = -1.0, None
    
    for ep in range(epochs):
        model.train()
        for xb, yb, cb in train_loader:
            xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                if model_type == 'std': out = model(xb)
                else: out = model(xb)
                loss = crit(out, yb)
            
            if use_bf16:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
        
        sched.step()
        
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for xb, yb, cb in val_loader:
                xb, yb, cb = xb.to(DEVICE), yb.to(DEVICE), cb.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    if model_type == 'std': o = model(xb)
                    else: o = model(xb)
                preds.extend(o.argmax(1).cpu().numpy()); trues.extend(yb.cpu().numpy())
        mcc = matthews_corrcoef(trues, preds)
        print(f"      [Epoch {ep+1:2d}/{epochs}] MCC: {mcc:.3f}")
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_model_state = copy.deepcopy(model.state_dict())
        
            
        if mcc > 0.999: 
            print(f"      [Epoch {ep+1:2d}/{epochs}] Converged! (MCC: {mcc:.3f})")
            break
        
    model.load_state_dict(best_model_state)
    del X, Y, Coords, dataset, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return best_mcc

def find_matched_std_dim(target_params, d_input, n_layers=4, seq_len=1024):
    for d in range(16, 2048, 16):
        test_m = Standard_Transformer(d_input, d_model=d, n_layers=n_layers, seq_len=seq_len)
        if count_parameters(test_m) > target_params: return d
    return 2048

def save_json(data, filename):
    with open(filename + '.tmp', 'w') as f: json.dump(data, f, indent=4)
    os.replace(filename + '.tmp', filename)

def run_sweeps():
    print(f"Starting Curriculum Sweeps on {GPU_NAME}")
    data_results, model_results = defaultdict(list), defaultdict(list)
    if os.path.exists(args.outfile):
        try:
            with open(args.outfile, 'r') as f:
                saved = json.load(f); data_results.update(saved.get('data_sweep', {})); model_results.update(saved.get('model_sweep', {}))
        except: pass

    # --- 1. SEQUENCE COMPLEXITY SWEEP (WITH CURRICULUM) ---
    SIZES, D_VEC_FIXED, N_REPEATS, EPOCHS = args.sizes, 8, args.repeats, args.epochs
    print(f"\n--- CURRICULUM COMPLEXITY SWEEP (D_VEC={D_VEC_FIXED}) ---")
    
    for r in range(N_REPEATS):
        print(f"\n--- Trial {r+1}/{N_REPEATS} ---")
        prev_cga, prev_std = None, None
        for size in SIZES:
            # Check if trial r is already recorded for this size
            if str(size) in data_results and len(data_results[str(size)]) > r:
                # We need to reconstruct the models to have something to transfer from
                # But typically curriculum trials depend on previous results.
                # For simplicity, if we resume, we re-run the full trial chain or just skip.
                continue
            
            seq_len = size * size
            print(f"  Grid: {size}x{size}")
            
            cga_m = CGA_Transformer(D_VEC_FIXED, 4, seq_len)
            if prev_cga: cga_m = transfer_weights(prev_cga, cga_m)
            cga_params = count_parameters(cga_m)
            std_d = find_matched_std_dim(cga_params, D_VEC_FIXED*32, 4, seq_len)
            
            # CGA Training
            c_mcc = train_one_config(cga_m, size, D_VEC_FIXED, epochs=EPOCHS, seed=100+r, model_type='cga')
            prev_cga = copy.deepcopy(cga_m)
            
            # Standard Training (Fresh for baseline fairness, but matched params)
            std_m = Standard_Transformer(D_VEC_FIXED*32, std_d, 4, 4, seq_len)
            s_mcc = train_one_config(std_m, size, D_VEC_FIXED, epochs=EPOCHS, seed=100+r, model_type='std')
            
            data_results[str(size)].append({'cga': c_mcc, 'std': s_mcc, 'std_dim': std_d})
            print(f"    MCC: CGA={c_mcc:.3f}, Standard={s_mcc:.3f} (Dim={std_d})")
            save_json({'data_sweep': data_results, 'model_sweep': model_results}, args.outfile)

    # --- 2. CAPACITY SWEEP (Constant Complexity, No Curriculum needed) ---
    FIXED_SIZE, D_VECS = 16, args.d_vecs
    print(f"\n--- CAPACITY SWEEP (SIZE={FIXED_SIZE}x{FIXED_SIZE}) ---")
    for dv in D_VECS:
        if str(dv) in model_results and len(model_results[str(dv)]) >= N_REPEATS: continue
        seq_len = FIXED_SIZE * FIXED_SIZE
        print(f"  D_VEC: {dv}")
        for r in range(N_REPEATS):
            cga_m = CGA_Transformer(dv, 4, seq_len)
            cga_params = count_parameters(cga_m)
            std_d = find_matched_std_dim(cga_params, dv*32, 4, seq_len)
            c_mcc = train_one_config(cga_m, FIXED_SIZE, dv, epochs=EPOCHS, seed=200+r, model_type='cga')
            std_m = Standard_Transformer(dv*32, std_d, 4, 4, seq_len)
            s_mcc = train_one_config(std_m, FIXED_SIZE, dv, epochs=EPOCHS, seed=200+r, model_type='std')
            model_results[str(dv)].append({'cga': c_mcc, 'std': s_mcc, 'std_dim': std_d})
            print(f"    Trial {r+1}: CGA={c_mcc:.3f}, Std={s_mcc:.3f}")
            save_json({'data_sweep': data_results, 'model_sweep': model_results}, args.outfile)

    plot_results(data_results, model_results)

def plot_results(data_results, model_results):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    x_grid = sorted([int(k) for k in data_results.keys()])
    cga_means = [np.mean([d['cga'] for d in data_results[str(s)]]) for s in x_grid]
    std_means = [np.mean([d['std'] for d in data_results[str(s)]]) for s in x_grid]
    plt.errorbar(x_grid, cga_means, label='Geometric (Curriculum)', fmt='-o', linewidth=2)
    plt.errorbar(x_grid, std_means, label='Standard (Baseline)', fmt='-s', alpha=0.7)
    plt.xlabel('Grid Size'); plt.ylabel('MCC'); plt.title('Robustness (Curriculum)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    grid_ratios = [np.mean([d['cga']/(max(d['std'], 0.01)) for d in data_results[str(s)]]) for s in x_grid]
    plt.plot(x_grid, grid_ratios, '-o', color='forestgreen')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xlabel('Grid Size'); plt.ylabel('Advantage Ratio'); plt.title('Scaling Advantage')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    x_model = sorted([int(k) for k in model_results.keys()])
    model_ratios = [np.mean([d['cga']/(max(d['std'], 0.01)) for d in model_results[str(dv)]]) for dv in x_model]
    plt.plot(x_model, model_ratios, '-o', color='purple')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xlabel('D_VEC'); plt.ylabel('Efficiency Ratio'); plt.title('Advantage vs. Params')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.savefig('curriculum_sweep_maxpool_plots.png')
    print("\n[INFO] Curriculum visualizations saved to: curriculum_sweep_maxpool_plots.png")

if __name__ == "__main__":
    run_sweeps()
