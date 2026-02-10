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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# =================================================================
# 0. SYSTEM PROFILES & TUNING
# =================================================================
IS_CLUSTER = True 

# Batch sizes for H200 or similar high-VRAM hardware
B_SMALL   = 2048  # For sizes 8, 16
B_MEDIUM  = 512   # For size 32
B_LARGE   = 128   # For size 64 (Seq len 4096)
B_XL      = 48    # For size 128 (Seq len 16384)

S_SMALL   = 3000
S_LARGE   = 1500

N_REPEATS_DEFAULT = 1
ALLOW_TF32 = True 

# Global for periodic saving
LAST_SAVE_TIME = time.time()

# =================================================================
# 1. ARGUMENT PARSING
# =================================================================

parser = argparse.ArgumentParser(description='MazeBench: Parallel Data Gen & AI Training Sweep')
parser.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 32, 64, 128], help='Sequence lengths for complexity sweep')
parser.add_argument('--d_vecs', type=int, nargs='+', default=[2, 4, 6, 8, 10, 12], help='Hidden vectors for capacity sweep')
parser.add_argument('--repeats', type=int, default=N_REPEATS_DEFAULT, help='Independent curriculum trials')
parser.add_argument('--epochs', type=int, default=40, help='Max training epochs per stage')
parser.add_argument('--outfile', type=str, default='mazebench_results.json', help='Output file')
parser.add_argument('--data_dir', type=str, default='data_pregen', help='Directory for data storage')
parser.add_argument('--workers', type=int, default=None, help='Workers for data generation (default: all cores)')

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

def conformal_projection_cpu(grid):
    # grid shape: (size*size)
    n_o = np.zeros(32); n_o[16], n_o[8] = 0.5, -0.5
    n_inf = np.zeros(32); n_inf[16], n_inf[8] = 1.0, 1.0
    out = np.zeros((len(grid), 32))
    out[grid == 1] = n_o
    out[grid == 0] = n_inf
    return out

def vector_where_max_norm(x):
    norms = torch.norm(x, p=2, dim=-1)
    idx = torch.argmax(norms, dim=1)   
    idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, 32)
    return torch.gather(x, 1, idx_exp).squeeze(1)

# =================================================================
# 3. NEURAL NETWORK ARCHITECTURES
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
        W_op = torch.matmul(self.weight.view(-1, 32), gp).view(self.weight.shape[0], self.weight.shape[1], 32, 32)
        W_real = W_op.permute(0, 3, 1, 2).reshape(self.weight.shape[0] * 32, self.weight.shape[1] * 32)
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
        v_flat = v.reshape(b, self.n_heads, s, -1) 
        out_flat = torch.matmul(attn_weights, v_flat) 
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
# 4. DATA GENERATION ENGINE (Parallel Background)
# =================================================================

def generate_single_maze(size):
    np.random.seed()
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
    
    projected = conformal_projection_cpu(grid.flatten())
    return projected.astype(np.float32), label

def safe_save(data, filename):
    new_file = filename + ".new"
    torch.save(data, new_file)
    if os.path.exists(filename):
        os.remove(filename)
    os.rename(new_file, filename)

def generate_batch(size, n_samples, workers=None, filename=None):
    if workers is None:
        workers = os.cpu_count()
    
    print(f"\n[DataGen] Generating {n_samples} samples for {size}x{size} using {workers} cores...")
    results = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_single_maze, size) for _ in range(n_samples)]
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i+1) % 500 == 0 or (i+1) == n_samples:
                print(f"\r  Progress: {i+1}/{n_samples} samples ({(i+1)/n_samples*100:.1f}%)", end="", flush=True)
    
    X = torch.from_numpy(np.stack([r[0] for r in results])).float()
    Y = torch.tensor([r[1] for r in results]).long()
    X = X.unsqueeze(2) # (N, Seq, 1, 32)
    
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        safe_save({'X': X, 'Y': Y}, filename)
    return X, Y

class BackgroundDataManager:
    def __init__(self, data_dir, workers):
        self.data_dir = data_dir
        self.workers = workers
        self.queue = []
        self.lock = threading.Lock()
        self.running = False

    def add_task(self, size, samples):
        filename = f"{self.data_dir}/maze_{size}_{samples}.pt"
        if not os.path.exists(filename):
            with self.lock:
                self.queue.append((size, samples, filename))
            if not self.running:
                self.start()

    def start(self):
        self.running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while True:
            task = None
            with self.lock:
                if self.queue:
                    task = self.queue.pop(0)
            
            if task:
                size, samples, filename = task
                generate_batch(size, samples, self.workers, filename)
            else:
                self.running = False
                break

    def ensure_ready(self, size, samples):
        filename = f"{self.data_dir}/maze_{size}_{samples}.pt"
        while not os.path.exists(filename):
            print(f"  [Waiting] Data for {size}x{size} not ready yet... checking again in 5s")
            time.sleep(5)
        return filename

# =================================================================
# 5. UTILITIES & TRAINING ENGINE
# =================================================================

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

def transfer_weights(source_model, target_model):
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    pretrained_dict = {k: v for k, v in source_dict.items() if k in target_dict and 'pos_emb' not in k}
    target_dict.update(pretrained_dict)
    target_model.load_state_dict(target_dict)
    return target_model

def load_pregenerated_data(size, n_samples, d_vec, data_manager):
    filename = data_manager.ensure_ready(size, n_samples)
    print(f"[INFO] Loading data from {filename}")
    data = torch.load(filename, weights_only=True)
    X, Y = data['X'], data['Y']
    
    if X.shape[2] != d_vec:
        X = X[:, :, :1, :].repeat(1, 1, d_vec, 1)
    return X, Y

def train_one_config(model, size, d_vec, epochs, seed, data_manager, save_callback=None):
    if size == 16:    epochs = max(epochs, 60)
    elif size >= 32:  epochs = max(epochs, 100)
    
    torch.manual_seed(seed); np.random.seed(seed)
    n_samples = S_SMALL if size < 32 else S_LARGE
    
    X, Y = load_pregenerated_data(size, n_samples, d_vec, data_manager)
    
    dataset = TensorDataset(X, Y)
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
    
    warmup_iters = 5
    def lr_lambda(epoch):
        if epoch < warmup_iters: return (epoch + 1) / warmup_iters
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_iters) / (epochs - warmup_iters)))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    crit = nn.CrossEntropyLoss()
    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=not use_bf16)
    
    best_mcc, best_model_state = -1.0, None
    
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                out = model(xb)
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
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    o = model(xb)
                preds.extend(o.argmax(1).cpu().numpy()); trues.extend(yb.cpu().numpy())
        mcc = matthews_corrcoef(trues, preds)
        print(f"      [Epoch {ep+1:2d}/{epochs}] MCC: {mcc:.3f}")
        
        if save_callback: save_callback()
        if mcc > best_mcc:
            best_mcc = mcc
            best_model_state = copy.deepcopy(model.state_dict())
        if mcc > 0.999: break
        
    model.load_state_dict(best_model_state)
    del X, Y, dataset, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return best_mcc

def find_matched_std_dim(target_params, d_input, n_layers=4, seq_len=1024):
    for d in range(16, 2048, 16):
        test_m = Standard_Transformer(d_input, d_model=d, n_layers=n_layers, seq_len=seq_len)
        if count_parameters(test_m) > target_params: return d
    return 2048

def save_json(data, filename):
    new_file = filename + '.new'
    try:
        with open(new_file, 'w') as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        time.sleep(2) # User requested delay
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(new_file, filename)
    except Exception as e:
        print(f"[ERROR] Periodic save failed: {e}")

def get_periodic_save_fn(data_results, model_results, filename):
    def periodic_save():
        global LAST_SAVE_TIME
        if time.time() - LAST_SAVE_TIME > 60:
            print(f"      [INFO] Periodic save triggered")
            save_json({'data_sweep': data_results, 'model_sweep': model_results}, filename)
            LAST_SAVE_TIME = time.time()
    return periodic_save

# =================================================================
# 6. RUN SWEEPS
# =================================================================

def run_sweeps():
    print(f"Starting MazeBench on {GPU_NAME}")
    data_results, model_results = defaultdict(list), defaultdict(list)
    if os.path.exists(args.outfile):
        try:
            with open(args.outfile, 'r') as f:
                saved = json.load(f); data_results.update(saved.get('data_sweep', {})); model_results.update(saved.get('model_sweep', {}))
        except: pass

    data_manager = BackgroundDataManager(args.data_dir, args.workers)
    
    # 1. Add all complexity sweep tasks
    for size in args.sizes:
        samples = S_SMALL if size < 32 else S_LARGE
        data_manager.add_task(size, samples)
    
    # 2. Add capacity sweep tasks (fixed size 16)
    data_manager.add_task(16, S_SMALL)

    SIZES, D_VEC_FIXED, N_REPEATS, EPOCHS = args.sizes, 8, args.repeats, args.epochs
    save_fn = get_periodic_save_fn(data_results, model_results, args.outfile)
    
    print(f"\n--- COMPLEXITY SWEEP (D_VEC={D_VEC_FIXED}) ---")
    for r in range(N_REPEATS):
        print(f"\n--- Trial {r+1}/{N_REPEATS} ---")
        prev_cga = None
        for size in SIZES:
            if str(size) in data_results and len(data_results[str(size)]) > r:
                continue
            
            seq_len = size * size
            print(f"  Grid: {size}x{size}")
            
            cga_m = CGA_Transformer(D_VEC_FIXED, 4, seq_len)
            if prev_cga: cga_m = transfer_weights(prev_cga, cga_m)
            cga_params = count_parameters(cga_m)
            std_d = find_matched_std_dim(cga_params, D_VEC_FIXED*32, 4, seq_len)
            
            c_mcc = train_one_config(cga_m, size, D_VEC_FIXED, EPOCHS, 100+r, data_manager, save_fn)
            prev_cga = copy.deepcopy(cga_m)
            
            std_m = Standard_Transformer(D_VEC_FIXED*32, std_d, 4, 4, seq_len)
            s_mcc = train_one_config(std_m, size, D_VEC_FIXED, EPOCHS, 100+r, data_manager, save_fn)
            
            data_results[str(size)].append({'cga': c_mcc, 'std': s_mcc, 'std_dim': std_d})
            print(f"    MCC: CGA={c_mcc:.3f}, Std={s_mcc:.3f} (Dim={std_d})")
            save_json({'data_sweep': data_results, 'model_sweep': model_results}, args.outfile)

    FIXED_SIZE = 16
    print(f"\n--- CAPACITY SWEEP (SIZE={FIXED_SIZE}x{FIXED_SIZE}) ---")
    for dv in args.d_vecs:
        if str(dv) in model_results and len(model_results[str(dv)]) >= N_REPEATS: continue
        seq_len = FIXED_SIZE * FIXED_SIZE
        print(f"  D_VEC: {dv}")
        for r in range(N_REPEATS):
            cga_m = CGA_Transformer(dv, 4, seq_len)
            cga_params = count_parameters(cga_m)
            std_d = find_matched_std_dim(cga_params, dv*32, 4, seq_len)
            c_mcc = train_one_config(cga_m, FIXED_SIZE, dv, EPOCHS, 200+r, data_manager, save_fn)
            std_m = Standard_Transformer(dv*32, std_d, 4, 4, seq_len)
            s_mcc = train_one_config(std_m, FIXED_SIZE, dv, EPOCHS, 200+r, data_manager, save_fn)
            model_results[str(dv)].append({'cga': c_mcc, 'std': s_mcc, 'std_dim': std_d})
            print(f"    Trial {r+1}: CGA={c_mcc:.3f}, Std={s_mcc:.3f}")
            save_json({'data_sweep': data_results, 'model_sweep': model_results}, args.outfile)

    plot_results(data_results, model_results)

def plot_results(data_results, model_results):
    if not data_results and not model_results: return
    plt.figure(figsize=(18, 5))
    if data_results:
        plt.subplot(1, 3, 1)
        x_grid = sorted([int(k) for k in data_results.keys()])
        cga_means = [np.mean([d['cga'] for d in data_results[str(s)]]) for s in x_grid]
        std_means = [np.mean([d['std'] for d in data_results[str(s)]]) for s in x_grid]
        plt.errorbar(x_grid, cga_means, label='Geometric (CGA)', fmt='-o', linewidth=2)
        plt.errorbar(x_grid, std_means, label='Standard (Baseline)', fmt='-s', alpha=0.7)
        plt.xlabel('Grid Size'); plt.ylabel('MCC'); plt.title('Robustness Sweep')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        grid_ratios = [np.mean([d['cga']/(max(d['std'], 0.01)) for d in data_results[str(s)]]) for s in x_grid]
        plt.plot(x_grid, grid_ratios, '-o', color='forestgreen')
        plt.axhline(1.0, color='gray', linestyle='--')
        plt.xlabel('Grid Size'); plt.ylabel('Advantage Ratio'); plt.title('Scaling Advantage')
        plt.grid(True, alpha=0.3)

    if model_results:
        plt.subplot(1, 3, 3)
        x_model = sorted([int(k) for k in model_results.keys()])
        model_ratios = [np.mean([d['cga']/(max(d['std'], 0.01)) for d in model_results[str(dv)]]) for dv in x_model]
        plt.plot(x_model, model_ratios, '-o', color='purple')
        plt.axhline(1.0, color='gray', linestyle='--')
        plt.xlabel('D_VEC'); plt.ylabel('Efficiency Ratio'); plt.title('Advantage vs. Capacity')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.savefig('mazebench_plots.png')
    print("\n[INFO] Visualizations saved to: mazebench_plots.png")

if __name__ == "__main__":
    run_sweeps()
