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

# 1. Batch Size Scaling
B_SMALL   = 256 if IS_CLUSTER else 64   # Len <= 32
B_MEDIUM  = 128 if IS_CLUSTER else 16   # Len <= 64
B_LARGE   = 64  if IS_CLUSTER else 4    # Len <= 128
B_XL      = 16  if IS_CLUSTER else 2    # Len > 128

# 2. Dataset Capacity
S_SMALL  = 10000 if IS_CLUSTER else 3000
S_LARGE  = 10000 if IS_CLUSTER else 6000

# 3. Statistical Significance
N_REPEATS_DEFAULT = 5 if IS_CLUSTER else 3

# 4. Hardware Speedup Flags
ALLOW_TF32 = True

# =================================================================
# EXPERIMENTAL CONFIGURATION AND INTERFACE
# =================================================================

parser = argparse.ArgumentParser(description='Dyck-N Systematic Sweep: Geometric vs. Standard')
parser.add_argument('--sizes', type=int, nargs='+', default=[32, 64, 128, 256], help='Sequence lengths')
parser.add_argument('--d_vecs', type=int, nargs='+', default=[2, 4, 6, 8], help='Hidden vectors for capacity sweep')
parser.add_argument('--repeats', type=int, default=N_REPEATS_DEFAULT, help='Independent curriculum trials')
parser.add_argument('--epochs', type=int, default=40, help='Max training epochs per stage')
parser.add_argument('--outfile', type=str, default='dyck_sweep_results.json', help='Output file')

if ('ipykernel' in sys.modules or any('ipykernel' in arg for arg in sys.argv)) and not IS_CLUSTER:
    print("[INFO] Jupyter environment detected. Initializing with test parameters.")
    args = parser.parse_args(args=['--sizes', '32', '48', '--repeats', '1', '--epochs', '20'])
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
# CONFORMAL GEOMETRIC ALGEBRA KERNEL (Cl(4,1))
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
        gp = get_gp_map(x.device)
        W_op = torch.einsum('oij,jlk->oilk', self.weight, gp)
        if x.dim() == 3: out = torch.einsum('bil,oilk->bok', x, W_op)
        else: out = torch.einsum('bsil,oilk->bsok', x, W_op)
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
        out = torch.einsum('bhsi,bhidl->bhsdl', attn_weights, v)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, d, 32))

class CGA_Transformer(nn.Module):
    def __init__(self, vocab_size=5, d_vectors=8, n_layers=4, seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_vectors * 32)
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
        b, s = x.shape
        x = self.embed(x).view(b, s, -1, 32)
        x = x + self.pos_emb[:, :s, :, :]
        for layer in self.layers:
            x = manifold_normalization(x + layer['attn'](x))
            x = manifold_normalization(x + layer['mlp'](x))
        pooled = self.pool(x).mean(dim=1) 
        pooled = torch.tanh(pooled).view(x.shape[0], -1)
        return self.head(pooled)

class Standard_Transformer(nn.Module):
    def __init__(self, vocab_size=5, d_input=None, d_model=256, n_heads=4, n_layers=4, seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.embed(x) + self.pos_emb[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x.mean(dim=1))

# =================================================================
# CURRICULUM LEARNING AND DATA SYNTHESIS UTILITIES
# =================================================================

def transfer_weights(source_model, target_model):
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    # Filter out positional embeddings
    pretrained_dict = {k: v for k, v in source_dict.items() if k in target_dict and 'pos_emb' not in k}
    target_dict.update(pretrained_dict)
    target_model.load_state_dict(target_dict)
    return target_model

def generate_dataset(size: int, n_samples: int, vocab_size=5):
    target_dev = torch.device('cpu') 
    X_list, Y_list = [], []
    PAIRS = {1: 2, 3: 4}
    OPENERS = [1, 3]

    for _ in range(n_samples):
        seq = []; stack = []
        current_len = 0
        while current_len < size:
            remaining = size - current_len
            can_push = True
            must_pop = (len(stack) == remaining)
            
            if must_pop: op = 'pop'
            elif len(stack) == 0: op = 'push'
            else: op = 'push' if np.random.rand() > 0.5 else 'pop'
                
            if op == 'push':
                char = np.random.choice(OPENERS)
                seq.append(char); stack.append(char)
            else:
                seq.append(PAIRS[stack.pop()])
            current_len += 1
            
        label = 1 if (np.random.rand() > 0.5) else 0
        if label == 0:
            idx = np.random.randint(0, size)
            orig = seq[idx]
            options = [c for c in [1, 2, 3, 4] if c != orig]
            seq[idx] = np.random.choice(options)
            
        X_list.append(seq)
        Y_list.append(label)
    
    X_t = torch.tensor(np.array(X_list, dtype=np.int64), device=target_dev)
    Y_t = torch.tensor(np.array(Y_list, dtype=np.int64), device=target_dev)
    return X_t.to(DEVICE), Y_t.to(DEVICE)

# =================================================================
# BENCHMARK EVALUATION ENGINE
# =================================================================

def train_one_config(model, size, epochs=40, seed=42, model_type='cga'):
    if size >= 128:  epochs = max(epochs, 80)
    
    torch.manual_seed(seed); np.random.seed(seed)
    n_samples = S_SMALL if size < 64 else S_LARGE
    X, Y = generate_dataset(size, n_samples)
    dataset = TensorDataset(X, Y)
    train_len = int(0.9 * len(dataset))
    train, val = random_split(dataset, [train_len, len(dataset)-train_len])
    
    if size <= 32:   batch_size = B_SMALL
    elif size <= 64: batch_size = B_MEDIUM
    elif size <= 128: batch_size = B_LARGE
    else:            batch_size = B_XL
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    
    model.to(DEVICE)
    lr = 0.001 if size <= 32 else 0.0005
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
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_model_state = copy.deepcopy(model.state_dict())
            
        if mcc > 0.999: 
            print(f"      [Epoch {ep+1:2d}/{epochs}] Converged! (MCC: {mcc:.3f})")
            break
        
    model.load_state_dict(best_model_state)
    del X, Y, dataset, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return best_mcc

def find_matched_std_dim(target_params, n_layers=4, seq_len=1024, vocab_size=5):
    for d in range(16, 2048, 16):
        test_m = Standard_Transformer(vocab_size=vocab_size, d_model=d, n_layers=n_layers, seq_len=seq_len)
        if count_parameters(test_m) > target_params: return d
    return 2048

def save_json(data, filename):
    with open(filename + '.tmp', 'w') as f: json.dump(data, f, indent=4)
    os.replace(filename + '.tmp', filename)

def run_sweeps():
    print(f"Starting DyckN Curriculum Sweeps on {GPU_NAME}")
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
        prev_cga = None
        for size in SIZES:
            if str(size) in data_results and len(data_results[str(size)]) > r:
                continue
            
            print(f"  Length: {size}")
            cga_m = CGA_Transformer(vocab_size=5, d_vectors=D_VEC_FIXED, n_layers=4, seq_len=size)
            if prev_cga: cga_m = transfer_weights(prev_cga, cga_m)
            
            cga_params = count_parameters(cga_m)
            std_d = find_matched_std_dim(cga_params, 4, size, 5)
            
            # CGA Training
            c_mcc = train_one_config(cga_m, size, epochs=EPOCHS, seed=100+r, model_type='cga')
            prev_cga = copy.deepcopy(cga_m)
            
            # Standard Training
            std_m = Standard_Transformer(vocab_size=5, d_model=std_d, n_layers=4, seq_len=size)
            s_mcc = train_one_config(std_m, size, epochs=EPOCHS, seed=100+r, model_type='std')
            
            data_results[str(size)].append({'cga': c_mcc, 'std': s_mcc, 'std_dim': std_d})
            print(f"    MCC: CGA={c_mcc:.3f}, Standard={s_mcc:.3f} (Dim={std_d})")
            save_json({'data_sweep': data_results, 'model_sweep': model_results}, args.outfile)

    # --- 2. CAPACITY SWEEP ---
    FIXED_SIZE, D_VECS = 64, args.d_vecs
    print(f"\n--- CAPACITY SWEEP (SIZE={FIXED_SIZE}) ---")
    for dv in D_VECS:
        if str(dv) in model_results and len(model_results[str(dv)]) >= N_REPEATS: continue
        print(f"  D_VEC: {dv}")
        for r in range(N_REPEATS):
            cga_m = CGA_Transformer(vocab_size=5, d_vectors=dv, n_layers=4, seq_len=FIXED_SIZE)
            cga_params = count_parameters(cga_m)
            std_d = find_matched_std_dim(cga_params, 4, FIXED_SIZE, 5)
            
            c_mcc = train_one_config(cga_m, FIXED_SIZE, epochs=EPOCHS, seed=200+r, model_type='cga')
            std_m = Standard_Transformer(vocab_size=5, d_model=std_d, n_layers=4, seq_len=FIXED_SIZE)
            s_mcc = train_one_config(std_m, FIXED_SIZE, epochs=EPOCHS, seed=200+r, model_type='std')
            
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
    plt.xlabel('Sequence Length'); plt.ylabel('MCC'); plt.title('Robustness (Curriculum)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    # Avoid div by zero
    grid_ratios = [np.mean([d['cga']/(max(d['std'], 0.01)) for d in data_results[str(s)]]) for s in x_grid]
    plt.plot(x_grid, grid_ratios, '-o', color='forestgreen')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xlabel('Sequence Length'); plt.ylabel('Advantage Ratio'); plt.title('Scaling Advantage')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    x_model = sorted([int(k) for k in model_results.keys()])
    model_ratios = [np.mean([d['cga']/(max(d['std'], 0.01)) for d in model_results[str(dv)]]) for dv in x_model]
    plt.plot(x_model, model_ratios, '-o', color='purple')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xlabel('D_VEC'); plt.ylabel('Efficiency Ratio'); plt.title('Advantage vs. Params')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.savefig('dyck_sweep_plots.png')
    print("\n[INFO] Sweep visualizations saved to: dyck_sweep_plots.png")

if __name__ == "__main__":
    run_sweeps()
