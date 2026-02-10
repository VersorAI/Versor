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

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from sklearn.metrics import matthews_corrcoef
except ImportError:
    def matthews_corrcoef(y_true, y_pred):
        y_true = np.array(y_true); y_pred = np.array(y_pred)
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
    print(f"BENCHMARK REPORT: DYCK-N TOPOLOGICAL ESTIMATION")
    print(f"{'='*80}")
    print(f"Date:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware:     {GPU_NAME}")
    print(f"Task:         Dyck-2 Language (Independent Runs)")
    print(f"{'='*80}\n")

# =================================================================
# 1. ALGEBRA KERNEL (Cl(4,1))
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

def manifold_normalization(A: torch.Tensor, eps: float = 1e-6):
    sig = get_metric_signature(A.device)
    norm_sq = torch.sum(A * A * sig, dim=-1)
    denom = torch.max(
        torch.sqrt(torch.abs(norm_sq) + eps).unsqueeze(-1),
        torch.norm(A, p=2, dim=-1, keepdim=True) / 4.0 + eps
    ).clamp(min=1.0)
    return A / denom

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
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, x):
        b, s, d, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.d_head, 32).transpose(1, 2)
        
        sig = get_metric_signature(x.device)
        q_flat = (q * sig).flatten(3) 
        k_flat = k.flatten(3)
        score = torch.matmul(q_flat, k_flat.transpose(-1, -2))
        score = score / (self.d_head * 32)**0.5
        score = score * self.scale
        attn_weights = torch.softmax(score, dim=-1)
        out = torch.einsum('bhsi,bhidl->bhsdl', attn_weights, v)
        return self.o_proj(out.transpose(1, 2).reshape(b, s, d, 32))

class CGA_Transformer(nn.Module):
    def __init__(self, vocab_size=5, d_vectors=4, n_layers=4, seq_len=1024):
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
    def __init__(self, vocab_size=5, d_model=128, n_heads=4, n_layers=4, seq_len=1024):
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

def generate_dataset(size: int, n_samples: int):
    # print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Data] Gen {n_samples} (Len {size})...", end=" ", flush=True)
    target_dev = torch.device('cpu') if size >= 256 else DEVICE
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
    return X_t, Y_t

def run_traininv_c_versorycle(model_name: str, model: nn.Module, size: int):
    n_samples = 4000 if size < 64 else 8000
    X, Y = generate_dataset(size, n_samples)
    dataset = TensorDataset(X, Y)
    train_len = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_len, len(dataset)-train_len])

    if size <= 32:   BATCH_SIZE, ACCUM_STEPS = 64, 1
    elif size == 64: BATCH_SIZE, ACCUM_STEPS = 32, 2
    else:            BATCH_SIZE, ACCUM_STEPS = 16, 4

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model.to(DEVICE)
    lr = 0.001 if size <= 32 else 0.0005
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    criterion = nn.CrossEntropyLoss()
    use_amp = "Standard" in model_name
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Train] {model_name} | B:{BATCH_SIZE}x{ACCUM_STEPS}", end="")
    history_mcc = []
    
    for epoch in range(25):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(xb)
                loss = criterion(out, yb) / ACCUM_STEPS
            if use_amp:
                scaler.scale(loss).backward()
                if (i+1)%ACCUM_STEPS==0: scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if (i+1)%ACCUM_STEPS==0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad()
        scheduler.step()
        
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=use_amp): o = model(xb)
                preds.extend(o.argmax(1).cpu().numpy()); trues.extend(yb.cpu().numpy())
        mcc = matthews_corrcoef(trues, preds)
        history_mcc.append(mcc)
        if epoch % 5 == 0: print(".", end="", flush=True)

    print(f" Final MCC: {mcc:.3f}")
    del X, Y, dataset, train_loader, val_loader; gc.collect(); torch.cuda.empty_cache()
    return history_mcc, mcc

if __name__ == "__main__":
    print_header()
    SIZES = [32, 64, 128]
    VOCAB_SIZE = 5
    results_summary = {}

    print(f"{'Length':<12} | {'Geometric (MCC)':<18} | {'Standard (MCC)':<18} | {'Delta':<10}")
    print("-" * 65)

    for size in SIZES:
        # GEOMETRIC
        d_vec = 8 if size >= 64 else 4
        model_versor = CGA_Transformer(vocab_size=VOCAB_SIZE, d_vectors=d_vec, n_layers=4, seq_len=size)
        hist_geo, mcc_geo = run_traininv_c_versorycle(f"CGA ({size})", model_versor, size)
        
        # STANDARD
        d_model = 256 if size >= 64 else 128
        model_std = Standard_Transformer(vocab_size=VOCAB_SIZE, d_model=d_model, seq_len=size)
        hist_std, mcc_std = run_traininv_c_versorycle(f"Std ({size})", model_std, size)
        
        delta = mcc_geo - mcc_std
        print(f"\r{size:<12} | {mcc_geo:.3f}              | {mcc_std:.3f}              | {delta:+.3f}")
        results_summary[size] = {'versor': hist_geo, 'std': hist_std}
        print("-" * 65)

    if sns: sns.set_theme(style="whitegrid", context="paper")
    else: plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    for size in SIZES:
        plt.plot(results_summary[size]['versor'], label=f"CGA ({size})", alpha=0.8)
        plt.plot(results_summary[size]['std'], label=f"Std ({size})", linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig("dyck_benchmark_scratch.png", dpi=300)
    print(f"[Output] Saved dyck_benchmark_scratch.png")
