import os
import sys
import json
import time
import math
import random
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# Add parent directory to path to import Model
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Try imports
try:
    from Model.model import VersorTransformer
except ImportError:
    # Fallback if running from a different directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Model.model import VersorTransformer

# Constants
RESULTS_DIR = "results"
DATA_DIR = "data"
SEEDS = [42, 43, 44, 45, 46]
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# -----------------------------------------------------------------------------
# 1. NLP Task: WikiText-2 (Character Level)
# -----------------------------------------------------------------------------

class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_len=64):
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.seq_len - 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

class NLPVersorWrapper(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, n_heads=4, n_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim * 32)
        self.transformer = VersorTransformer(embed_dim, n_heads, n_layers, n_classes=vocab_size, use_rotor_pool=False)
        
    def forward(self, x):
        # x: (B, S)
        B, S = x.shape
        h = self.embedding(x).view(B, S, self.embed_dim, 32)
        for block in self.transformer.blocks:
            h = block(h)
        h_flat = h.view(B, S, -1)
        logits = self.transformer.classifier(h_flat)
        return logits

def run_nlp_task(seed):
    set_seed(seed)
    # Data Setup
    data_path = os.path.join(DATA_DIR, "wikitext-2-valid.txt")
    ensure_dir(DATA_DIR)
    
    if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
        # data = "This is a synthetic text for testing the language modeling capabilities of Versor. " * 1000
        # Let's try to get a slightly better synthetic dataset if download fails
        # or use the one we might have downloaded successfully earlier
        try:
            url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/wikitext-2/valid.txt"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = response.read().decode('utf-8')
            with open(data_path, "w") as f:
                f.write(data)
        except:
             data = "This is a synthetic text for testing the language modeling capabilities of Versor. " * 1000
    else:
        with open(data_path, "r") as f:
            data = f.read()

    data = data[:10000] 
    dataset = NLPDataset(data, seq_len=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = NLPVersorWrapper(dataset.vocab_size, embed_dim=8, n_heads=2, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    n_epochs = 5 # Increased from 2
    
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        steps = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
            if steps >= 50: break
        losses.append(epoch_loss / steps)
        
    return {"final_loss": losses[-1], "perplexity": math.exp(losses[-1]), "bpc": losses[-1] / math.log(2)}

# -----------------------------------------------------------------------------
# 2. Vision Task: Synthetic Cifar-Like
# -----------------------------------------------------------------------------

class VisionVersorWrapper(nn.Module):
    def __init__(self, num_classes=10, embed_dim=16, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_size = 4
        self.embed_dim = embed_dim
        self.input_dim = 3 * self.patch_size * self.patch_size
        self.patch_emb = nn.Linear(self.input_dim, embed_dim * 32)
        
        # Add Learnable Positional Encodings (8x8 grid of patches = 64)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, embed_dim, 32) * 0.02)
        
        self.transformer = VersorTransformer(embed_dim, n_heads, n_layers, n_classes=num_classes, use_rotor_pool=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Normalize: [0, 1] -> [-0.5, 0.5]
        x = x - 0.5
        
        # Patchify
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * self.patch_size * self.patch_size)
        
        # Embed and Add Position
        h = self.patch_emb(x).view(B, -1, self.embed_dim, 32)
        h = h + self.pos_emb
        
        logits = self.transformer(h)
        return logits

def run_vision_task(seed):
    set_seed(seed)
    num_samples = 1000 # Increased for better statistics
    images = torch.zeros(num_samples, 3, 32, 32)
    labels = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        cls = i % 10
        labels[i] = cls
        # Create distinct spatial features per class
        if cls < 4: # Corner features
            r, c = (cls // 2) * 24, (cls % 2) * 24
            images[i, :, r:r+8, c:c+8] = 1.0
        elif cls < 8: # Line features
            pos = (cls - 4) * 8
            if cls % 2 == 0: images[i, :, pos:pos+2, :] = 1.0 # Horizontal
            else: images[i, :, :, pos:pos+2] = 1.0 # Vertical
        else: # Center/Large features
            if cls == 8: images[i, :, 12:20, 12:20] = 1.0
            else: images[i, :, 4:28, 4:28] = 0.5 # Background wash
            
    images += torch.randn_like(images) * 0.1 # Add some noise
            
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = VisionVersorWrapper(num_classes=10, embed_dim=16, n_heads=4, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    n_epochs = 10 # Increased from 3
    accs = []
    
    for epoch in range(n_epochs):
        correct = 0
        total = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        accs.append(correct / total)
        
    return {"final_accuracy": accs[-1]}

# -----------------------------------------------------------------------------
# 3. Graph Task: Geometric Priors
# -----------------------------------------------------------------------------

class GeometricGraphWrapper(nn.Module):
    def __init__(self, input_dim=4, embed_dim=8, n_heads=2, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim * 32)
        self.transformer = VersorTransformer(embed_dim, n_heads, n_layers, n_classes=1, use_rotor_pool=True)
        
    def forward(self, x):
        B, N, D_in = x.shape
        h = self.input_proj(x).view(B, N, -1, 32)
        out = self.transformer(h)
        return out

def run_graph_task(seed):
    set_seed(seed)
    num_samples = 500
    num_atoms = 10
    pos = torch.rand(num_samples, num_atoms, 3) * 2 - 1
    atom_types = torch.randint(0, 5, (num_samples, num_atoms, 1)).float()
    inputs = torch.cat([pos, atom_types], dim=-1)
    
    mins = pos.min(dim=1)[0]
    maxs = pos.max(dim=1)[0]
    dims = maxs - mins
    targets = (dims.prod(dim=-1)).unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GeometricGraphWrapper(input_dim=4, embed_dim=8, n_heads=2, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    n_epochs = 10 # Increased from 5
    mses = []
    
    for epoch in range(n_epochs):
        epoch_mse = 0
        steps = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_mse += loss.item()
            steps += 1
        mses.append(epoch_mse / steps)
        
    return {"final_mse": mses[-1]}

# -----------------------------------------------------------------------------
# Multi-Seed Aggregation
# -----------------------------------------------------------------------------

def run_all_seeds():
    ensure_dir(RESULTS_DIR)
    print(f"Starting Multimodal Experiments on {len(SEEDS)} seeds: {SEEDS}")
    print(f"Epochs - NLP: 5, Vision: 10, Graph: 10")
    
    agg_results = {
        "nlp": [],
        "vision": [],
        "graph": []
    }
    
    start_total = time.time()
    
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        
        # NLP
        try:
            res = run_nlp_task(seed)
            agg_results["nlp"].append(res)
            print(f"[Seed {seed}] NLP Perplexity: {res['perplexity']:.4f}, BPC: {res['bpc']:.4f}")
        except Exception as e:
            print(f"[Seed {seed}] NLP Failed: {e}")
            
        # Vision
        try:
            res = run_vision_task(seed)
            agg_results["vision"].append(res)
            print(f"[Seed {seed}] Vision Accuracy: {res['final_accuracy']:.4f}")
        except Exception as e:
            print(f"[Seed {seed}] Vision Failed: {e}")
            
        # Graph
        try:
            res = run_graph_task(seed)
            agg_results["graph"].append(res)
            print(f"[Seed {seed}] Graph MSE: {res['final_mse']:.4f}")
        except Exception as e:
            print(f"[Seed {seed}] Graph Failed: {e}")
            
    # Calculate Statistics
    final_stats = {}
    
    # NLP Stats
    nlp_perps = [r["perplexity"] for r in agg_results["nlp"]]
    nlp_bpcs = [r["bpc"] for r in agg_results["nlp"]]
    final_stats["nlp"] = {
        "mean_perplexity": float(np.mean(nlp_perps)),
        "std_perplexity": float(np.std(nlp_perps)),
        "mean_bpc": float(np.mean(nlp_bpcs)),
        "std_bpc": float(np.std(nlp_bpcs)),
        "runs": len(nlp_perps)
    }
    
    # Vision Stats
    vision_accs = [r["final_accuracy"] for r in agg_results["vision"]]
    final_stats["vision"] = {
        "mean_accuracy": float(np.mean(vision_accs)),
        "std_accuracy": float(np.std(vision_accs)),
        "runs": len(vision_accs)
    }
    
    # Graph Stats
    graph_mses = [r["final_mse"] for r in agg_results["graph"]]
    final_stats["graph"] = {
        "mean_mse": float(np.mean(graph_mses)),
        "std_mse": float(np.std(graph_mses)),
        "runs": len(graph_mses)
    }
    
    duration = time.time() - start_total
    final_stats["metadata"] = {
        "duration_total": duration,
        "seeds": SEEDS,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"multimodal_multiseed_{timestamp}.json")
    with open(filename, 'w') as f:
        json.dump(final_stats, f, indent=2)
        
    print("\n" + "="*50)
    print("FINAL AGGREGATED RESULTS")
    print("="*50)
    print(f"NLP Perplexity: {final_stats['nlp']['mean_perplexity']:.4f} ± {final_stats['nlp']['std_perplexity']:.4f} (BPC: {final_stats['nlp']['mean_bpc']:.4f})")
    print(f"Vision Accuracy: {final_stats['vision']['mean_accuracy']:.4f} ± {final_stats['vision']['std_accuracy']:.4f}")
    print(f"Graph MSE: {final_stats['graph']['mean_mse']:.4f} ± {final_stats['graph']['std_mse']:.4f}")
    print(f"Total time: {duration:.2f}s")
    print(f"Saved to: {filename}")
    
    return final_stats

# Backward compatibility wrappers if called individually
def run_nlp_task_wrapper(): return run_nlp_task(SEED)
def run_vision_task_wrapper(): return run_vision_task(SEED)
def run_graph_task_wrapper(): return run_graph_task(SEED)

if __name__ == "__main__":
    run_all_seeds()
