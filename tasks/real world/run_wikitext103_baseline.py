import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import json
import numpy as np

# Config (Same as Versor script)
DATA_PATH = "/Users/mac/Desktop/Versor/wiki.train.tokens"
SAVE_PATH = "results/real_wikitext_baseline_transformer.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
SEQ_LEN = 128
D_MODEL = 152 
N_HEADS = 8
N_LAYERS = 2
LR = 1e-3
MAX_STEPS = 2000 

class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, seq_len=128):
        self.data = data_tensor
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

class StandardTransformerNLP(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embedding(x) + self.pos_emb
        h = self.encoder(h)
        return self.head(h)

def train_baseline():
    print(f"Loading WikiText-103 (Standard Baseline) from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data_text = f.read(5_000_000) # Same 5MB Subset
    
    chars = sorted(list(set(data_text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    
    data_tensor = torch.tensor([char_to_idx[c] for c in data_text], dtype=torch.long)
    split_idx = int(0.9 * len(data_tensor))
    train_data = data_tensor[:split_idx]
    val_data = data_tensor[split_idx:]
    
    train_dataset = NLPDataset(train_data, seq_len=SEQ_LEN)
    val_dataset = NLPDataset(val_data, seq_len=SEQ_LEN)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = StandardTransformerNLP(vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.4f}M")
    print(f"Starting Training on {DEVICE}...")
    start_time = time.time()
    
    history = []
    step = 0
    for epoch in range(1):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % 50 == 0:
                perp = math.exp(loss.item())
                print(f"Step {step}/{MAX_STEPS} | Loss: {loss.item():.4f} | Perp: {perp:.4f}")
                history.append({"step": step, "loss": loss.item(), "perplexity": perp})
            
            if step >= MAX_STEPS: break
        if step >= MAX_STEPS: break
        
    print(f"Final Validation...")
    model.eval()
    val_loss = 0; val_steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            val_loss += loss.item(); val_steps += 1
            if val_steps >= 100: break
            
    avg_val_loss = val_loss / val_steps
    val_perp = math.exp(avg_val_loss)
    print(f"Baseline Result: Loss {avg_val_loss:.4f} | Perplexity {val_perp:.4f}")
    
    results = {
        "final_stats": {
            "validation_loss": avg_val_loss,
            "validation_perplexity": val_perp,
            "training_time": time.time() - start_time
        },
        "history": history
    }
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    train_baseline()
