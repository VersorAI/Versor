import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import numpy as np

# Add parent directory to path to import Model
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Model.model import VersorTransformer

# Config
DATA_PATH = "/Users/mac/Desktop/Versor/wiki.train.tokens"
SAVE_PATH = "results/real_wikitext_results.json"
DEVICE = torch.device("cpu") # MPS is slow for the geometric product loop
BATCH_SIZE = 64
SEQ_LEN = 128
EMBED_DIM = 24
N_HEADS = 4
N_LAYERS = 2
LR = 1e-3
MAX_STEPS = 2000 # Enough to see if it converges to 3.22

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

class NLPVersorWrapper(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, n_heads=4, n_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.ga_dim = 32
        self.embedding = nn.Embedding(vocab_size, embed_dim * self.ga_dim)
        # We set use_rotor_pool=False because NLP requires per-token next-character 
        # prediction. The internal forward pass is bypassed in the wrapper to 
        # allow for S-length sequence output rather than a pooled summary.
        self.transformer = VersorTransformer(embed_dim, n_heads, n_layers, n_classes=vocab_size, use_rotor_pool=False)
        
    def forward(self, x):
        B, S = x.shape
        h = self.embedding(x).view(B, S, self.embed_dim, self.ga_dim)
        # Process through blocks
        for block in self.transformer.blocks:
            h = block(h)
        # Reshape for classification head
        h_flat = h.view(B, S, -1)
        logits = self.transformer.classifier(h_flat)
        return logits

def train_wikitext():
    print(">>> [DEBUG] train_wikitext() started")
    print(f"Loading WikiText-103 from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    with open(DATA_PATH, "rb") as f:
        # Load as bytes for faster mapping
        data_bytes = f.read()
    
    unique_bytes = sorted(list(set(data_bytes)))
    vocab_size = len(unique_bytes)
    byte_to_idx = np.zeros(256, dtype=np.long)
    for i, b in enumerate(unique_bytes):
        byte_to_idx[b] = i
    print(f"Vocab Size: {vocab_size}")
    
    # Map all bytes to indices efficiently
    data_tensor = torch.from_numpy(byte_to_idx[np.frombuffer(data_bytes, dtype=np.uint8)].copy())
    
    # Split: Train/Val (95/5) for full dataset
    split_idx = int(0.95 * len(data_tensor))
    train_data = data_tensor[:split_idx]
    val_data = data_tensor[split_idx:]
    
    train_dataset = NLPDataset(train_data, seq_len=SEQ_LEN)
    val_dataset = NLPDataset(val_data, seq_len=SEQ_LEN)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = NLPVersorWrapper(vocab_size, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"Starting Training on {DEVICE}...")
    start_time = time.time()
    
    history = []
    step = 0
    # Increase MAX_STEPS for full dataset convergence
    MAX_TRAIN_STEPS = 1000 
    for epoch in range(1): # One pass (or partial pass) is enough
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % 100 == 0 or step <= 10:
                perp = math.exp(loss.item())
                bpc = loss.item() / math.log(2)
                print(f"Step {step}/{MAX_TRAIN_STEPS} | Loss: {loss.item():.4f} | Perp: {perp:.4f} | BPC: {bpc:.4f}")
                history.append({
                    "step": step,
                    "loss": loss.item(),
                    "perplexity": perp,
                    "bpc": bpc
                })
            
            if step >= MAX_TRAIN_STEPS:
                break
        if step >= MAX_TRAIN_STEPS: break
        
    print(f"Final Validation...")
    model.eval()
    val_loss = 0
    val_steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            val_loss += loss.item()
            val_steps += 1
            if val_steps >= 100: break # Quick validation
            
    avg_val_loss = val_loss / val_steps
    val_perp = math.exp(avg_val_loss)
    val_bpc = avg_val_loss / math.log(2)
    
    print(f"Validation Result: Loss {avg_val_loss:.4f} | Perplexity {val_perp:.4f} | BPC {val_bpc:.4f}")
    
    # Save results
    results = {
        "final_stats": {
            "validation_loss": avg_val_loss,
            "validation_perplexity": val_perp,
            "validation_bpc": val_bpc,
            "training_time": time.time() - start_time
        },
        "history": history
    }
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    import json
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    train_wikitext()
