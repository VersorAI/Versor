import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import json
import math
import sys
import os
import random
from typing import List, Tuple, Dict

# =================================================================
# SYSTEM ARCHITECTURE AND OPTIMIZATION DIRECTIVES
# =================================================================
# Ensure high-performance matrix multiplications on Nvidia Ampere+
if torch.cuda.is_available():
    # Recommended settings for modern PyTorch
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

# =================================================================
# CONFORMAL GEOMETRIC ALGEBRA CORE (Cl(4,1))
# =================================================================

class Cl41Algebra:
    """
    Metric Signature: \mathbb{R}^{4,1} (+, +, +, +, -)
    Geometric product derivation via precomputed Cayley transformation matrices.
    """
    def __init__(self, device: torch.device):
        self.device = device
        # Basis metric: e1..e4 = +1, e5 = -1
        self.metric = [1, 1, 1, 1, -1] 
        self.dim = 32
        
        # Precompute the multiplication table (Cayley Table)
        # Shape: (32, 32, 32)
        # Usage: table[i, j, k] is the coefficient of blade k in the product of blade i and j
        self.gp_map = self._generate_cayley_table().to(device)
        
        # Precompute signs for quadratic norm (E_A * ~E_A)
        self.quad_norm_signs = self._generate_quad_norm_signs().to(device)
        
        # Precompute signs for Reverse operator
        self.reverse_signs = self._generate_reverse_signs().to(device)

    def _generate_reverse_signs(self) -> torch.Tensor:
        signs = torch.zeros(32, dtype=torch.float32)
        for i in range(32):
            grade = bin(i).count('1')
            signs[i] = -1.0 if (grade * (grade - 1) // 2) % 2 == 1 else 1.0
        return signs

    def _generate_cayley_table(self) -> torch.Tensor:
        n_dim = 32
        table = torch.zeros((n_dim, n_dim, n_dim), dtype=torch.float32)
        
        for i in range(n_dim):
            for j in range(n_dim):
                # The result blade index is XOR (basis vectors cancel or combine)
                k = i ^ j
                
                # Calculate the sign of the product
                sign = 1.0
                
                # 1. Anticommutativity (Count swaps)
                # Convert to list of active basis bits
                bits_i = [b for b in range(5) if (i >> b) & 1]
                bits_j = [b for b in range(5) if (j >> b) & 1]
                
                # Concatenate and bubble-sort to count swaps
                combined = bits_i + bits_j
                swaps = 0
                for _ in range(len(combined)):
                    for idx in range(len(combined) - 1):
                        if combined[idx] > combined[idx+1]:
                            combined[idx], combined[idx+1] = combined[idx+1], combined[idx]
                            swaps += 1
                if swaps % 2 == 1: sign *= -1.0
                
                # 2. Metric Signature (Squared vectors)
                # Any bit present in both i and j is squared.
                common = i & j
                for b in range(5):
                    if (common >> b) & 1:
                        sign *= self.metric[b]
                
                table[i, j, k] = sign
        return table

    def _generate_quad_norm_signs(self) -> torch.Tensor:
        # Calculates the scalar part of (E_A * ~E_A)
        # ~E_A (Reverse) = (-1)^(grade*(grade-1)/2) E_A
        # E_A * E_A depends on the metric of constituent vectors
        signs = torch.zeros(32, dtype=torch.float32)
        for i in range(32):
            grade = bin(i).count('1')
            reverse_sign = -1.0 if (grade * (grade - 1) // 2) % 2 == 1 else 1.0
            
            metric_sign = 1.0
            for b in range(5):
                if (i >> b) & 1: metric_sign *= self.metric[b]
            
            signs[i] = reverse_sign * metric_sign
        return signs

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Einsum: ...i (blade 1), ...j (blade 2), ijk (table) -> ...k (result)
        return torch.einsum('...i, ...j, ijk -> ...k', a, b, self.gp_map)

    def manifold_normalization(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Canonical normalization for multivector states.
        Utilizes a stabilized L2-norm projection to preserve manifold integrity
        and prevent numerical divergence in high-depth recursive sequences.
        """
        l2_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (l2_norm + eps)

# =================================================================
# ARCHITECTURAL COMPARISON: RECURSIVE ISOMETRIES VS EUCLIDEAN RNN
# =================================================================

class Versor_Rotor(nn.Module):
    """
    Rotor-based recursive architecture for O(1) context representation.
    The hidden state evolution is modeled as a sequence of isometric transformations:
    \Psi_{t+1} = \text{Norm}( R_t * \Psi_t )
    """
    def __init__(self, d_vectors: int, algebra: Cl41Algebra):
        super().__init__()
        self.d_vectors = d_vectors
        self.algebra = algebra
        
        # 1. Initialize Embedding
        self.token_embedding = nn.Embedding(4, 32)
        
        # Initialization of bivector generators for geometric rotations.
        with torch.no_grad():
            self.token_embedding.weight.normal_(0, 0.1) 
            self.token_embedding.weight[:, 0] = 1.0
            # Zero out the Vector and Trivector/Quadvector parts specifically?
            # Actually, standard GA practice is to use Bivectors for rotors.
            # R = exp(B). We'll let the model learn which parts to use.
            
        # Classifier: Map multivector state to 2 classes
        self.classifier = nn.Sequential(
            nn.Linear(d_vectors * 32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        # 2. Structural Gating Mechanism
        # Initialize gate to be "mostly open" (bias = 1.0) to encourage information flow
        self.gate = nn.Linear(32 + (d_vectors * 32), d_vectors)
        with torch.no_grad():
            self.gate.bias.fill_(1.0)
        
    def forward(self, x):
        b, seq_len = x.shape
        
        # 1. Initialize State (Psi) as Identity Rotor
        # Shape: (Batch, D_Vectors, 32)
        h = torch.zeros(b, self.d_vectors, 32, device=x.device)
        h[:, :, 0] = 1.0 
        
        # 2. Precompute and Normalize Input Rotors
        # r_in: (Batch, Seq_Len, 32)
        r_in = self.token_embedding(x)
        r_in = self.algebra.manifold_normalization(r_in)
        
        # 3. Recursive Loop (The O(1) Mechanism)
        for t in range(seq_len):
            r_t = r_in[:, t] # (Batch, 32)
            
            # --- GEOMETRIC FORGET GATE ---
            # Gate determines how much of the new rotation to apply
            gate_in = torch.cat([h.flatten(1), r_t], dim=-1)
            g = torch.sigmoid(self.gate(gate_in)).unsqueeze(-1)
            
            # Isotropic state update: \Psi_{t+1} = R_t \cdot \Psi_t
            r_t_unsq = r_t.unsqueeze(1) 
            h_next = self.algebra.geometric_product(r_t_unsq, h)
            
            # Pure Gated Update (Interpolate in multivector space then project back)
            h = (1 - g) * h + g * h_next
            
            # Re-project to the manifold to prevent error accumulation
            h = self.algebra.manifold_normalization(h)
            
        # 4. Classification
        return self.classifier(h.view(b, -1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Standard_LSTM(nn.Module):
    def __init__(self, target_params: int):
        super().__init__()
        # Auto-scale hidden dim to match parameter count of Versor-Model
        # Param count approx: 4 * ((32 + h) * h + h)
        embed_dim = 32
        
        # Simple heuristic to find hidden dim
        h = 8
        while True:
            dummy_lstm = nn.LSTM(embed_dim, h, batch_first=True)
            dummy_head = nn.Sequential(
                nn.Linear(h, h * 2),
                nn.ReLU(),
                nn.Linear(h * 2, 2)
            )
            p = sum(p.numel() for p in dummy_lstm.parameters()) + sum(p.numel() for p in dummy_head.parameters()) + (4*32)
            if p >= target_params:
                break
            h += 4
            
        self.embed = nn.Embedding(4, embed_dim)
        self.lstm = nn.LSTM(embed_dim, h, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(h, h * 2),
            nn.ReLU(),
            nn.Linear(h * 2, 2)
        )
        self.h_dim = h

    def forward(self, x):
        x = self.embed(x)
        output, (h_n, _) = self.lstm(x)
        # Use a MLP head for standard model too for fairness
        return self.head(h_n.squeeze(0))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def transfer_weights(source_model, target_model):
    """Transfer weights between models for curriculum learning."""
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    # Filter out mismatching sizes if any, but they should match in this script
    pretrained_dict = {k: v for k, v in source_dict.items() if k in target_dict}
    target_dict.update(pretrained_dict)
    target_model.load_state_dict(target_dict)
    return target_model

# =================================================================
# DATA SYNTHESIS: DYCK-N LINGUISTIC SEQUENCES
# =================================================================

def generate_dyck_n(n_samples: int, depth: int):
    """
    Generates balanced/corrupted bracket sequences.
    Labels: 1 = Valid, 0 = Corrupted.
    """
    data, labels = [], []
    pairs = {0: 1, 2: 3} # 0:(, 1:), 2:[, 3:]
    opens = [0, 2]
    
    for _ in range(n_samples):
        seq = []
        stack = []
        target_len = depth * 2
        
        # Valid generation logic
        while len(seq) < target_len:
            # Force push if stack empty or must fill to reach length
            must_push = len(stack) == 0
            # Force pop if stack matches remaining space
            must_pop = len(stack) == (target_len - len(seq))
            
            if must_pop:
                op = stack.pop()
                seq.append(pairs[op])
            elif must_push:
                char = random.choice(opens)
                seq.append(char)
                stack.append(char)
            else:
                # Random choice
                if random.random() > 0.5:
                    char = random.choice(opens)
                    seq.append(char)
                    stack.append(char)
                else:
                    op = stack.pop()
                    seq.append(pairs[op])
        
        # Corruption (50% chance)
        label = 1
        if random.random() > 0.5:
            label = 0
            # Corrupt one random position
            idx = random.randint(0, len(seq)-1)
            orig = seq[idx]
            opts = [0, 1, 2, 3]
            opts.remove(orig)
            seq[idx] = random.choice(opts)
            
        data.append(seq)
        labels.append(label)
        
    return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# =================================================================
# OPTIMIZATION PROTOCOL AND EVALUATION
# =================================================================

def train_model(model, depth, device, n_epochs=50, batch_size=256, lr=0.001):
    model.to(device)
    # Use torch.compile on the model for 2x-5x speedup in the RNN loop
    if hasattr(torch, 'compile') and sys.platform != 'win32':
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"    [WARN] torch.compile failed: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # OneCycleLR is often better for reaching high accuracy fast
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*2, total_steps=n_epochs, 
        pct_start=0.1, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss()
    
    # Larger dataset for statistically significant results
    n_samples_train = 8000
    n_samples_test = 2000
    train_x, train_y = generate_dyck_n(n_samples_train, depth)
    test_x, test_y = generate_dyck_n(n_samples_test, depth)
    
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    model.train()
    for epoch in range(n_epochs):
        # Mini-batch loop
        perm = torch.randperm(train_x.size(0))
        for i in range(0, train_x.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x, batch_y = train_x[idx], train_y[idx]
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            # Gradient clipping to ensure numerical stability in deep recursions
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(test_x)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == test_y).float().mean().item()
        
        # Step the scheduler
        scheduler.step()
        
        # Verbose print to see progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Ep {epoch+1} | Acc: {acc:.3f} | LR: {current_lr:.6f}")
            
        model.train()
        if acc > 0.995: break # Early stop via Accuracy
        if current_lr < 1e-6: # Early stop via LR Decay
            print(f"    [INFO] Stopping early: LR reached minimum threshold.")
            break
        
    return acc

# =================================================================
# 5. MAIN SWEEP
# =================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--depths', type=int, nargs='+', default=[20, 50, 100, 200, 500], help='Nesting depths')
    parser.add_argument('--repeats', type=int, default=3, help='Trials per depth')
    parser.add_argument('--d_vectors', type=int, default=16, help='Width of Versor-Model')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- STARTING DYCK-N RECURSIVE SWEEP ---")
    print(f"Device: {device}")
    
    # Init Algebra Engine
    algebra = Cl41Algebra(device)
    
    results = {}
    prev_versor = None
    prev_lstm = None
    
    for d in args.depths:
        seq_len = d * 2
        print(f"\n[Depth {d} | Seq Length {seq_len}]")
        results[d] = {'versor': [], 'lstm': []}
        
        for r in range(args.repeats):
            # 1. Versor-Model
            versor = Versor_Rotor(d_vectors=args.d_vectors, algebra=algebra)
            if prev_versor is not None and r == 0:
                versor = transfer_weights(prev_versor, versor)
            v_p_versorarams = versor.count_parameters()
            
            # 2. LSTM (Matched)
            lstm = Standard_LSTM(target_params=v_p_versorarams)
            if prev_lstm is not None and r == 0:
                lstm = transfer_weights(prev_lstm, lstm)
            l_params = lstm.count_parameters()
            
            if r == 0:
                print(f"  Params Match: Versor={v_p_versorarams} | LSTM={l_params}")
            
            # Train
            t0 = time.time()
            acc_g = train_model(versor, d, device, args.epochs)
            t_g = time.time() - t0
            
            t0 = time.time()
            acc_l = train_model(lstm, d, device, args.epochs)
            t_l = time.time() - t0
            
            print(f"    Run {r+1}: Versor={acc_g*100:.1f}% ({t_g:.1f}s) | LSTM={acc_l*100:.1f}% ({t_l:.1f}s)")
            
            results[d]['versor'].append(acc_g)
            results[d]['lstm'].append(acc_l)
            
            # Save the successful models for the next stage of the curriculum
            if r == 0: # Usually reuse the first trial of each depth for the next stage
                prev_versor = versor
                prev_lstm = lstm
            
    # Save
    with open('dyck_rotor_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\n[INFO] Results saved to dyck_rotor_results.json")