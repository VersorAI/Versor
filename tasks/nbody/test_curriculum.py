#!/usr/bin/env python3
"""
THE KILLER TEST: Curriculum Generalization
Train on short sequences, test on long sequences
This is what Versor SHOULD excel at due to geometric invariance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from datetime import datetime

sys.path.append('..')

from data_gen import generate_gravity_data
from models import StandardTransformer, VersorRotorRNN, GraphNetworkSimulator

def train_model_on_length(model, seq_length, n_samples=100, epochs=20, device='cpu'):
    """Train model on specific sequence length"""
    print(f"  Training on L={seq_length}...", end=' ', flush=True)
    
    # Generate training data
    train_data = generate_gravity_data(n_samples=n_samples, n_steps=seq_length, device=device)
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 1:]
    
    # Train
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    print(f"Done (final loss={loss.item():.4f})")
    return model

def evaluate_on_length(model, seq_length, n_samples=20, device='cpu'):
    """Evaluate model on specific sequence length"""
    model.eval()
    
    test_data = generate_gravity_data(n_samples=n_samples, n_steps=seq_length, device=device)
    X_test = test_data[:, :-1]
    Y_test = test_data[:, 1:]
    
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        try:
            pred = model(X_test)
            mse = loss_fn(pred, Y_test).item()
            return {"mse": float(mse), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

def main():
    print("="*70)
    print("CURRICULUM GENERALIZATION TEST")
    print("="*70)
    print("\nThis tests the KEY HYPOTHESIS of geometric models:")
    print("→ Learned geometric relationships should transfer across scales\n")
    
    print("Protocol:")
    print("  1. Train all models on SHORT sequences (L=50)")
    print("  2. Test on SHORT (L=50), MEDIUM (L=100), LONG (L=150)")
    print("  3. Measure Transfer Gap = MSE(L_test) / MSE(L_train)\n")
    
    print("Expected:")
    print("  - Transformer: Transfer Gap ↑↑ (overfits to length)")
    print("  - GNS: Transfer Gap ↑ (graph structure bias)")
    print("  - Versor: Transfer Gap ↓ (geometric invariance!)\n")
    
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Training configuration
    TRAIN_LENGTH = 50
    TEST_LENGTHS = [50, 100, 150]
    TRAIN_SAMPLES = 100
    TEST_SAMPLES = 20
    EPOCHS = 20
    
    print("="*70)
    print(f"TRAINING PHASE (L={TRAIN_LENGTH})")
    print("="*70)
    
    models = {
        "Transformer": StandardTransformer(n_particles=5).to(device),
        "Versor": VersorRotorRNN(n_particles=5).to(device),
        "GNS": GraphNetworkSimulator(n_particles=5).to(device)
    }
    
    # Train all models
    for name, model in models.items():
        print(f"\n{name}:")
        train_model_on_length(model, TRAIN_LENGTH, TRAIN_SAMPLES, EPOCHS, device)
    
    # Evaluate on all lengths
    print("\n" + "="*70)
    print("EVALUATION PHASE")
    print("="*70)
    
    results = {
        "config": {
            "train_length": TRAIN_LENGTH,
            "test_lengths": TEST_LENGTHS,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "epochs": EPOCHS,
            "seed": 42
        },
        "models": {}
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        model_results = {}
        
        for L in TEST_LENGTHS:
            print(f"  L={L:3d}...", end=' ', flush=True)
            result = evaluate_on_length(model, L, TEST_SAMPLES, device)
            
            if result["success"]:
                mse = result["mse"]
                print(f"MSE={mse:.4f}")
                model_results[f"L{L}"] = mse
            else:
                print(f"FAILED: {result['error']}")
                model_results[f"L{L}"] = None
        
        # Calculate transfer gaps
        if model_results.get("L50") and model_results.get("L50") > 0:
            model_results["transfer_gap_100"] = model_results.get("L100", 0) / model_results["L50"]
            model_results["transfer_gap_150"] = model_results.get("L150", 0) / model_results["L50"]
        
        results["models"][name] = model_results
    
    # Analysis
    print("\n" + "="*70)
    print("TRANSFER GAP ANALYSIS")
    print("="*70)
    
    print(f"\n{'Model':<15} | {'L=50':<10} | {'L=100':<10} | {'L=150':<10} | {'Gap 100':<10} | {'Gap 150'}")
    print("-" * 85)
    
    for name in ["Transformer", "Versor", "GNS"]:
        if name in results["models"]:
            r = results["models"][name]
            l50 = r.get("L50", 0)
            l100 = r.get("L100", 0)
            l150 = r.get("L150", 0)
            gap100 = r.get("transfer_gap_100", 0)
            gap150 = r.get("transfer_gap_150", 0)
            
            print(f"{name:<15} | {l50:<10.4f} | {l100:<10.4f} | {l150:<10.4f} | {gap100:<10.2f}x | {gap150:<10.2f}x")
    
    # Find winner
    print("\n" + "="*70)
    print("WINNER ANALYSIS")
    print("="*70)
    
    gaps = {}
    for name in results["models"]:
        gap = results["models"][name].get("transfer_gap_150", float('inf'))
        if gap and gap > 0:
            gaps[name] = gap
    
    if gaps:
        best_model = min(gaps, key=gaps.get)
        best_gap = gaps[best_model]
        
        print(f"\n✓ BEST GENERALIZATION: {best_model}")
        print(f"  Transfer Gap (L=150): {best_gap:.2f}x")
        print(f"\n  This means {best_model} performance degrades LEAST")
        print(f"  when tested on 3× longer sequences than training!")
        
        if best_model == "Versor":
            print(f"\n  🎯 HYPOTHESIS CONFIRMED!")
            print(f"  Geometric priors enable scale-invariant learning!")
            print(f"\n  → This is your KILLER RESULT vs baselines! ←")
        else:
            print(f"\n  ⚠️  Unexpected: {best_model} won")
            print(f"  This suggests Versor may need:")
            print(f"  - More training epochs")
            print(f"  - Better hyperparameters")
            print(f"  - Architecture improvements")
    
    # Save results
    filename = f"curriculum_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {filename}")
    
    # Generate paper text
    print("\n" + "="*70)
    print("LATEX FOR PAPER")
    print("="*70)
    
    print("""
% Add this to your paper (after Table 2):

\\subsection{Curriculum Generalization Across Sequence Lengths}

To evaluate whether Versor's geometric priors enable scale-invariant learning,
we trained all models on short sequences (L=50) and tested on progressively
longer sequences (L=100, 150). We measure the \\textbf{Transfer Gap}: 
the ratio of test MSE on long sequences to training-length MSE.

Results show that""")
    
    if gaps and best_model == "Versor":
        versor_gap = gaps["Versor"]
        transformer_gap = gaps.get("Transformer", 0)
        print(f""" Versor achieves the smallest transfer gap ({versor_gap:.2f}×),
outperforming the Transformer ({transformer_gap:.2f}×). This demonstrates that
geometric relationships learned on short trajectories transfer to longer horizons
without additional training, validating the hypothesis that conformal manifold
representations capture scale-invariant physical laws.""")
    else:
        print(" [RESULTS TO BE FILLED BASED ON ACTUAL EXPERIMENTS]")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
