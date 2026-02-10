import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Physics.models import StandardTransformer, GraphNetworkSimulator, HamiltonianNN, EquivariantGNN, MambaSimulator, VersorRotorRNN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

models = {
    "Transformer (d=128)": StandardTransformer(d_model=128),
    "GNS (hidden=64)": GraphNetworkSimulator(hidden_dim=64),
    "HNN (hidden=128)": HamiltonianNN(hidden_dim=128),
    "Mamba (d_model=256)": MambaSimulator(d_model=256),
    "EGNN (hidden=64)": EquivariantGNN(hidden_dim=64),
    "Versor (Base)": VersorRotorRNN(),
}

print(f"{'Model':<25} | {'Parameters':<15}")
print("-" * 45)

for name, model in models.items():
    print(f"{name:<25} | {count_parameters(model):<15,}")
