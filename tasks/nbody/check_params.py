import torch
import sys
import os

# Ensure we can import models from Physics/tasks/nbody
root_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(os.path.join(root_dir, "tasks/nbody"))
sys.path.append(os.path.join(root_dir, "library"))

try:
    from models import (StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, 
                        HamiltonianNN, MultiChannelVersor, HamiltonianVersorNN, 
                        EquivariantGNN, MambaSimulator)
except ImportError as e:
    print(f"Import Error: {e}")
    # Try different path
    sys.path.append(os.getcwd())
    from models import (StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, 
                        HamiltonianNN, MultiChannelVersor, HamiltonianVersorNN, 
                        EquivariantGNN, MambaSimulator)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

models = {
    "Transformer (d=128)": StandardTransformer(n_particles=5),
    "Versor (0.007M claim)": VersorRotorRNN(n_particles=5),
    "Versor (Multi-Channel)": MultiChannelVersor(n_particles=5, n_channels=16, n_heads=4),
    "GNS": GraphNetworkSimulator(n_particles=5),
    "HNN": HamiltonianNN(n_particles=5),
    "Mamba": MambaSimulator(n_particles=5),
    "EGNN": EquivariantGNN(n_particles=5),
    "Ham-Versor": HamiltonianVersorNN(n_particles=5)
}

print("="*60)
print(f"{'Model':<30} | {'Parameter Count':<20}")
print("-" * 60)
for name, model in models.items():
    print(f"{name:<30} | {count_parameters(model):,}")
print("="*60)
