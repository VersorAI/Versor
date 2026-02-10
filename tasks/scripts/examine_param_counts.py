import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Model.model import VersorTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

configs = [
    {"embed_dim": 4, "n_layers": 2},
    {"embed_dim": 8, "n_layers": 2},
    {"embed_dim": 16, "n_layers": 4},
    {"embed_dim": 32, "n_layers": 8},
    {"embed_dim": 64, "n_layers": 12},
    {"embed_dim": 128, "n_layers": 12},
]

print(f"{'Embed Dim':<10} | {'Layers':<10} | {'Parameters':<15}")
print("-" * 40)

for config in configs:
    try:
        model = VersorTransformer(
            embed_dim=config["embed_dim"], 
            n_heads=max(2, config["embed_dim"] // 8), 
            n_layers=config["n_layers"], 
            n_classes=10
        )
        params = count_parameters(model)
        print(f"{config['embed_dim']:<10} | {config['n_layers']:<10} | {params:<15,}")
    except Exception as e:
        print(f"Error with config {config}: {e}")
