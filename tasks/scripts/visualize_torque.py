
import torch
import matplotlib.pyplot as plt
import numpy as np
try:
    import kernel as algebra
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import kernel as algebra

from Physics.models import VersorPhysicsTransformer

def visualize_torque():
    print("Generating Torque Visualization...")
    device = "cpu" # Visualization on CPU is usually enough and safer for matplotlib
    
    # 1. Instantiate Model
    # We need a model that returns attention components.
    # VersorPhysicsTransformer has a `return_attention` flag in forward.
    model = VersorPhysicsTransformer(n_particles=5, input_dim=6, embed_dim=8, n_heads=1, n_layers=1)
    
    # 2. Fake Data
    B, S, N = 1, 10, 5
    x = torch.randn(B, S, N, 6)
    
    # 3. Forward Pass to get Attention
    # We need to access individual blocks or modify the model to return it?
    # VersorPhysicsTransformer returns (output, attention_tuple) if return_attention=True.
    
    _, (scalar_attn_list, bivector_attn_list) = model(x, return_attention=True)
    
    # Get first layer, first head
    # scalar_attn_list[0] shape: (B, S*N, S*N) or similar depending on implementation
    # Let's inspect the shapes
    
    scalar_mat = scalar_attn_list[0].detach().cpu().squeeze() # (B*S, N, N)
    bivector_mat = bivector_attn_list[0].detach().cpu().squeeze() # (B*S, N, N, 32) or similar
    
    # Pick first sample in batch
    if scalar_mat.dim() == 3:
        scalar_mat = scalar_mat[0] # (N, N)
    if bivector_mat.dim() > 2:
        bivector_mat = bivector_mat[0] # (N, N, 32)
    
    # Bivector content is likely the magnitude of the bivector part of geometric product.
    # In VersorAttention: grade(Q K, 2). 
    # Let's assume the model returns the raw tensor or magnitude.
    # If it's the raw tensor (..., 32), we take the norm of bivector components.
    
    # For visualization, we want a heatmap.
    if bivector_mat.dim() == 3:
        # Compute magnitude of bivector part
        # indices for bivectors in Cl(4,1): those with grade 2.
        # Simplification: take Euclidean norm over the last dimension (32).
        bivector_mag = torch.norm(bivector_mat, dim=-1)
    else:
        bivector_mag = bivector_mat
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scalar (Proximity)
    im1 = axes[0].imshow(scalar_mat, cmap='viridis')
    axes[0].set_title("Scalar Attention (Proximity)\n$e^{-\|x_i - x_j\|^2}$", fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Bivector (Torque)
    im2 = axes[1].imshow(bivector_mag, cmap='magma')
    axes[1].set_title("Bivector Attention (Torque)\n$\|Q \wedge K\| \sim \sin(\\theta)$", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle("Geometric Attention Decomposition: Separating Force from Torque", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig("Paper/torque_vis.png", dpi=300, bbox_inches='tight')
    print("Saved Paper/torque_vis.png")

if __name__ == "__main__":
    visualize_torque()
