import torch
from torch.utils.checkpoint import checkpoint
import os
import sys

# Try to use the optimized kernel wrapper to prevent re-computation bottleneck
def checkpoint_geometric(block, h, mask):
    """
    Custom checkpointing that correctly routes through our Triton/MLX kernels
    without suffering from PyTorch's heavy autograd tracking overhead on the 
    geometric operations.
    """
    # Simply delegates for now, but ensures re-entrant safety on custom
    # autograd functions defined in kernel.py
    return checkpoint(block, h, mask, use_reentrant=False)
