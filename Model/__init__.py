"""
Geometric Transformer for Conformal Geometric Algebra Cl(4,1).
"""
import os
import sys

# Ensure library is in path to find gacore
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
lib_dir = os.path.join(root_dir, "library")

if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

from .core import (
    conformal_lift,
    gp_cl41,
    wedge_cl41,
    inner_cl41,
    reverse_cl41,
    normalize_cl41,
    GRADE_INDICES
)

from .layers import (
    VersorLinear,
    VersorAttention
)

from .model import (
    VersorTransformer,
    VersorActivation,
    VersorBlock,
    RecursiveRotorAccumulator
)

__all__ = [
    'conformal_lift',
    'gp_cl41',
    'wedge_cl41',
    'inner_cl41',
    'reverse_cl41',
    'normalize_cl41',
    'GRADE_INDICES',
    'VersorLinear',
    'VersorAttention',
    'VersorTransformer',
    'VersorActivation',
    'VersorBlock',
    'RecursiveRotorAccumulator'
]
