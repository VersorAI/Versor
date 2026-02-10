"""
Geometric Transformer for Conformal Geometric Algebra Cl(4,1).

This package implements a native Geometric Algebra Transformer for classifying
Ising Model phase transitions using multivector representations and geometric products.
"""

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
