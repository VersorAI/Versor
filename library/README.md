# gacore: High-Performance Geometric Algebra Core

[![PyPI version](https://img.shields.io/pypi/v/gacore.svg)](https://pypi.org/project/gacore/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**gacore** is a hyper-optimized Geometric Algebra (GA) library design for deep learning, scientific computing, and physics-informed neural networks. It is a descendant of the `clifford` library, specialized for high-dimensional Conformal Geometric Algebra (CGA) such as $\Cl(4,1)$.

## 🚀 Key Features

- **Multi-Backend Acceleration**: Native support for **NVIDIA (Triton)**, **Apple Silicon (MLX)**, and high-performance **CPU** backends.
- **Matrix Isomorphism Turbo Path**: Leverages the isomorphism $\Cl \cong \text{Mat}(4, \mathbb{C}) \oplus \text{Mat}(4, \mathbb{C})$ for state-of-the-art per-step latency (surpassing standard Transformers).
- **Bit-Masked Basis Contraction**: Eliminates memory bottlenecks of Cayley tables, achieving up to **78x speedup**.
- **PyTorch Integration**: Drop-in compatibility with PyTorch modules, including autograd-ready layers and custom optimizers (`CayleyAdam`).
- **Signature Agnostic**: General support for arbitrary Clifford signatures $(p, q, r)$.

## 🛠 Installation

```bash
pip install gacore
```

## 🏎 Quick Start

```python
import gacore.kernel as algebra
import torch

# Define a Cl(4,1) signature
signature = torch.tensor([1, 1, 1, 1, -1])

# High-performance Geometric Product
a = torch.randn(1024, 32)
b = torch.randn(1024, 32)
c = algebra.geometric_product(a, b, signature, method="matrix")

# Manifold Normalization
c_norm = algebra.manifold_normalization(c, signature)
```

## 📊 Performance

| Backend | Engine | Latency (ms) |
| :--- | :--- | :--- |
| **CPU** | Bit-Masked | 1.54 |
| **CPU** | **Matrix Turbo** | **1.08** |
| **NVIDIA** | Triton | < 0.1 |

*Benchmarks measured on Apple M4 for a single-layer Versor update with $L=50$.*

## 📚 Documentation
For detailed usage, mathematical derivations, and advanced configurations, please refer to the [official documentation](https://github.com/VersorAI/Versor).

## 📄 License
gacore is released under the **BSD 3-Clause License**. See [LICENSE.txt](LICENSE.txt) for details.
