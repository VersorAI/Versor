# Versor: Geometric Transformer for Conformal Geometric Algebra

A native Geometric Algebra Transformer architecture based on Conformal Geometric Algebra $\mathcal{C}l(4,1)$, implementing multivector representations and geometric products for deep learning.

## Quick Start

Start with the interactive tutorial:

```bash
jupyter notebook quickstart.ipynb
```

This notebook walks you through:
1. Importing the Versor architecture
2. Creating a simple dataset (learning x²)
3. Training the model
4. Testing and evaluation

The quickstart provides a minimal working example you can adapt to your own problems!

## 📁 Repository Structure

```
Versor/
├── quickstart.ipynb          # 👈 START HERE! Interactive tutorial
├── Model/                    # Core Versor architecture
│   ├── __init__.py
│   ├── core.py              # Geometric algebra operations (Cl(4,1))
│   ├── layers.py            # VersorLinear, VersorAttention
│   └── model.py             # VersorTransformer, VersorBlock
├── tasks/                    # Task-specific implementations
│   ├── nlp/                 # Natural language processing tasks
│   ├── vision/              # Computer vision tasks
│   ├── nbody/               # N-body physics simulations
│   ├── topology/            # Topological reasoning tasks
│   ├── multimodal/          # Multimodal learning
│   ├── scripts/             # Analysis and benchmarking scripts
│   └── figures/             # Generated plots and visualizations
├── library/                 # Utility functions and helpers
├── GAPU                     # Purpose-built silicon architecture               
├── gatr/                     # GATr baseline implementation
├── data/                     # Datasets
├── results/                  # Experimental results
├── requirements.txt          # Python dependencies
└── kernel.py                # Custom CUDA kernels
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Versor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For CUDA acceleration:
```bash
# Ensure you have CUDA toolkit installed
# The custom kernels in kernel.py will be compiled automatically
```

## Core Components

### Model Architecture

- **VersorTransformer**: Full geometric transformer for Cl(4,1)
- **VersorBlock**: High-performance block with Geometric Product Attention (GPA)
- **VersorAttention**: Attention mechanism using geometric products
- **VersorLinear**: Linear layer preserving multivector structure
- **RecursiveRotorAccumulator**: Rotor-based sequence pooling

### Geometric Algebra Operations

- `conformal_lift`: Lift 4D points to Cl(4,1) multivectors
- `gp_cl41`: Geometric product in Cl(4,1)
- `wedge_cl41`: Wedge (exterior) product
- `inner_cl41`: Inner product
- `reverse_cl41`: Clifford conjugation
- `normalize_cl41`: Manifold normalization

## Example Tasks

### NLP: Dyck Language Recognition
```bash
cd tasks/nlp
python dyck_rotor.py --depths 20 50 100 --repeats 3
```

### Vision: CIFAR Classification
```bash
cd tasks/vision
# See task-specific README for details
```

### Physics: N-Body Simulation
```bash
cd tasks/nbody
# See task-specific README for details
```

### Real-world Benchmarks (Full Data)
```bash
# Molecular Dynamics (MD17 Salicylic Acid)
python tasks/real_world/run_md17_real.py

# Language Modeling (WikiText-103 Character-level)
python tasks/real_world/run_wikitext103_real.py
```

## 📊 Benchmarking

Run comprehensive benchmarks:

```bash
# Small-scale benchmark
python tasks/scripts/benchmark_versor_small.py

# Large-scale benchmark
python tasks/scripts/benchmark_versor_large.py

# Compare with GATr baseline
python tasks/scripts/benchmark_gatr.py

# Generate scaling plots
python tasks/scripts/generate_plot.py
```

## Adapting to Your Problem

The `quickstart.ipynb` provides a template for:
- **Regression tasks**: Predict continuous values
- **Classification tasks**: Modify the output layer
- **Sequence modeling**: Use the rotor accumulator
- **Custom data**: Adapt the data loading section

Key steps:
1. Prepare your data in the appropriate format
2. Lift data to Cl(4,1) using `conformal_lift` or custom embedding
3. Configure model hyperparameters (embed_dim, n_heads, n_layers)
4. Train with standard PyTorch training loop
5. Evaluate and visualize results

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Huy:2026wcd,
    author = "Huy, Truong Minh and Hirst, Edward",
    title = "{Versor: A Geometric Sequence Architecture}",
    eprint = "2602.10195",
    archivePrefix = "arXiv",
    primaryClass = "cs.LG",
    month = "2",
    year = "2026"
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

See LICENSE file for details.

## Resources
Browse `tasks/` for domain-specific implementations

## Troubleshooting

**Import errors**: Ensure you're in the repository root and have installed all dependencies

**CUDA errors**: Check CUDA compatibility with your PyTorch version

**Memory issues**: Reduce batch size or model dimensions

**Convergence issues**: Try adjusting learning rate or using gradient clipping

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the authors.

---
