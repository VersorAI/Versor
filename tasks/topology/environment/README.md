# Versor Environment Setup

This directory contains the necessary configuration to replicate the Versor curriculum benchmarks on high-performance compute clusters (A100/H100).

## 1. Installation

It is recommended to use a clean virtual environment or a Conda environment:

```bash
# Create and activate environment
python -m venv geo_env
source geo_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Hardware Acceleration
The code is optimized for NVIDIA Ampere and Hopper architectures. 
- **TensorFloat-32 (TF32):** Enabled by default in `master_sweep_curriculum.py` via `torch.backends.cuda.matmul.allow_tf32 = True`.
- **BFloat16:** The training engine automatically detects H100/A100 support and utilizes `torch.amp.autocast` with `dtype=torch.bfloat16` for numerical stability in the $Cl_{4,1}$ manifold.

## 3. Running the Systematic Sweep
To reproduce the scaling results and the "Euclidean Crisis" baseline collapse (128x128 grid), execute the following command:

```bash
python master_sweep_curriculum.py --sizes 8 16 32 64 128 --repeats 5 --epochs 100 --outfile results_final.json
```

### Parameters:
- `--sizes`: Defines the grid resolution (Sequence length = size^2).
- `--repeats`: Number of independent trials for statistical significance (calculates mean MCC and standard deviation).
- `--epochs`: Maximum epochs per curriculum stage.
- `--d_vecs`: Hidden vector capacity (default sweep: 2 to 12).

## 4. Output
The script generates:
1. `curriculum_sweep_results.json`: Raw MCC data and parameter counts.
2. `curriculum_sweep_plots.png`: Visual comparison of CGA vs. Standard Transformer scaling and Efficiency Ratios.
