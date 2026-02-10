# DyckN Environment Setup

This directory contains the necessary configuration to replicate the DyckN curriculum benchmarks using Geometric Algebra Transformers.

## 1. Installation

It is recommended to use a clean virtual environment or a Conda environment:

```bash
# Create and activate environment
python -m venv dyckn_env
source dyckn_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Hardware Acceleration
The code is optimized for NVIDIA Ampere and Hopper architectures. 
- **TensorFloat-32 (TF32):** Enabled by default in `sweep.py` via `torch.backends.cuda.matmul.allow_tf32 = True`.
- **BFloat16:** The training engine automatically detects H100/A100 support and utilizes `torch.amp.autocast` with `dtype=torch.bfloat16`.

## 3. Running the Systematic Sweep
To reproduce the scaling results for Dyck-N languages, execute the following command from the parent directory:

```bash
python sweep.py --sizes 32 64 128 256 --repeats 5 --epochs 40 --outfile dyck_sweep_results.json
```

### Parameters:
- `--sizes`: Defines the sequence lengths.
- `--repeats`: Number of independent trials for statistical significance.
- `--epochs`: Maximum epochs per curriculum stage.
- `--d_vecs`: Hidden vector capacity (e.g., 2 4 6 8).

## 4. Output
The script generates:
1. `dyck_sweep_results.json`: Raw MCC data and parameter counts.
2. `dyck_sweep_plots.png`: Visual comparison of CGA vs. Standard Transformer scaling and Efficiency Ratios.
