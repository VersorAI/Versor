# Experimental Results & Verification

This directory contains the raw data and verification reports for the Versor project benchmarks.

## Data Files

- `multi_seed_results_20260130.json`: **Source for Table 1**. N-Body Dynamics stats (MSE 5.210) and Energy Drift (133%).
- `ood_mass_results.json`: **Source for Table 2 (OOD Mass)**. Verifies the -63.9% error reduction on heavy masses.
- `scaling_results.json`: Scaling performance data for latency and memory usage across sequence lengths (Figure 3).
- `latency_verification_report.json`: Per-step CPU latency measurements for all models.
- `ablation_stats.json`: Source for quantitative ablation study (Table 4).
- `cifar10_results_*.json`: Benchmarks for multimodal vision tasks.

## Verification Note

All latency measurements reported in the paper were verified on standard CPU hardware. Early development logs (previously stored as `.log` files) tracked internal optimization progress, including the transition from Python-based recurrence to a high-performance C++ core. The current JSON reports reflect the final, reproducible measurements used in the manuscript.
