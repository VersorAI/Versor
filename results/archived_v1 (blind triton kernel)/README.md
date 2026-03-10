# Experimental Results & Verification

This directory contains the raw data and verification reports for the Versor project benchmarks.

## Data Files → Paper Claims

| File | Paper Location | Notes |
|---|---|---|
| `multi_seed_results_20260130.json` | **Table 1** (N-Body MSE/Drift, 8 models) | 5-seed mean ± std |
| `ood_mass_results.json` | **Table 2 OOD row** (+1933.7% / –63.9%) | Separate 30-step eval protocol |
| `curriculum_stats.json` | **Appendix Table `tab:curriculum`** (L=50/100/150 gaps) | Reproducible: see `curriculum_test_20260303_074824.json` |
| `ablation_replacement_stats.json` | **Appendix Table `tab:ablation_replacement`** | Source: `tasks/nbody/results/ablation_stats_rigorous.json` |
| `ablation_stats.json` | **Section 7.4 (knockout ablation)** | Different experiment from replacement table (w/o Norm → Diverged) |
| `scaling_results.json` | **Figure 3 (scaling plot)** | Latency vs sequence length |
| `multi-channel-stats.json` | **Table 1, Versor Multi-Channel row** | MSE=3.067, single seed, 30 epochs |
| `gapu_simulation_results.json` | **Appendix GAPU Table** | Simulated architecture comparison |
| `cifar10_results_*.json` | Multimodal vision task | |
| `latency_verification_report.json` | **Table 1 Latency column** | See note below |