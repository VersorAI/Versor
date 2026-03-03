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

## Latency Measurement Note

All latency values in the paper are **per-step CPU inference** (batch=1, seq=50, N=5 particles) on **Apple M4 CPU**. The values in the paper table (Versor 1.54/1.05 ms, Transformer 1.10 ms) were measured at submission time and may differ slightly from fresh re-runs due to system load and thermal state. The `latency_verification_report.json` documents both the paper values and the latest re-run for transparency.

**Transformer Large latency** (12.32 ms in paper) was measured in an earlier hardware run; the `transformer_large_bench.json` shows 24.88 ms from a later isolated benchmark. The paper value has been updated to 24.88 ms.

## Ablation Table Correspondence

The paper contains **two separate ablation experiments**:

1. **Knockout ablation** (`ablation_stats.json`): "w/o Manifold Norm → Diverged (NaN)", "w/o Recursive Rotor → 3.44". These appear in the main text discussion (Section 7.4).

2. **Replacement ablation** (`ablation_replacement_stats.json`): Strip LayerNorm → Manifold Norm (8.7 MSE), Strip MLP → Clifford Layer (19.2 MSE). This is the source for Appendix Table `tab:ablation_replacement`. Script: `tasks/nbody/true_ablation.py`.

## GATr Benchmark

The paper Table 1 reports GATr MSE = 8.32 ± 1.80 (downscaled to ≈0.1M params). The `tasks/nbody/results/gatr_stats.json` from `run_gatr.py` reports MSE = 10.68 (different seed/training config). The paper figure was obtained using an earlier training run with 3 seeds matching the exact multi-seed protocol. **The GATr result in the paper comes from a run that required the external `gatr` library** (not included in this repo) and was run separately. A footnote in the paper marks GATr with ‡ indicating it was downscaled for fair comparison.

## Verification Note

All latency measurements reported in the paper were verified on standard CPU hardware (Apple M4). The `ablation_replacement_stats.json` provides the definitive source for the ablation table. The `curriculum_stats.json` values have been independently reproduced (see `tasks/nbody/curriculum_test_20260303_074824.json`).
