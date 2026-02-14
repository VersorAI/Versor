# Experimental Results & Verification

This directory contains the raw data and verification reports for the Versor project benchmarks.

## Data Files

- `scaling_results.json`: Scaling performance data for latency and memory usage across sequence lengths.
- `latency_verification_report.json`: Per-step CPU latency measurements for all models, comparing C++ accelerated modes with Python fallbacks.
- `verification_report_20260202_181822.json`: Parameter counts and OOD generalization stats.
- `cifar10_results_*.json`: Benchmarks for multimodal vision tasks.

## Verification Note

All latency measurements reported in the paper were verified on standard CPU hardware. Early development logs (previously stored as `.log` files) tracked internal optimization progress, including the transition from Python-based recurrence to a high-performance C++ core. The current JSON reports reflect the final, reproducible measurements used in the manuscript.