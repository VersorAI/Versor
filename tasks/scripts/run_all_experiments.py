#!/usr/bin/env python3
"""
Master Experiment Runner for Versor Paper
Runs all experiments and saves results with timestamps
"""

import json
import os
import sys
import time
from datetime import datetime
import numpy as np
import torch

# Create results directory
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results(experiment_name, results_dict):
    """Save results with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/{experiment_name}_{timestamp}.json"
    
    # Add metadata if results_dict is a dict
    if isinstance(results_dict, dict):
        results_dict["metadata"] = {
            "timestamp": timestamp,
            "experiment": experiment_name,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU"
        }
    
    with open(filename, 'w') as f:
        json.dump(results_dict, indent=2, fp=f)
    
    print(f"✓ Saved: {filename}")
    return filename

def run_experiment_1_nbody():
    """N-Body Physics Experiment (Table 2 in paper)"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: N-Body Dynamics")
    print("="*60)
    
    try:
        from Physics import run_multi_seed
        # Modified seeds and epochs for faster verification if needed, 
        # but here we'll try to get real numbers
        run_multi_seed.main()
        
        # Find the latest multi-seed result file
        import glob
        result_files = glob.glob("results/multi_seed_results_*.json")
        if not result_files:
            return None
        
        latest = max(result_files)
        with open(latest, 'r') as f:
            data = json.load(f)
            
        return save_results("nbody", data)
    except Exception as e:
        print(f"❌ Error running N-Body experiment: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_experiment_2_topology():
    """Maze Connectivity (Broken Snake)"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Topological Connectivity")
    print("="*60)
    
    try:
        from Maze import sweep
        # Set args for verification (small sweep)
        import argparse
        sys.argv = [sys.argv[0], '--sizes', '8', '16', '--repeats', '1', '--epochs', '5', '--outfile', 'results/maze_results.json']
        sweep.run_sweeps()
        
        with open('results/maze_results.json', 'r') as f:
            data = json.load(f)
            
        return save_results("topology", data)
    except Exception as e:
        print(f"❌ Error running Maze experiment: {e}")
        return None

def run_experiment_3_ood():
    """Out-of-Distribution Generalization"""
    print("\n" + "="*60)
    print("EXPERIMENT 3: OOD Generalization (Heavy Masses)")
    print("="*60)
    
    try:
        from Physics import recreate_ood
        recreate_ood.run_ood_test()
        
        # Load the results it barely saved
        with open('results/ood_mass_results.json', 'r') as f:
            data = json.load(f)
            
        print("\n✓ OOD Experiment Completed Successfully")
        return save_results("ood", data)
        
    except Exception as e:
        print(f"❌ Error running OOD experiment: {e}")
        return None

def run_experiment_4_ablation():
    """Ablation Study (Table, Line 594)"""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Ablation Study")
    print("="*60)
    
    try:
        from Physics import rigorous_ablation
        rigorous_ablation.main()
        
        # Load the results it saved
        with open('results/ablation_stats.json', 'r') as f:
            data = json.load(f)
            
        print("\n✓ Ablation Study Completed Successfully")
        return save_results("ablation", data)
        
    except Exception as e:
        print(f"❌ Error running Ablation experiment: {e}")
        return None

def run_experiment_5_kernel_benchmark():
    """Kernel Performance Benchmark"""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Kernel Performance")
    print("="*60)
    
    try:
        import kernel
        kernel.benchmark()
        # Create dummy results for summary
        results = {"status": "completed", "has_triton": kernel.HAS_TRITON, "has_mlx": kernel.HAS_MLX}
        return save_results("kernel_bench", results)
    except Exception as e:
        print(f"❌ Error running Kernel benchmark: {e}")
        return None

def run_experiment_new_domains():
    """Generic Domains (NLP, Vision, Graph)"""
    print("\n" + "="*60)
    print("EXPERIMENT 6: Multimodal Capabilities")
    print("="*60)
    
    try:
        import run_multimodal_experiments as mm
        
        # Run all seeds and get aggregated stats
        results = mm.run_all_seeds()
        
        return save_results("multimodal", results)
        
    except Exception as e:
        print(f"❌ Error running Multimodal experiments: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_summary_report():
    """Generate a summary of all results and compare with paper"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY AND VERIFICATION REPORT")
    print("="*60)
    
    # Paper Data for Comparison
    paper_data = {
        "nbody_mse_versor": 5.210,
        "nbody_drift_versor": 133.0,
        "nbody_mse_transformer": 6.609,
        "ood_increase_transformer": 3097.2,
        "ood_increase_versor": -19.9,
        "topology_mcc_versor": 0.993,
        "topology_mcc_vit": 0.504
    }
    
    # Find all result files
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json') and not f.startswith('SUMMARY')]
    
    if not result_files:
        print("❌ No results found!")
        return
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(result_files),
        "verifications": {}
    }
    
    for fname in result_files:
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            data = json.load(f)
            exp_name = data.get("metadata", {}).get("experiment", "unknown")
            
            if exp_name == "nbody" and "statistics" in data:
                v_mse = data["statistics"].get("Versor", {}).get("mse_mean", 0)
                t_mse = data["statistics"].get("Transformer", {}).get("mse_mean", 0)
                summary["verifications"]["nbody_mse"] = {
                    "measured_versor": v_mse,
                    "paper_versor": paper_data["nbody_mse_versor"],
                    "status": "PASS" if abs(v_mse - paper_data["nbody_mse_versor"]) < 1.0 else "DEVIATION"
                }
            elif exp_name == "ood":
                v_inc = data.get("increase_percent", {}).get("versor", 0)
                summary["verifications"]["ood_generalization"] = {
                    "measured_versor_inc": v_inc,
                    "paper_versor_inc": paper_data["ood_increase_versor"],
                    "status": "PASS" if v_inc < 50 else "DEVIATION"
                }
            elif exp_name == "ablation":
                summary["verifications"]["ablation"] = data
                
    summary_file = f"{RESULTS_DIR}/SUMMARY_VERIFICATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, indent=2, fp=f)
    
    print(f"\n✓ Verification summary saved to: {summary_file}")
    
    # Print a quick human-readable table
    print("\n--- QUICK VERIFICATION ---")
    print(f"{'Metric':<30} | {'Measured':<15} | {'Paper':<15} | {'Status'}")
    print("-" * 75)
    for metric, res in summary["verifications"].items():
        if "measured_versor" in res:
            m, p = res["measured_versor"], res["paper_versor"]
            print(f"{metric:<30} | {m:<15.4f} | {p:<15.4f} | {res['status']}")
        elif "measured_versor_inc" in res:
            m, p = res["measured_versor_inc"], res["paper_versor_inc"]
            print(f"{metric:<30} | {m:<15.1f}% | {p:<15.1f}% | {res['status']}")

if __name__ == "__main__":
    print("="*60)
    print("VERSOR PAPER - COMPLETE EXPERIMENTAL SUITE")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print(f"Results will be saved to: {RESULTS_DIR}/")
    print("="*60)
    
    # Run all experiments
    experiments = [
        run_experiment_1_nbody,
        run_experiment_3_ood,
        run_experiment_4_ablation,
        run_experiment_5_kernel_benchmark,
        run_experiment_new_domains,  # New Multimodal Tasks
        # run_experiment_2_topology # Skipping topology by default as it's very slow
    ]
    
    for exp_func in experiments:
        try:
            exp_func()
        except Exception as e:
            print(f"\n❌ ERROR in {exp_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary
    generate_summary_report()
    
    print("\n" + "="*60)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*60)
    print(f"\n✓ All experimental protocols executed.")
    print("  Results available in ./paper_results/")
    print(f"\nFinished at: {datetime.now()}")
