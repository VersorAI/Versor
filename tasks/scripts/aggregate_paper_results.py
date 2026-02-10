import json
import os
import numpy as np
import matplotlib.pyplot as plt

def generate_final_report():
    print("Generating Final Paper Report...")
    results_dir = "/Users/mac/Desktop/Versor/results"
    
    # 1. Load Multi-Seed Results
    multi_seed_files = [f for f in os.listdir(results_dir) if f.startswith("multi_seed_results") and f.endswith(".json")]
    if multi_seed_files:
        latest_ms = sorted(multi_seed_files)[-1]
        with open(os.path.join(results_dir, latest_ms), "r") as f:
            ms_data = json.load(f)
        
        print(f"\n--- N-BODY ROLLOUT RESULTS (From {latest_ms}) ---")
        for model, stats in ms_data["statistics"].items():
            print(f"{model:20} | MSE: {stats['mse_mean']:.4f} ± {stats['mse_std']:.4f} | Drift: {stats['drift_mean']:.1f}%")
            
    # 2. Load Variable-N Results
    var_n_files = [f for f in os.listdir(results_dir) if f.startswith("variable_n_results") and f.endswith(".json")]
    if var_n_files:
        latest_vn = sorted(var_n_files)[-1]
        with open(os.path.join(results_dir, latest_vn), "r") as f:
            vn_data = json.load(f)
            
        print(f"\n--- ZERO-SHOT N-GENERALIZATION (From {latest_vn}) ---")
        for model, m_data in vn_data["models"].items():
            if "seeds" in m_data:
                # Aggregate across seeds
                n_res = {}
                for seed, s_data in m_data["seeds"].items():
                    if "test_n" in s_data:
                        for n, res in s_data["test_n"].items():
                            if n not in n_res: n_res[n] = []
                            if res["success"]: n_res[n].append(res["mse"])
                
                print(f"{model}:")
                for n in sorted(n_res.keys(), key=int):
                    if n_res[n]:
                        print(f"  N={n}: {np.mean(n_res[n]):.4f}")
                    else:
                        print(f"  N={n}: FAILED")

if __name__ == "__main__":
    generate_final_report()
