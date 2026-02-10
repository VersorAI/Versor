import sys
import os
import json
import numpy as np
import torch
from datetime import datetime

# Add paths
sys.path.append("Physics")
from train import train

RESULTS_DIR = "paper_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_main_results():
    seeds = [42, 123, 456, 789, 1011] + [i for i in range(1000, 1015)]
    model_types = ['std', 'versor', 'gns', 'hnn', 'egnn']
    
    all_results = []
    
    filename = f"{RESULTS_DIR}/main_results_20_seeds.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_results = json.load(f)
        done_seeds = [r['seed'] for r in all_results]
        seeds = [s for s in seeds if s not in done_seeds]
    
    for seed in seeds:
        print(f"\n>>> Running Core Seeds: Seed {seed} (Main Experiments)")
        try:
            res = train(seed=seed, model_types=model_types)
            all_results.append({"seed": seed, "metrics": res})
            # Save progress
            with open(filename, 'w') as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            
    return all_results

def run_ablation_results():
    seeds = [42, 123, 456, 789, 1011]
    # Ablation configs are a bit harder to run with current train() since it doesn't expose architecture flags
    # We would need to modify train() more or have multiple model classes.
    # Looking at train.py, it uses VersorRotorRNN which is one of the ablation points.
    # I'll modify train.py or models.py if needed, but for now let's see.
    pass

if __name__ == "__main__":
    run_main_results()
