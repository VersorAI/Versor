
import os
import subprocess
import sys

def main():
    print("="*60)
    print("GATR EXPERIMENT WRAPPER")
    print("="*60)
    
    cwd = os.getcwd()
    base_dir = os.path.abspath("gatr_results")
    gatr_root = os.path.abspath("gatr")
    
    # Add gatr directory to PYTHONPATH so we can import 'gatr' package
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{gatr_root}:{env.get('PYTHONPATH', '')}"
    
    print(f"Results Dir: {base_dir}")
    print(f"GATr Root: {gatr_root}")
    
    # 1. Generate Data
    print("\n[Step 1] Generating N-Body Data (5 bodies)...")
    # We invoke the script using python.
    # Note: The script is in gatr/scripts/generate_nbody_dataset.py
    gen_script = os.path.join(gatr_root, "scripts", "generate_nbody_dataset.py")
    
    cmd_gen = f"python {gen_script} base_dir={base_dir}"
    print(f"Running: {cmd_gen}")
    
    ret = subprocess.call(cmd_gen, shell=True, env=env)
    if ret != 0:
        print("❌ Data generation failed.")
        return

    # 2. Train Model
    print("\n[Step 2] Training GATr Model...")
    train_script = os.path.join(gatr_root, "scripts", "nbody_experiment.py")
    
    # Arguments to speed up training for quick result
    # steps=2000 should be enough to get a number. (Versor trains for 5 epochs of 200 samples? 1000 steps).
    # But gatr is slow?
    # I'll use 2000 steps.
    
    cmd_train = (
        f"python {train_script} "
        f"base_dir={base_dir} "
        f"training.steps=2000 "
        f"training.batchsize=16 "
        f"data.subsample=1.0 "
        f"run_name=gatr5body "
        f"training.eval_device=cpu" # Use CPU for safety or cuda if available
    )
    # If cuda is available, let it use it? 
    # gatr config defaults to cuda.
    
    print(f"Running: {cmd_train}")
    ret = subprocess.call(cmd_train, shell=True, env=env)
    
    # 3. Harvest Results
    print("\n[Step 3] Harvesting Results...")
    metrics_path = os.path.join(base_dir, "experiments", "nbody", "gatr5body", "metrics")
    
    if os.path.exists(metrics_path):
        print(f"Found metrics at: {metrics_path}")
        with open(metrics_path, 'r') as f:
            content = f.read()
            print(content)
            
            # Parse MSE
            # file content format: 'key: value\n...'
            try:
                for line in content.splitlines():
                    if "final_validation_loss" in line or "test_loss" in line or "eval_loss" in line:
                         print(f"PARSED RESULT: {line}")
            except:
                pass
    else:
        print("❌ Metrics file not found.")

if __name__ == "__main__":
    main()
