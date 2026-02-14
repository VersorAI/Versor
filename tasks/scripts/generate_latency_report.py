
import torch
import time
import json
import os
import sys

# Append project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tasks.nbody.models import (
    StandardTransformer, VersorRotorRNN, MultiChannelVersor, HamiltonianVersorNN
)

def benchmark_latency():
    device = "cpu"
    N = 5
    L = 50
    iters = 100
    
    results = {}
    
    model_configs = {
        "Transformer": lambda: StandardTransformer(n_particles=N),
        "Versor": lambda: VersorRotorRNN(n_particles=N),
        "Versor-4ch": lambda: MultiChannelVersor(n_particles=N, n_channels=4),
        "Ham-Versor": lambda: HamiltonianVersorNN(n_particles=N)
    }
    
    # Test with C++ Enabled
    print("Benchmarking with C++ Accelerated: ON")
    if "VERSOR_NO_CPP" in os.environ:
        del os.environ["VERSOR_NO_CPP"]
    
    results["cpp_enabled"] = {}
    for name, config in model_configs.items():
        model = config().to(device)
        model.eval()
        x = torch.randn(1, L, N, 6).to(device)
        # Warmup
        for _ in range(5): _ = model(x)
        
        t_start = time.time()
        for _ in range(iters):
            _ = model(x)
        dt = (time.time() - t_start) / iters * 1000
        results["cpp_enabled"][name] = dt
        print(f"  {name}: {dt:.3f} ms")

    # Test with C++ Disabled (Python Fallback)
    # Must use subprocess to ensure VERSOR_NO_CPP is set BEFORE module import
    print("\\nBenchmarking with C++ Accelerated: OFF (Python fallback)")
    
    import subprocess
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    fallback_script = f'''
import torch
import time
import os
import sys
import json

# Set env var BEFORE any imports
os.environ["VERSOR_NO_CPP"] = "1"

sys.path.insert(0, "{os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))}")

from tasks.nbody.models import VersorRotorRNN, MultiChannelVersor, HamiltonianVersorNN

device = "cpu"
N = 5
L = 50
iters = 50

results = {{}}

model_configs = {{
    "Versor": lambda: VersorRotorRNN(n_particles=N),
    "Versor-4ch": lambda: MultiChannelVersor(n_particles=N, n_channels=4),
    "Ham-Versor": lambda: HamiltonianVersorNN(n_particles=N)
}}

for name, config in model_configs.items():
    model = config().to(device)
    model.eval()
    x = torch.randn(1, L, N, 6).to(device)
    # Warmup
    for _ in range(3): _ = model(x)
    
    t_start = time.time()
    for _ in range(iters):
        _ = model(x)
    dt = (time.time() - t_start) / iters * 1000
    results[name] = dt
    print(f"{{name}}: {{dt:.3f}} ms")

with open("{tmp_path}", "w") as f:
    json.dump(results, f)
'''
    
    try:
        proc = subprocess.run(
            [sys.executable, "-c", fallback_script],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(proc.stdout)
        if proc.returncode != 0:
            raise Exception(f"Subprocess failed: {proc.stderr}")
        
        with open(tmp_path, 'r') as f:
            results["python_fallback"] = json.load(f)
        
        os.unlink(tmp_path)
            
    except Exception as e:
        print(f"  WARNING: Python fallback measurement failed: {e}")
        print(f"  stderr: {proc.stderr if 'proc' in locals() else 'N/A'}")
        print(f"  Using NULL values - MUST BE MEASURED MANUALLY")
        results["python_fallback"] = {
            "Versor": None,
            "Versor-4ch": None,
            "Ham-Versor": None
        }
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Save to file
    with open("latency_verification_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nReport saved to latency_verification_report.json")

if __name__ == "__main__":
    benchmark_latency()
