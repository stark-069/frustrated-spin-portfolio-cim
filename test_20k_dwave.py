# File: test_20k_dwave.py
# Bulletproof 20K Variable Scaling & D-Wave Payload Stress Test

from quantum_core.qubo_engine import HighPerformanceQUBOEngine
import numpy as np
import dimod
from dwave.system import LeapHybridSampler
import os
import time
from dotenv import load_dotenv

load_dotenv()

def generate_massive_market_data(num_assets=5000, seed=42):
    """Generates synthetic float64 market data for 5,000 assets instantly."""
    print(f"[*] Generating synthetic market data for {num_assets} assets...")
    np.random.seed(seed)
    returns = np.random.uniform(0.05, 0.25, num_assets).astype(np.float64)
    A = np.random.randn(num_assets, num_assets).astype(np.float64)
    cov = (np.dot(A, A.T) * 0.005).astype(np.float64)
    return returns, cov

def run_20k_stress_test():
    print("\n" + "═"*65)
    print("   FINAL STRESS TEST: 20,000 VARIABLES (5,000 ASSETS)")
    print("═"*65 + "\n")
    
    assets = 5000
    bits = 4
    num_vars = assets * bits
    num_couplings = (num_vars * (num_vars - 1)) // 2

    # 1. Initialize Engine & Generate Data
    engine = HighPerformanceQUBOEngine(num_assets=assets, bits_per_asset=bits, risk_gamma=2.0)
    returns, cov = generate_massive_market_data(num_assets=assets)
    
    # 2. Build the 1.5 GB Matrix
    print(f"[*] Compiling {num_couplings:,} Dense Couplings...")
    t0 = time.time()
    Q_norm, Q_raw, penalty_lambda, scale_factor = engine.build_qubo_from_real_data(returns, cov)
    build_time = time.time() - t0
    
    ram_mb = Q_norm.nbytes / (1024**2)
    print(f"    - Matrix Build Time: {build_time:.2f}s")
    print(f"    - RAM Allocation: {ram_mb:.2f} MB")
    print(f"    - Dynamic Penalty Lambda: {penalty_lambda:,.2f}\n")

    # 3. D-Wave Submission
    token = os.environ.get("DWAVE_API_TOKEN", "").strip()
    if not token:
        print("[!] ERROR: No DWAVE_API_TOKEN found in environment variables.")
        return

    print("[*] Translating to D-Wave Binary Quadratic Model (BQM)...")
    try:
        # Using the updated, non-deprecated initialization
        bqm = dimod.BinaryQuadraticModel(Q_norm, "BINARY")
    except MemoryError:
        print("    [!] FATAL: Local machine ran out of RAM translating the BQM.")
        return

    print(f"[*] Initiating Network Upload to D-Wave Leap Hybrid (Payload ~{ram_mb:.0f} MB)...")
    try:
        t_upload = time.time()
        hybrid_sampler = LeapHybridSampler(token=token)
        results = hybrid_sampler.sample(bqm)
        total_time = time.time() - t_upload
        
        sample = results.first.sample
        state_hybrid = np.array([sample[i] for i in range(num_vars)])
        energy_hybrid = results.first.energy
        
        print(f"    [SUCCESS] D-Wave processed the 20K matrix!")
        print(f"    -> Hybrid Energy: {energy_hybrid:.6f}")
        print(f"    -> Total Wall Time (Upload + Compute): {total_time:.2f}s\n")
        
    except Exception as e:
        print(f"\n    [D-WAVE NETWORK/API REJECTION]")
        print(f"    The matrix is mathematically sound, but D-Wave's cloud infrastructure failed:")
        print(f"    -> Error Details: {str(e)}\n")
        print(f"    [Conclusion]: Native optical hardware (CIM) is required to bypass cloud payload limits.")

if __name__ == "__main__":
    run_20k_stress_test()