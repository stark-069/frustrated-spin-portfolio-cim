# File: benchmark_dwave.py
import json
import os
import time
import numpy as np
from dotenv import load_dotenv
from market_data import load_close, moments
import dimod
from dwave.system import LeapHybridSampler
from quantum_core.qubo_engine import HighPerformanceQUBOEngine

TICKERS_50 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'BRK-B', 'META', 'UNH', 'XOM', 'JNJ',
    'JPM', 'PG', 'V', 'LLY', 'HD', 'MA', 'CVX', 'ABBV', 'MRK', 'AVGO',
    'PEP', 'KO', 'COST', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT', 'SBUX', 'ORCL',
    'WMT', 'DIS', 'BMY', 'SCHW', 'AMGN', 'IBM', 'GE', 'CAT', 'GILD', 'INTC',
    'RTX', 'HON', 'AMAT', 'BKNG', 'T', 'PM', 'LOW', 'PLD', 'GS', 'MS'
]

PROFILES = {
    "Pure_Alpha": 0.1,
    "Aggressive_Growth": 0.5,
    "Balanced_Standard": 2.0,
    "Moderate_Conservative": 5.0,
    "Conservative_Yield": 8.0,
    "Min_Volatility": 15.0
}

load_dotenv()


def run_dwave_suite():
    print("\n" + "═"*60)
    print("   RUNNING BENCHMARK: D-WAVE LEAP HYBRID")
    print("═"*60)

    token = os.environ.get("DWAVE_API_TOKEN", "").strip()
    if not token:
        print("[!] ERROR: DWAVE_API_TOKEN is missing from environment.")
        return

    os.makedirs("reports", exist_ok=True)
    ret_50, cov_50, _ = moments(load_close(TICKERS_50, "sp500_50", period="1y"))

    results = {"solver": "D-Wave Leap Hybrid", "tests": {}}

    # 1. 50-Asset Suite across 6 profiles
    print("\n[*] Executing 50-Asset Suite across 6 Profiles...")
    res_50 = {}
    sampler = LeapHybridSampler(token=token)
    
    for prof_name, gamma in PROFILES.items():
        eng = HighPerformanceQUBOEngine(num_assets=50, bits_per_asset=4, risk_gamma=gamma)
        Q_norm, _, _, _ = eng.build_qubo_from_real_data(ret_50, cov_50)
        bqm = dimod.BinaryQuadraticModel(Q_norm, "BINARY")
        
        t0 = time.time()
        try:
            sampleset = sampler.sample(bqm)
            elapsed = time.time() - t0
            energy = float(sampleset.first.energy)
            res_50[prof_name] = {"time": elapsed, "energy": energy, "status": "Success"}
            print(f"    - [{prof_name}] Time: {elapsed:.2f}s | Energy: {energy:.4f}")
        except Exception as e:
            res_50[prof_name] = {"time": None, "energy": None, "status": f"Failed: {str(e)}"}
            print(f"    - [{prof_name}] Failed: {e}")

    results["tests"]["50_assets"] = res_50

    # 2. 5000-Asset Stress Test (Disqualification Proof)
    print("\n[*] Executing 5000-Asset Payload Rejection Test...")
    res_5000 = {}
    np.random.seed(42)
    ret_5000 = np.random.uniform(0.05, 0.25, 5000).astype(np.float64)
    A = np.random.randn(5000, 5000).astype(np.float64)
    cov_5000 = (np.dot(A, A.T) * 0.005).astype(np.float64)
    
    eng_5000 = HighPerformanceQUBOEngine(num_assets=5000, bits_per_asset=4, risk_gamma=2.0)
    Q_norm_5000, _, _, _ = eng_5000.build_qubo_from_real_data(ret_5000, cov_5000)
    
    try:
        bqm_5000 = dimod.BinaryQuadraticModel(Q_norm_5000, "BINARY")
        sampler.sample(bqm_5000)
    except Exception as e:
        err_msg = str(e)
        print(f"    [D-WAVE DISQUALIFIED]: {err_msg}")
        for prof_name in PROFILES.keys():
            res_5000[prof_name] = {"time": None, "energy": None, "status": "Disqualified: >200M Biases Limit Exceeded"}

    results["tests"]["5000_assets"] = res_5000

    with open("reports/results_dwave.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[SUCCESS] D-Wave benchmark written to 'reports/results_dwave.json'")

if __name__ == "__main__":
    run_dwave_suite()