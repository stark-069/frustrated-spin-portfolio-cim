# File: benchmark_classical.py
import json
import os
import time
import numpy as np
from market_data import load_close, moments
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

def run_classical_suite():
    print("\n" + "═"*60)
    print("   RUNNING BENCHMARK: CLASSICAL CPU SOLVER")
    print("═"*60)
    
    os.makedirs("reports", exist_ok=True)
    
    # Data 50
    ret_50, cov_50, _ = moments(load_close(TICKERS_50, "sp500_50", period="1y"))

    # Data 5000 (Synthetic)
    engine_5000 = HighPerformanceQUBOEngine(num_assets=5000, bits_per_asset=4)
    np.random.seed(42)
    ret_5000 = np.random.uniform(0.05, 0.25, 5000).astype(np.float64)
    A = np.random.randn(5000, 5000).astype(np.float64)
    cov_5000 = (np.dot(A, A.T) * 0.005).astype(np.float64)

    results = {"solver": "Classical CPU", "tests": {}}

    # Test 1 & 3: 50 Assets across 6 profiles
    print("\n[*] Running 50-Asset Suite across 6 Profiles...")
    res_50 = {}
    for prof_name, gamma in PROFILES.items():
        eng = HighPerformanceQUBOEngine(num_assets=50, bits_per_asset=4, risk_gamma=gamma)
        Q_norm, _, _, _ = eng.build_qubo_from_real_data(ret_50, cov_50)
        state, energy, time_taken = eng.simulated_annealing(Q_norm, num_reads=250, num_sweeps=100)
        res_50[prof_name] = {"time": time_taken, "energy": energy, "status": "Success"}
        print(f"    - [{prof_name}] Time: {time_taken:.2f}s | Energy: {energy:.4f}")
    results["tests"]["50_assets"] = res_50

    # Test 2 & 4: 5000 Assets across 6 profiles
    print("\n[*] Running 5000-Asset Suite across 6 Profiles...")
    res_5000 = {}
    for prof_name, gamma in PROFILES.items():
        eng = HighPerformanceQUBOEngine(num_assets=5000, bits_per_asset=4, risk_gamma=gamma)
        try:
            Q_norm, _, _, _ = eng.build_qubo_from_real_data(ret_5000, cov_5000)
            state, energy, time_taken = eng.simulated_annealing(Q_norm, num_reads=250, num_sweeps=100)
            res_5000[prof_name] = {"time": time_taken, "energy": energy, "status": "Success"}
        except MemoryError:
            res_5000[prof_name] = {"time": None, "energy": None, "status": "Failed: Out-of-Memory"}
            print(f"    - [{prof_name}] Out-of-Memory")
            continue
        print(f"    - [{prof_name}] Time: {time_taken:.2f}s | Energy: {energy:.4f}")
    results["tests"]["5000_assets"] = res_5000

    with open("reports/results_classical.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[SUCCESS] Classical benchmark written to 'reports/results_classical.json'")

if __name__ == "__main__":
    run_classical_suite()