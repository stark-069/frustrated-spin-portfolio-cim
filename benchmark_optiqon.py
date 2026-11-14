# File: benchmark_optiqon.py
import json
import os
import numpy as np
from cim_cloud import CIMClient, DEFAULT_BASE_URL
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

from dotenv import load_dotenv
import sys
load_dotenv()


def matrix_to_qubo_dict(Q):
    n = Q.shape[0]
    out = {}
    for i in range(n):
        for j in range(i, n):
            v = Q[i, j] + (Q[j, i] if i != j else 0.0)
            if v != 0.0:
                out[(i, j)] = float(v)
    return out


def run_optiqon_suite(stress_5000=False):
    api_key = os.getenv("CIM_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set CIM_API_KEY (see .env.example).")
    client = CIMClient(api_key=api_key, base_url=os.getenv("CIM_BASE_URL", DEFAULT_BASE_URL))

    print("\n" + "=" * 60)
    print("   RUNNING BENCHMARK: OPTIQON CIM (cim_cloud SDK)")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)
    ret_50, cov_50, _ = moments(load_close(TICKERS_50, "sp500_50", period="1y"))
    results = {"solver": "Quanfluence Optiqon CIM", "tests": {}}

    # 50-asset suite. Reported time = hardware execution time (excludes cloud/API transmission).
    print("\n[*] Executing 50-Asset Suite on CIM...")
    res_50 = {}
    for prof_name, gamma in PROFILES.items():
        eng = HighPerformanceQUBOEngine(num_assets=50, bits_per_asset=4, risk_gamma=gamma)
        Q_norm, _, _, _ = eng.build_qubo_from_real_data(ret_50, cov_50)
        try:
            job = client.submit_qubo(matrix_to_qubo_dict(Q_norm), label=f"50 assets {prof_name}")
            res = job.wait_for_completion()
            t, e = res.execution_time_ms / 1000.0, float(res.energy)
            res_50[prof_name] = {"time": t, "energy": e, "status": "Success"}
            print(f"    - [{prof_name}] Time: {t:.4f}s | Energy: {e:.4f}")
        except Exception as exc:
            res_50[prof_name] = {"time": None, "energy": None, "status": f"Error: {exc}"}
            print(f"    - [{prof_name}] Error: {exc}")
    results["tests"]["50_assets"] = res_50

    # Optional 5,000-asset payload-ceiling attempt (Sec. IV-C); needs tens of GB of RAM.
    if stress_5000:
        print("\n[*] Attempting 5,000-asset payload (expected to fail on serialization)...")
        np.random.seed(42)
        ret_5000 = np.random.uniform(0.05, 0.25, 5000).astype(np.float64)
        A = np.random.randn(5000, 5000).astype(np.float64)
        cov_5000 = (np.dot(A, A.T) * 0.005).astype(np.float64)
        res_5000 = {}
        for prof_name, gamma in PROFILES.items():
            try:
                eng = HighPerformanceQUBOEngine(num_assets=5000, bits_per_asset=4, risk_gamma=gamma)
                Q_norm, _, _, _ = eng.build_qubo_from_real_data(ret_5000, cov_5000)
                res = client.submit_qubo(matrix_to_qubo_dict(Q_norm)).wait_for_completion()
                res_5000[prof_name] = {"time": res.execution_time_ms / 1000.0, "energy": float(res.energy), "status": "Success"}
            except Exception as exc:
                res_5000[prof_name] = {"time": None, "energy": None, "status": f"Failed: {type(exc).__name__}"}
        results["tests"]["5000_assets"] = res_5000

    with open("reports/results_optiqon.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[SUCCESS] Optiqon benchmark written to 'reports/results_optiqon.json'")


if __name__ == "__main__":
    run_optiqon_suite(stress_5000="--stress-5000" in sys.argv)
