# File: real_market_test.py
import os
from dotenv import load_dotenv
from market_data import load_prices, moments
import pandas as pd
import numpy as np
from cim_cloud import CIMClient, DEFAULT_BASE_URL
from quantum_core.qubo_engine import HighPerformanceQUBOEngine

load_dotenv()

# 1. Define the Universes
US_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'JPM',
    'V', 'PG', 'MA', 'XOM', 'UNH', 'HD', 'CVX', 'ABBV', 'MRK', 'PEP'
]

INDIA_TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
    'SBIN.NS', 'ITC.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
    'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATASTEEL.NS', 'HCLTECH.NS',
    'WIPRO.NS', 'POWERGRID.NS', 'ULTRACEMCO.NS'
]

def fetch_and_prepare_data(tickers, start_date='2024-01-01', end_date='2026-01-01'):
    print(f"[*] Downloading data for {len(tickers)} assets (sequential ingestion)...")
    return moments(load_prices(tickers, start_date, end_date))

def decode_weights(binary_state, num_assets, bits_per_asset=4):
    K = (2**bits_per_asset) - 1
    bit_weights = np.array([2**k / K for k in range(bits_per_asset)])
    
    # Unpack dictionary from the CIM hardware into a sequential list
    if isinstance(binary_state, dict):
        # JSON keys are strings, so we check for string indices as well
        binary_state = [binary_state.get(i, binary_state.get(str(i), 0)) for i in range(num_assets * bits_per_asset)]
    
    state_matrix = np.array(binary_state).reshape((num_assets, bits_per_asset))
    return np.dot(state_matrix, bit_weights)

def matrix_to_qubo_dict(Q_matrix):
    Q_dict = {}
    N = Q_matrix.shape[0]
    for i in range(N):
        for j in range(i, N):
            val = Q_matrix[i, j]
            if i != j:
                val += Q_matrix[j, i]
            if val != 0.0:
                Q_dict[(i, j)] = val
    return Q_dict

def run_case_study_hardware(market_name, tickers, client):
    print(f"\n==================================================")
    print(f"   HARDWARE CASE STUDY: {market_name.upper()} MARKETS")
    print(f"==================================================")
    
    returns, cov, valid_tickers = fetch_and_prepare_data(tickers)
    num_assets = len(valid_tickers)
    
    if num_assets == 0:
        print("[!] No valid assets downloaded. Aborting.")
        return
        
    engine = HighPerformanceQUBOEngine(num_assets=num_assets, bits_per_asset=4, risk_gamma=2.5)
    current_weights = np.array([1.0 / num_assets] * num_assets)
    c_penalty = 0.005 # 0.5% market impact fee
    
    print("[*] Building QUBO Matrix...")
    Q_norm, _, _, _ = engine.build_qubo_from_real_data(
        returns, cov, current_weights=current_weights, c_penalty=c_penalty
    )
    
    Q_dict = matrix_to_qubo_dict(Q_norm)
    
    print(f"[*] Dispatching to physical Optiqon CIM via API...")
    job = client.submit_qubo(Q_dict, label=f"{market_name} Optimization Run")
    result = job.wait_for_completion()
    
    exec_time_sec = result.execution_time_ms / 1000.0
    
    optimal_weights = decode_weights(result.spins, num_assets)
    
    expected_portfolio_return = np.dot(optimal_weights, returns)
    portfolio_variance = np.dot(optimal_weights.T, np.dot(cov, optimal_weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = expected_portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    print(f"\n--- FINANCIAL RESULTS ({market_name}) ---")
    print(f"Hardware Execution Time: {exec_time_sec:.4f}s")
    print(f"Ground State Energy:     {result.energy:.4f}")
    print(f"Budget Used:             {np.sum(optimal_weights):.2f} (Target: 1.00)")
    print(f"Expected Return:         {expected_portfolio_return * 100:.2f}%")
    print(f"Portfolio Volatility:    {portfolio_volatility * 100:.2f}%")
    print(f"Sharpe Ratio:            {sharpe_ratio:.4f}")
    
    print("\n--- TOP ALLOCATIONS ---")
    allocations = [(valid_tickers[i], optimal_weights[i]) for i in range(num_assets) if optimal_weights[i] > 0.01]
    allocations.sort(key=lambda x: x[1], reverse=True)
    
    for ticker, weight in allocations:
        print(f"{ticker:>12}: {weight * 100:>5.1f}%")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    API_KEY = os.getenv("CIM_API_KEY", "").strip()
    if not API_KEY:
        raise SystemExit("Set CIM_API_KEY (see .env.example).")
    client = CIMClient(api_key=API_KEY, base_url=os.getenv("CIM_BASE_URL", DEFAULT_BASE_URL))
    
    run_case_study_hardware("US / International", US_TICKERS, client)
    run_case_study_hardware("Indian (NIFTY)", INDIA_TICKERS, client)