# File: 100assetindia.py
import os
from dotenv import load_dotenv
from market_data import load_prices, moments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from cim_cloud import CIMClient, DEFAULT_BASE_URL
from quantum_core.qubo_engine import HighPerformanceQUBOEngine

load_dotenv()

# The Complete NIFTY 100 Universe
NIFTY_UNIVERSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'BHARTIARTL.NS', 
    'INFY.NS', 'ITC.NS', 'SBIN.NS', 'HINDUNILVR.NS', 'LT.NS', 
    'BAJFINANCE.NS', 'HCLTECH.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 
    'KOTAKBANK.NS', 'ASIANPAINT.NS', 'TITAN.NS', 'NTPC.NS', 'AXISBANK.NS', 
    'ULTRACEMCO.NS', 'ONGC.NS', 'BAJAJFINSV.NS', 'POWERGRID.NS', 'WIPRO.NS', 
    'M&M.NS', 'TATASTEEL.NS', 'LTIM.NS', 'COALINDIA.NS', 'NESTLEIND.NS', 
    'TECHM.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'GRASIM.NS', 'HINDALCO.NS', 
    'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'DRREDDY.NS', 'DIVISLAB.NS', 'CIPLA.NS', 
    'APOLLOHOSP.NS', 'BRITANNIA.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'HEROMOTOCO.NS', 
    'TATACONSUM.NS', 'BPCL.NS', 'SHREECEM.NS', 'PIDILITIND.NS', 'HAVELLS.NS', 
    'DABUR.NS', 'GODREJCP.NS', 'TVSMOTOR.NS', 'CHOLAFIN.NS', 'VEDL.NS', 
    'SIEMENS.NS', 'ICICIPRULI.NS', 'AMBUJACEM.NS', 'INDIGO.NS', 'GAIL.NS', 
    'SRF.NS', 'MCDOWELL-N.NS', 'ICICIGI.NS', 'MUTHOOTFIN.NS', 'BOSCHLTD.NS', 
    'TORNTPHARM.NS', 'PIIND.NS', 'TRENT.NS', 'MARICO.NS', 'BERGEPAINT.NS', 
    'COLPAL.NS', 'BEL.NS', 'HAL.NS', 'LODHA.NS', 'POLYCAB.NS', 
    'JINDALSTEL.NS', 'CGPOWER.NS', 'SHRIRAMFIN.NS', 'TATACOMM.NS', 'MAXHEALTH.NS', 
    'ZOMATO.NS', 'DMART.NS', 'PAYTM.NS', 'NYKAA.NS', 'DLF.NS', 
    'BANKBARODA.NS', 'PNB.NS', 'UNIONBANK.NS', 'IOB.NS', 'CANBK.NS', 
    'IDFCFIRSTB.NS', 'YESBANK.NS', 'INDUSINDBK.NS', 'BANDHANBNK.NS', 'AUBANK.NS', 
    'MOTHERSON.NS', 'CUMMINSIND.NS', 'ASHOKLEY.NS', 'UPL.NS', 'JSWSTEEL.NS'
]

def fetch_and_prepare_data(tickers, start_date='2024-01-01', end_date='2026-01-01'):
    print(f"[*] Downloading data for {len(tickers)} assets (sequential ingestion)...")
    return moments(load_prices(tickers, start_date, end_date))

def decode_weights(binary_state, num_assets, bits_per_asset=4):
    K = (2**bits_per_asset) - 1
    bit_weights = np.array([2**k / K for k in range(bits_per_asset)])
    if isinstance(binary_state, dict):
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

def run_nifty_cost_study():
    print("==================================================")
    print("  NIFTY MARKET TRANSACTION COST SUPREMACY STUDY")
    print("==================================================\n")
    
    returns, cov, valid_tickers = fetch_and_prepare_data(NIFTY_UNIVERSE)
    num_assets = len(valid_tickers)
    
    # Strict safety check to prevent ZeroDivisionError
    if num_assets == 0:
        print("[!] No valid assets downloaded. Check internet or API limit.")
        return
        
    print(f"[*] Real Data Validated: {num_assets} Assets ({num_assets * 4} Variables)\n")
    
    TRANSACTION_COSTS = [0.0, 0.005, 0.02, 0.05] # 0%, 0.5%, 2%, 5%
    API_KEY = os.getenv("CIM_API_KEY", "").strip()
    if not API_KEY:
        raise SystemExit("Set CIM_API_KEY (see .env.example).")
    client = CIMClient(api_key=API_KEY, base_url=os.getenv("CIM_BASE_URL", DEFAULT_BASE_URL))
    engine = HighPerformanceQUBOEngine(num_assets=num_assets, bits_per_asset=4, risk_gamma=2.5)
    
    current_weights = np.array([1.0 / num_assets] * num_assets)
    results = []

    for c_penalty in TRANSACTION_COSTS:
        print(f"---> Simulating Market Impact Cost: {c_penalty * 100:.1f}%")
        
        Q_norm, _, _, _ = engine.build_qubo_from_real_data(
            returns, cov, current_weights=current_weights, c_penalty=c_penalty
        )
        
        # 1. CLASSICAL SOLVER
        print("     [CPU] Running Classical Annealer (250 reads, 100 sweeps)...")
        state_cpu, energy_cpu, time_cpu = engine.simulated_annealing(Q_norm, num_reads=250, num_sweeps=100)
        budget_cpu = np.sum(decode_weights(state_cpu, num_assets))
        
        # 2. OPTIQON CIM
        print("     [CIM] Dispatching to Optiqon API...")
        Q_dict = matrix_to_qubo_dict(Q_norm)
        try:
            job = client.submit_qubo(Q_dict, label=f"Nifty Cost Study {c_penalty*100}%")
            res = job.wait_for_completion()
            time_cim = res.execution_time_ms / 1000.0
            energy_cim = res.energy
            budget_cim = np.sum(decode_weights(res.spins, num_assets))
        except Exception as e:
            print(f"     [!] API Error: {e}")
            time_cim, energy_cim, budget_cim = 0, 0, 0

        print(f"     => CPU Latency: {time_cpu:.3f}s | CIM Latency: {time_cim:.3f}s")
        print(f"     => CPU Energy: {energy_cpu:.2f} | CIM Energy: {energy_cim:.2f}\n")
        
        results.append({
            "Cost (%)": c_penalty * 100,
            "CPU Latency (s)": time_cpu,
            "CIM Latency (s)": time_cim,
            "CPU Energy": energy_cpu,
            "CIM Energy": energy_cim,
            "CPU Budget": budget_cpu,
            "CIM Budget": budget_cim
        })

    # --- OUTPUT TABLE ---
    df = pd.DataFrame(results)
    print("==================================================")
    print("               FINAL STUDY RESULTS")
    print("==================================================")
    print(df.to_markdown(index=False, floatfmt=".3f"))
    
    # --- OUTPUT GRAPH ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(df['Cost (%)'], df['CPU Energy'], marker='o', label='Classical CPU', color='#1f77b4', linewidth=2)
    ax1.plot(df['Cost (%)'], df['CIM Energy'], marker='s', label='Optiqon CIM', color='#2ca02c', linewidth=2)
    ax1.set_title(f'Solution Quality vs Transaction Costs (Live NIFTY Data - {num_assets} Assets)', fontsize=14)
    ax1.set_ylabel('Ground State Energy (Lower is Better)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(df['Cost (%)'], df['CPU Latency (s)'], marker='o', label='Classical CPU', color='#1f77b4', linewidth=2)
    ax2.plot(df['Cost (%)'], df['CIM Latency (s)'], marker='s', label='Optiqon CIM', color='#2ca02c', linewidth=2)
    ax2.set_title(f'Solver Latency vs Transaction Costs (Live NIFTY Data)', fontsize=14)
    ax2.set_xlabel('Market Impact / Transaction Cost (%)', fontsize=12)
    ax2.set_ylabel('Execution Time (Seconds)')
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('nifty_100_cost_analysis.png', dpi=300)
    print("\n[*] Benchmark plot saved as 'nifty_100_cost_analysis.png'")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_nifty_cost_study()