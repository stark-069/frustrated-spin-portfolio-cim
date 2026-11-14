#------------------------------------------------------------------------------
#
#            (c) Copyright 2026 by QUANFLUENCE PRIVATE LIMITED
#                          All rights reserved.
#
#   Trade Secret of QUANFLUENCE PRIVATE LIMITED  Do not disclose.
#
#   Use of this file in any form or means is permitted only
#   with a valid, written license agreement with QUANFLUENCE PRIVATE LIMTED.
#   The licensee shall strictly limit use of information contained herein
#   to the conditions specified in the written license agreement.
#
#   Licensee shall keep all information contained herein confidential
#   and shall protect same in whole or in part from disclosure and
#   dissemination to all third parties.
#
#                         QUANFLUENCE PRIVATE LIMITED
#                        E-Mail: assist@quanfluence.com
#                             www.quanfluence.com
#
#------------------------------------------------------------------------------
#  File        : main.py
#  Author      : B R Nikilesh
#  Description : Master Entry point for the Quantum Finance Vertical Pipeline.
#------------------------------------------------------------------------------

import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from intent_parser import parse_intent
from data_fetcher  import fetch_all, fetch_multiple_stocks, add_features, apply_scenario
from config        import (EXECUTION_MODE, BASE_FEATURES, MACRO_FEATURES,
                            NEWS_FEATURES, USE_MACRO_DATA, USE_CRYPTO_DATA, USE_NEWS_DATA,
                            FRED_API_KEY, NEWS_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature set builder
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df) -> list:
    cols = BASE_FEATURES.copy()
    if USE_NEWS_DATA:
        cols += [c for c in NEWS_FEATURES if c in df.columns]
    if USE_MACRO_DATA and FRED_API_KEY != "YOUR_FRED_API_KEY":
        cols += [c for c in MACRO_FEATURES if c in df.columns]
    return [c for c in cols if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline router
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(prompt: str) -> dict:
    _banner(prompt)

    # ── 1. Parse intent ──────────────────────────────────────────────────────
    intent      = parse_intent(prompt)
    task        = intent["task"]
    ticker      = intent["ticker"]
    tickers     = intent["tickers"]
    scenario    = intent.get("scenario")
    option_type = intent.get("option_type", "call")
    strike      = intent.get("strike")

    print(f"[Intent] task={task} | target={ticker} | scenario={scenario}\n")

    # ── 2. Route to Quantum Core ─────────────────────────────────────────────
    
    # --- ROUTE A: SYSTEMIC RISK (Macroeconomic Contagion) ---
    if task == "systemic_risk":
        from quantum_core.systemic_risk import run_systemic_risk
        
        # Synthetic Interbank Network (For Proof of Concept)
        bank_names = ["JPMorgan", "GoldmanSachs", "Citi", "MorganStanley", "SiliconValleyBank", "CreditSuisse"]
        health_scores = np.array([80.0, 75.0, 70.0, 65.0, 20.0, 25.0])
        liability_matrix = np.array([
            [0,  10, 15,  5,  2,  0],  # JPM
            [10,  0, 20, 10,  5,  5],  # GS
            [15, 20,  0, 10,  8, 10],  # Citi
            [5,  10, 10,  0,  4,  5],  # MS
            [2,   5,  8,  4,  0, 20],  # SVB
            [0,   5, 10,  5, 20,  0]   # CS
        ])
        
        target_bank = "SiliconValleyBank" if "svb" in prompt.lower() or "silicon" in prompt.lower() else None
        target_bank = "CreditSuisse" if "credit" in prompt.lower() else target_bank
        
        result = run_systemic_risk(bank_names, health_scores, liability_matrix, shock_bank=target_bank)

    # --- ROUTE B: FRACTIONAL PORTFOLIO OPTIMIZATION ---
    # --- ROUTE B: FRACTIONAL PORTFOLIO OPTIMIZATION ---
    elif task == "portfolio_optimization":
        from quantum_core.portfolio_optimization import run_portfolio_optimization
        
        if len(tickers) < 2:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            
        tickers = tickers[:32] 
        data_dict = fetch_multiple_stocks(tickers)
        
        prices_df = pd.DataFrame({t: data_dict[t]["close"] for t in tickers if not data_dict[t].empty}).dropna()
        returns_df = prices_df.pct_change().dropna()
        expected_ret = {t: float(returns_df[t].mean()) for t in tickers if t in returns_df.columns}
        valid_tickers = list(expected_ret.keys())
        cov_matrix = returns_df[valid_tickers].cov().values

        # Extract risk penalty from intent
        risk_penalty = intent.get("risk_penalty", 2.0)
        print(f"[Portfolio] Applying Risk Penalty (Gamma) = {risk_penalty}")

        # Pass it to the optimization execution engine
        result = run_portfolio_optimization(expected_ret, cov_matrix, valid_tickers, bits_per_asset=4, risk_penalty=risk_penalty)

    # --- ROUTE C: STANDARD PREDICTIVE / RISK TASKS ---
    else:
        data = fetch_all(ticker, include_macro=USE_MACRO_DATA, include_crypto=USE_CRYPTO_DATA, include_news=USE_NEWS_DATA)
        df_raw = data["merged_df"] if not data["merged_df"].empty else data["price_df"]
        sentiment = data["sentiment"]

        if df_raw.empty:
            return {"task": task, "ticker": ticker, "error": f"No price data found for '{ticker}'."}

        if scenario:
            df_raw = apply_scenario(df_raw, scenario)

        df = add_features(df_raw)
        feature_cols = get_feature_cols(df)

        if task == "stock_prediction":
            from quantum_core.stock_prediction import run_stock_prediction
            result = run_stock_prediction(df, feature_cols, scenario)

        elif task == "options_pricing":
            from quantum_core.options_pricing import run_options_pricing
            result = run_options_pricing(df, ticker, option_type=option_type, K=strike, T=30/252, r=0.05)

        elif task == "risk_modeling":
            from quantum_core.risk_modeling import run_risk_modeling
            result = run_risk_modeling(df, ticker)
            
        else:
            result = {"error": f"Unrecognised task: {task}"}

        # Attach metadata for Route C
        result["data_sources"] = {
            "price":  "Yahoo Finance",
            "macro":  "FRED" if not data["macro_df"].empty else "unavailable",
            "news":   sentiment.get("source", "none"),
        }

    # Attach global metadata
    result["prompt"] = prompt
    result["intent"] = intent
    result["mode"]   = EXECUTION_MODE
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Output formatter
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict):
    if "error" in result and len(result) <= 5:
        print(f"\n{'═'*62}")
        print(f"  [ERROR] {result['error']}")
        print(f"{'═'*62}\n")
        return

    skip = {"prompt", "intent", "mode", "weights", "default_cascade", "surviving_institutions"}
    
    print(f"\n{'═'*62}")
    print(f"  RESULT  —  {result.get('task', 'Unknown Task')}")
    print(f"  Solver  —  {result.get('solver_mode', result.get('mode','?')).upper()}")
    print(f"{'═'*62}")

    # Standard Dict Printing
    for k, v in result.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        elif isinstance(v, list):
            print(f"  {k}: {', '.join(str(x) for x in v)}")
        else:
            print(f"  {k}: {v}")

    # Special Printing: Fractional Portfolio Weights
    if "weights" in result:
        print(f"  portfolio_allocation:")
        for t, w in sorted(result["weights"].items(), key=lambda item: item[1], reverse=True):
            bar = "█" * int(w / 5) # Scale bar down for percentages
            print(f"    {t:14s} {bar}  {w:.2f}%")
            
    # Special Printing: Systemic Risk Cascade
    if "default_cascade" in result:
        print(f"\n  [!] SYSTEMIC CASCADE WARNING:")
        print(f"      Defaulted: {', '.join(result['default_cascade'])}")
        print(f"      Survived : {', '.join(result['surviving_institutions'])}")

    print(f"{'═'*62}\n")


def _banner(prompt: str):
    print(f"\n{'─'*62}")
    print(f"  QUANFLUENCE MASTER COMPILER  |  {EXECUTION_MODE.upper()}")
    print(f"  Prompt : {prompt}")
    print(f"{'─'*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = run_pipeline(prompt)
        print_result(result)
    else:
        print("\n  Quanfluence Master Compiler — Interactive Mode")
        print(f"  Execution Backend: {EXECUTION_MODE.upper()}")
        print("  Type 'quit' to exit\n")
        print("  Examples:")
        print("    > Optimize portfolio with AAPL MSFT GOOGL AMZN TSLA META")
        print("    > Simulate systemic contagion if Silicon Valley Bank defaults")
        print("    > Predict AAPL stock price next week")
        print("    > What if TSLA drops to 150, will it go up?")
        print("    > Analyze risk for NVDA\n")

        while True:
            try:
                prompt = input(">>> ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                result = run_pipeline(prompt)
                print_result(result)
            except KeyboardInterrupt:
                break
            except Exception as e:
                import traceback
                print(f"\n[Error] {e}")
                traceback.print_exc()

if __name__ == "__main__":
    main()