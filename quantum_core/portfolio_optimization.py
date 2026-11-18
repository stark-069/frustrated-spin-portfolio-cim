#------------------------------------------------------------------------------
#
#   (c) Copyright 2026 Qurious Labs,
#   Centre for Interdisciplinary Areas in Quantum Computing,
#   Indian Institute of Technology Indore, Indore 453552, India.
#
#   Property of Qurious Labs, IIT Indore.
#   Released under the MIT License (see LICENSE in the repository root).
#
#------------------------------------------------------------------------------
#  File        : quantum_core/portfolio_optimization.py
#  Description : Markowitz portfolio optimization as a QUBO problem,
#                solved on the Optiqon machine.
#
#  QUBO formulation:
#    Variables : x_i ∈ {0,1}  — include asset i in portfolio (1) or not (0)
#    Objective : minimize  -Σ μ_i x_i  +  λ Σ_ij σ_ij x_i x_j
#                          + μ_pen (Σ x_i - k)²        [cardinality constraint]
#    where μ_i = expected return, σ_ij = covariance, k = target # of assets
#------------------------------------------------------------------------------

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optiqon_solver import solve_qubo
from config import RISK_PENALTY, CONSTRAINT_PEN

def calculate_dynamic_penalties(expected_returns, cov_matrix, risk_penalty):
    """
    Dynamically scales the constraint penalty based on the input data's magnitude
    to prevent the quantum solver from violating the fractional budget.
    """
    max_return_pull = np.sum(np.abs(expected_returns))
    max_risk_pull = risk_penalty * np.sum(np.abs(cov_matrix))
    
    # The penalty must mathematically overpower the absolute maximum possible reward
    budget_penalty = (max_return_pull + max_risk_pull) * 1.5 
    
    return budget_penalty
# ─────────────────────────────────────────────────────────────────────────────
#  QUBO builder
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_qubo(expected_returns: np.ndarray,
                         cov_matrix: np.ndarray,
                         risk_penalty: float = 2.0,
                         budget_penalty: float = 5.0,
                         bits_per_asset: int = 4) -> dict:
    """
    Fractional Markowitz QUBO for a 128-spin architecture.
    Variables: x_{i,q} ∈ {0,1} represents the q-th bit of asset i's weight.
    """
    num_assets = len(expected_returns)
    M = bits_per_asset
    
    # K is the normalization factor for the bits. For 4 bits, 2^4 - 1 = 15.
    # This bounds our maximum weight for any single asset.
    K = float(2**M - 1) 
    Q = {}

    for i in range(num_assets):
        for q in range(M):
            idx_k = i * M + q
            bit_val_k = (2**q) / K
            
            # 1. Linear Return Bias (maximize return -> minimize negative return)
            return_term = -float(expected_returns[i]) * bit_val_k
            
            # 2. Budget Constraint (Linear part): -2 * budget_penalty * w_i
            budget_linear = -2.0 * budget_penalty * bit_val_k
            
            # 3. Budget Constraint & Risk (Diagonal part): w_i^2 terms
            # Since x^2 = x for binary variables, we apply this to the diagonal.
            risk_diag = risk_penalty * float(cov_matrix[i, i]) * (bit_val_k ** 2)
            budget_diag = budget_penalty * (bit_val_k ** 2)
            
            Q[(idx_k, idx_k)] = Q.get((idx_k, idx_k), 0.0) + return_term + budget_linear + risk_diag + budget_diag
            
            # Quadratic terms (Off-diagonals)
            for j in range(num_assets):
                for r in range(M):
                    idx_l = j * M + r
                    bit_val_l = (2**r) / K
                    
                    if idx_k < idx_l:
                        # 4. Risk Penalty (Covariance between different bits/assets)
                        risk_cross = 2.0 * risk_penalty * float(cov_matrix[i, j]) * bit_val_k * bit_val_l
                        
                        # 5. Budget Constraint (Cross terms): 2 * w_i * w_j
                        budget_cross = 2.0 * budget_penalty * bit_val_k * bit_val_l
                        
                        Q[(idx_k, idx_l)] = Q.get((idx_k, idx_l), 0.0) + risk_cross + budget_cross

    return Q
# ─────────────────────────────────────────────────────────────────────────────
#  Decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_portfolio(sample, asset_names: list, bits_per_asset: int = 4) -> dict:
    """
    Translates a 128-bit raw quantum solver output back into human-readable 
    portfolio allocation percentages.
    """
    M = bits_per_asset
    K = float(2**M - 1)  # Normalization factor (15.0 for 4 bits)
    portfolio_weights = {}
    
    # Standardize sample input format (handles both dict and array outputs from solvers)
    if isinstance(sample, dict):
        sample_list = [sample[idx] for idx in sorted(sample.keys())]
    else:
        sample_list = list(sample)
        
    for i, asset in enumerate(asset_names):
        accumulated_val = 0.0
        
        # Reconstruct the fractional value from the M bits allocated to this asset
        for q in range(M):
            idx = i * M + q
            bit = sample_list[idx]
            accumulated_val += (2**q) * bit
            
        # Calculate the final percentage allocation
        fractional_weight = accumulated_val / K
        portfolio_weights[asset] = round(fractional_weight * 100, 2)
        
    return portfolio_weights


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_optimization(expected_returns_dict: dict,
                               cov_matrix: np.ndarray,
                               tickers: list,
                               bits_per_asset: int = 4,
                               risk_penalty: float = 2.0) -> dict:
    
    n = len(tickers)
    expected_returns = np.array([expected_returns_dict[t] for t in tickers])
    ann_returns = expected_returns * 252
    ann_cov = cov_matrix * 252

# --- DYNAMIC SCALING ENGINE ---
    max_return_pull = np.sum(np.abs(ann_returns))
    max_risk_pull = risk_penalty * np.sum(np.abs(ann_cov))
    
    # K is the precision factor (15 for 4 bits)
    K = float(2**bits_per_asset - 1)
    
    # Scale the penalty by K to overpower the fractional squaring loophole
    dynamic_budget_pen = (max_return_pull + max_risk_pull) * K
    # ------------------------------ 
    # ------------------------------

    total_spins = n * bits_per_asset
    precision_levels = 2**bits_per_asset - 1

    print(f"\n[Portfolio] {n} assets: {tickers}")
    print(f"[Portfolio] Fractional Encoding: {bits_per_asset} bits/asset ({precision_levels} precision levels)")
    print(f"[Portfolio] Dynamic Budget Penalty computed: {dynamic_budget_pen:.2f}")

    # Pass the dynamically computed penalty into the QUBO builder
    Q = build_portfolio_qubo(ann_returns, ann_cov,
                             risk_penalty=risk_penalty,       
                             budget_penalty=dynamic_budget_pen,     
                             bits_per_asset=bits_per_asset)

    result = solve_qubo(Q, label="FractionalPortfolio")
    weights_dict = decode_portfolio(result["solution"], tickers, bits_per_asset=bits_per_asset)

    w = np.array([weights_dict[t] / 100.0 for t in tickers])
    exp_return = float(np.dot(w, ann_returns))
    port_risk = float(np.sqrt(np.dot(w.T, np.dot(ann_cov, w))))
    sharpe = exp_return / (port_risk + 1e-9)

    active_assets = {k: v for k, v in weights_dict.items() if v > 0}

    return {
        "task": "Fractional Portfolio Optimization",
        "n_assets_input": n,
        "total_physical_spins": total_spins,
        "qubo_energy": round(result["energy"], 6),
        "qubo_n_terms": len(Q),
        "solver_mode": result.get("solver_mode", result.get("mode", "unknown")),
        "expected_return_pct": round(exp_return * 100, 3),
        "portfolio_risk_pct": round(port_risk * 100, 3),
        "sharpe_ratio": round(sharpe, 4),
        "weights": active_assets, 
    }