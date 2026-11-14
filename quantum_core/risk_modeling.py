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
#  File        : quantum_core/risk_modeling.py
#  Author      : B R Nikilesh
#  Description : Quantum-assisted risk analysis via QUBO on Optiqon.
#
#  Approach:
#    Value at Risk (VaR) = identify the loss threshold at confidence level α.
#    Classically: sort returns, take quantile.
#    QUBO formulation: binary selection of loss scenarios that:
#      (a) maximizes total captured loss (finds the worst scenarios)
#      (b) subject to: total probability of selected scenarios ≤ (1 - α)
#    The maximum loss in the selected set is the quantum VaR estimate.
#
#    Also computes: CVaR, Sharpe, Sortino, Max Drawdown.
#------------------------------------------------------------------------------

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optiqon_solver import solve_qubo
from config import N_BINS, CONSTRAINT_PEN


# ─────────────────────────────────────────────────────────────────────────────
#  QUBO for VaR: select the worst (1-α) probability mass of loss scenarios
# ─────────────────────────────────────────────────────────────────────────────

def build_var_qubo(losses:      np.ndarray,
                   probs:       np.ndarray,
                   alpha:       float = 0.95,
                   con_penalty: float = CONSTRAINT_PEN) -> dict:
    """
    QUBO for tail loss scenario selection (VaR computation).

    Variables : x_i ∈ {0,1} — select loss scenario i
    Objective : minimize  -Σ loss_i * x_i               (maximize captured loss)
                         + λ (Σ prob_i * x_i - (1-α))²  (select exactly tail mass)

    The lowest-energy solution selects the worst (1-α) fraction of scenarios,
    whose maximum is the VaR estimate.

    Parameters
    ----------
    losses      : loss values per bin (positive = loss), shape (n,)
    probs       : probability weight per bin, shape (n,)
    alpha       : confidence level (0.95 = 95% VaR)
    con_penalty : constraint penalty strength

    Returns
    -------
    Q : QUBO dict
    """
    n      = len(losses)
    target = 1.0 - alpha     # tail probability mass to capture
    Q      = {}

    # Normalize losses
    l_max   = losses.max() + 1e-9
    norm_l  = losses / l_max

    for i in range(n):
        reward   = -float(norm_l[i])                                        # maximize loss
        con_diag = con_penalty * (probs[i]**2 - 2 * target * probs[i])     # constraint diagonal
        Q[(i, i)] = reward + con_diag

    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = 2.0 * con_penalty * float(probs[i] * probs[j])

    return Q


# ─────────────────────────────────────────────────────────────────────────────
#  Classical risk metrics (reference)
# ─────────────────────────────────────────────────────────────────────────────

def classical_risk_metrics(returns: np.ndarray,
                            alpha: float = 0.95,
                            rf: float = 0.05 / 252) -> dict:
    losses  = -returns
    c_var   = float(np.quantile(losses, alpha))
    tail    = losses[losses >= c_var]
    c_cvar  = float(tail.mean()) if len(tail) > 0 else c_var

    excess  = returns - rf
    sharpe  = float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))
    down    = excess[excess < 0]
    sortino = float(np.mean(excess) / (np.std(down) + 1e-9) * np.sqrt(252)) if len(down) > 0 else sharpe

    return {
        "classical_var_pct":    round(c_var * 100, 4),
        "classical_cvar_pct":   round(c_cvar * 100, 4),
        "classical_sharpe":     round(sharpe, 4),
        "classical_sortino":    round(sortino, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  QUBO VaR decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_var(solution: dict, losses: np.ndarray, probs: np.ndarray) -> dict:
    selected = [i for i, v in solution.items() if v == 1]

    if not selected:
        return {"quantum_var_pct": 0.0, "quantum_cvar_pct": 0.0}

    sel_losses = losses[selected]
    sel_probs  = probs[selected]

    q_var  = float(np.max(sel_losses))
    q_cvar = float(np.average(sel_losses, weights=sel_probs + 1e-9))

    return {
        "quantum_var_pct":  round(q_var * 100, 4),
        "quantum_cvar_pct": round(q_cvar * 100, 4),
        "tail_scenarios":   len(selected),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Max Drawdown
# ─────────────────────────────────────────────────────────────────────────────

def max_drawdown(prices: np.ndarray) -> float:
    peak = prices[0]
    mdd  = 0.0
    for p in prices:
        peak = max(peak, p)
        dd   = (peak - p) / peak
        mdd  = max(mdd, dd)
    return round(mdd * 100, 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_modeling(df, ticker: str, alpha: float = 0.95) -> dict:
    """
    Full quantum risk analysis for a single asset.

    Parameters
    ----------
    df     : DataFrame with 'returns' and 'close' columns
    ticker : asset name
    alpha  : VaR confidence level (default 95%)
    """
    returns = df["returns"].dropna().values
    prices  = df["close"].values
    losses  = -returns   # positive = loss

    print(f"\n[Risk] {ticker} | {len(returns)} observations | α={alpha}")

    # Discretize loss distribution into bins
    n_bins     = 2 ** N_BINS   # 16 bins
    hist, edges = np.histogram(losses, bins=n_bins, density=False)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    probs       = hist / (hist.sum() + 1e-9)

    # Only keep bins with positive probability
    mask        = probs > 0
    bin_losses  = bin_centers[mask]
    bin_probs   = probs[mask]

    print(f"[Risk] QUBO over {len(bin_losses)} non-empty loss bins")

    # Build and solve QUBO
    Q = build_var_qubo(bin_losses, bin_probs, alpha=alpha)
    result = solve_qubo(Q, label="VaR-RiskModeling")

    q_metrics  = decode_var(result["solution"], bin_losses, bin_probs)
    c_metrics  = classical_risk_metrics(returns, alpha=alpha)
    mdd        = max_drawdown(prices)
    ann_vol    = float(np.std(returns) * np.sqrt(252) * 100)
    ann_ret    = float(np.mean(returns) * 252 * 100)

    return {
        "task":                    "Risk Modeling",
        "ticker":                  ticker,
        "confidence_level":        alpha,
        "annualized_return_pct":   round(ann_ret, 4),
        "annualized_vol_pct":      round(ann_vol, 4),
        "max_drawdown_pct":        mdd,
        "qubo_energy":             round(result["energy"], 6),
        "qubo_n_terms":            len(Q),
        "solver_mode":             result["mode"],
        **q_metrics,
        **c_metrics,
    }