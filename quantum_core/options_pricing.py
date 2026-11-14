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
#  File        : quantum_core/options_pricing.py
#  Author      : B R Nikilesh
#  Description : Options pricing and optimal strike selection via QUBO.
#
#  Classical pricing : Black-Scholes (reference)
#  Quantum component : QUBO over discretized price paths selects the
#                      expected-payoff-maximizing scenario.
#                      This encodes the Monte Carlo path selection problem:
#                      "which subset of price paths maximizes expected payoff
#                       subject to a budget (probability) constraint?"
#------------------------------------------------------------------------------

import sys, os
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optiqon_solver import solve_qubo
from config import N_BINS, CONSTRAINT_PEN


# ─────────────────────────────────────────────────────────────────────────────
#  Black-Scholes reference
# ─────────────────────────────────────────────────────────────────────────────

def black_scholes(S: float, K: float, T: float, r: float,
                  sigma: float, option_type: str = "call") -> float:
    """
    Standard Black-Scholes formula for European options.
    S=spot, K=strike, T=time to expiry (years), r=risk-free rate, sigma=vol.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-9)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ─────────────────────────────────────────────────────────────────────────────
#  Discretized price path generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_price_paths(S: float, K: float, T: float, r: float,
                          sigma: float, n_paths: int = 16) -> tuple:
    """
    Generate n_paths discretized terminal price scenarios using GBM.
    Returns (prices, probabilities, payoffs).
    """
    # Log-normal terminal price distribution
    mu_ln    = np.log(S) + (r - 0.5 * sigma**2) * T
    sigma_ln = sigma * np.sqrt(T)

    # Evenly spaced quantiles as discrete scenarios
    quantiles = np.linspace(0.01, 0.99, n_paths)
    prices    = np.exp(mu_ln + sigma_ln * norm.ppf(quantiles))

    # Probability weight of each scenario (uniform over quantile bins)
    probs = np.ones(n_paths) / n_paths

    return prices, probs


def compute_payoffs(prices: np.ndarray, K: float, option_type: str) -> np.ndarray:
    if option_type.lower() == "call":
        return np.maximum(prices - K, 0)
    else:
        return np.maximum(K - prices, 0)


# ─────────────────────────────────────────────────────────────────────────────
#  QUBO builder: select paths that maximize expected payoff
# ─────────────────────────────────────────────────────────────────────────────

def build_options_qubo(payoffs:     np.ndarray,
                        probs:       np.ndarray,
                        budget:      float       = 0.5,
                        con_penalty: float       = CONSTRAINT_PEN) -> dict:
    """
    QUBO for path selection:
        Variables : x_i ∈ {0,1} — select price path i
        Objective : minimize  -Σ payoff_i * prob_i * x_i
                             + λ (Σ prob_i * x_i - budget)²

    The constraint ensures selected paths sum to ~budget probability mass
    (selects the "most likely profitable" half of the distribution).

    Parameters
    ----------
    payoffs     : payoff per path, shape (n,)
    probs       : probability weight per path, shape (n,)
    budget      : target total probability of selected paths (default 0.5)
    con_penalty : constraint penalty coefficient

    Returns
    -------
    Q : QUBO dict
    """
    n = len(payoffs)
    Q = {}

    # Normalize payoffs to [0,1] for numerical stability
    p_max = payoffs.max() + 1e-9
    norm_payoffs = payoffs / p_max

    # ── Diagonal: payoff reward ───────────────────────────────────────────────
    for i in range(n):
        reward = -float(norm_payoffs[i] * probs[i])
        # Constraint diagonal: λ * prob_i² - 2λ * budget * prob_i
        con_diag = con_penalty * (probs[i]**2 - 2 * budget * probs[i])
        Q[(i, i)] = reward + con_diag

    # ── Off-diagonal: constraint cross terms: 2λ * prob_i * prob_j ───────────
    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = 2.0 * con_penalty * probs[i] * probs[j]

    return Q


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_option_price(solution: dict, prices: np.ndarray,
                         payoffs: np.ndarray, probs: np.ndarray,
                         r: float, T: float) -> dict:
    """Compute option price from selected paths."""
    selected = [i for i, v in solution.items() if v == 1]

    if not selected:
        # Fallback: use all paths
        selected = list(range(len(prices)))

    sel_payoffs = payoffs[selected]
    sel_probs   = probs[selected]
    total_prob  = sel_probs.sum() + 1e-9

    q_price = float(np.exp(-r * T) * np.sum(sel_payoffs * sel_probs) / total_prob)
    return {
        "quantum_price":     round(q_price, 4),
        "n_paths_selected":  len(selected),
        "total_probability": round(float(total_prob), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_options_pricing(df, ticker: str, option_type: str = "call",
                         K: float = None, T: float = 30 / 252,
                         r: float = 0.05) -> dict:
    """
    Full options pricing pipeline.

    Parameters
    ----------
    df          : DataFrame with 'close' and 'volatility' columns
    ticker      : asset name
    option_type : "call" or "put"
    K           : strike price (None → ATM)
    T           : time to expiry in years
    r           : risk-free rate (annual)
    """
    S     = float(df["close"].iloc[-1])
    sigma = float(df["volatility"].iloc[-1]) * np.sqrt(252)   # annualize
    if K is None:
        K = S   # at-the-money

    print(f"\n[Options] {option_type.upper()} | S={S:.2f} K={K:.2f} T={T:.4f} σ={sigma:.4f}")

    # Generate price scenarios
    n_paths = 2 ** N_BINS   # 16 paths (4 binary variables → 4-bit encoding)
    prices, probs = generate_price_paths(S, K, T, r, sigma, n_paths)
    payoffs       = compute_payoffs(prices, K, option_type)

    # Black-Scholes reference
    bs_price = black_scholes(S, K, T, r, sigma, option_type)

    # Build and solve QUBO
    Q = build_options_qubo(payoffs, probs, budget=0.5)

    print(f"[Options] QUBO size: {len(Q)} terms | paths: {n_paths}")
    result = solve_qubo(Q, label=f"Options-{option_type.upper()}")

    decoded = decode_option_price(result["solution"], prices, payoffs, probs, r, T)

    return {
        "task":              "Options Pricing",
        "ticker":            ticker,
        "option_type":       option_type.upper(),
        "spot_price":        round(S, 4),
        "strike_price":      round(K, 4),
        "expiry_years":      round(T, 6),
        "volatility_annual": round(sigma * 100, 2),
        "risk_free_rate":    round(r * 100, 2),
        "bs_reference":      round(bs_price, 4),
        "qubo_energy":       round(result["energy"], 6),
        "qubo_n_terms":      len(Q),
        "solver_mode":       result["mode"],
        **decoded,
        "relative_error_pct": round(
            abs(decoded["quantum_price"] - bs_price) / (bs_price + 1e-9) * 100, 2
        ),
    }