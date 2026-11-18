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
#  File        : quantum_core/stock_prediction.py
#  Description : Stock trend prediction via QUBO on the Optiqon machine.
#
#  Approach:
#    Each binary variable x_i represents whether feature i is in its
#    "bullish" regime (1) or "bearish" regime (0).
#    The QUBO encodes:
#      - Feature alignment score  (reward features pointing same direction)
#      - Momentum coherence term  (reward consistent trend across features)
#      - Penalty for contradictory signals
#    Lowest-energy solution → predicted market direction.
#------------------------------------------------------------------------------

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optiqon_solver import solve_qubo
from config import N_BINS, CONSTRAINT_PEN


# ─────────────────────────────────────────────────────────────────────────────
#  QUBO builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_qubo(feature_values: np.ndarray,
                           labels: np.ndarray) -> dict:
    """
    Build a QUBO that encodes "which features agree on UP direction?".

    Variables : x_i ∈ {0,1} for each feature i
        x_i = 1  → feature i is in bullish regime
        x_i = 0  → feature i is in bearish regime

    QUBO objective (minimize):
        - Σ_i  score_i * x_i               (reward high-scoring bullish features)
        - Σ_ij corr_ij * x_i * x_j         (reward agreeing feature pairs)
        + Σ_ij |corr_ij| * x_i * (1-x_j)   (penalize disagreeing pairs)

    Parameters
    ----------
    feature_values : 1D array of normalized feature values [0,1], shape (n_features,)
    labels         : recent return signs (+1 / -1), shape (n_samples,)
                     used to calibrate each feature's bullish score

    Returns
    -------
    Q : QUBO dict {(i,j): value}
    """
    n = len(feature_values)
    Q = {}

    # ── Diagonal: bullish score per feature ──────────────────────────────────
    # If feature value > 0.5 (above midpoint) → bullish signal → negative bias (encourage x_i=1)
    # If feature value < 0.5 → bearish signal → positive bias (discourage x_i=1)
    recent_trend = float(np.sign(np.mean(labels[-10:])))   # +1 or -1

    for i in range(n):
        v = float(feature_values[i])
        # Score: how much does this feature value support the recent trend?
        bullish_score = (v - 0.5) * recent_trend * 2.0
        Q[(i, i)] = -bullish_score   # minimize → reward bullish agreement

    # ── Off-diagonal: feature correlation rewards ─────────────────────────────
    # Reward pairs that agree (both bullish or both bearish)
    for i in range(n):
        for j in range(i + 1, n):
            vi = float(feature_values[i])
            vj = float(feature_values[j])
            agreement = (vi - 0.5) * (vj - 0.5) * 4.0   # positive if same side
            Q[(i, j)] = -agreement * 0.5                   # small cross-term

    return Q


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_prediction(solution: dict, feature_values: np.ndarray) -> dict:
    """
    Decode QUBO solution into a UP/DOWN signal with confidence.

    Confidence = fraction of selected features that vote bullish,
                 weighted by feature magnitude.
    """
    n          = len(feature_values)
    bull_votes = 0.0
    bear_votes = 0.0

    for i in range(n):
        v      = float(feature_values[i])
        weight = abs(v - 0.5) * 2.0   # how extreme is the feature?
        if solution.get(i, 0) == 1:
            bull_votes += weight
        else:
            bear_votes += weight

    total      = bull_votes + bear_votes + 1e-9
    bull_frac  = bull_votes / total
    signal     = "UP 📈" if bull_frac >= 0.5 else "DOWN 📉"
    confidence = max(bull_frac, 1 - bull_frac) * 100

    return {
        "signal":     signal,
        "confidence": round(confidence, 2),
        "bull_score": round(bull_frac * 100, 2),
        "bear_score": round((1 - bull_frac) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Accuracy evaluator (walk-forward on test set)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_accuracy(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2) -> float:
    """
    Simple walk-forward accuracy: run QUBO on each test sample, compare to label.
    """
    split    = int(len(X) * (1 - test_ratio))
    X_test   = X[split:]
    y_test   = y[split:]
    y_train  = y[:split]

    correct = 0
    for feat_row, true_label in zip(X_test, y_test):
        Q      = build_prediction_qubo(feat_row, y_train)
        result = solve_qubo(Q, label="StockPred-Eval")
        pred   = decode_prediction(result["solution"], feat_row)
        pred_sign = 1 if "UP" in pred["signal"] else -1
        if pred_sign == true_label:
            correct += 1

    return correct / len(X_test) * 100 if len(X_test) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_stock_prediction(df, feature_cols: list, scenario: dict | None = None) -> dict:
    """
    End-to-end stock trend prediction.

    Parameters
    ----------
    df           : DataFrame with computed features (from data_fetcher.add_features)
    feature_cols : list of column names to use as features
    scenario     : what-if overrides already applied upstream (for display only)

    Returns
    -------
    Result dict with signal, confidence, accuracy, QUBO metadata
    """
    from data_fetcher import normalize_features

    print(f"\n[StockPred] Features: {feature_cols}")
    X = normalize_features(df, feature_cols)

    # Labels: sign of next-day return
    returns = df["returns"].values
    y       = np.sign(returns)
    y[y == 0] = 1

    # Latest feature vector → prediction
    latest_features = X[-1]
    Q               = build_prediction_qubo(latest_features, y)

    print(f"[StockPred] QUBO size: {len(Q)} terms | variables: {len(feature_cols)}")
    result = solve_qubo(Q, label="StockPrediction")

    pred = decode_prediction(result["solution"], latest_features)

    # Accuracy on held-out test set (skip if very short history)
    accuracy = 0.0
    if len(X) > 20:
        accuracy = evaluate_accuracy(X, y, test_ratio=0.2)

    return {
        "task":               "Stock Prediction",
        "signal":             pred["signal"],
        "confidence_pct":     pred["confidence"],
        "bull_score_pct":     pred["bull_score"],
        "bear_score_pct":     pred["bear_score"],
        "test_accuracy_pct":  round(accuracy, 2),
        "qubo_energy":        round(result["energy"], 6),
        "qubo_n_terms":       len(Q),
        "scenario_applied":   scenario is not None,
        "solver_mode":        result["mode"],
    }