# File: quantum_core/qubo_engine.py
# High-Precision (float64) Normalized QUBO Compiler

import numpy as np
import time

class HighPerformanceQUBOEngine:
    def __init__(self, num_assets=50, bits_per_asset=4, risk_gamma=2.0):
        self.num_assets = num_assets
        self.bits_per_asset = bits_per_asset
        self.num_variables = num_assets * bits_per_asset
        self.risk_gamma = float(risk_gamma)
        self.K = float(2**bits_per_asset - 1)  # Precision denominator (15 for 4-bit)

    def build_qubo_from_real_data(self, returns: np.ndarray, cov: np.ndarray, current_weights: np.ndarray = None, c_penalty: float = 0.0):
        """
        Builds a mathematically stable, float64 QUBO matrix from real return/cov data.
        Applies dynamic penalty scaling, transaction costs, and spectral normalization.
        """
        returns = returns.astype(np.float64)
        cov = cov.astype(np.float64)
        N = self.num_variables

        # 1. Fractional Bit Weight Expansion Array
        bit_weights = np.array([2**k / self.K for k in range(self.bits_per_asset)], dtype=np.float64)

        # 2. Risk Matrix (Gamma * Covariance)
        weight_outer = np.outer(bit_weights, bit_weights)
        Q_risk = self.risk_gamma * np.kron(cov, weight_outer)

        # 3. Return Matrix (-1 * Expected Returns on Diagonal)
        R_diag = np.kron(returns, bit_weights)
        Q_return = -np.diag(R_diag)

        # 4. Dynamic Penalty Calculation (Rigorous Boundary Enforcement)
        max_objective_value = max(np.max(np.abs(returns)), np.max(np.abs(cov)))
        penalty_lambda = float(max_objective_value * 2.5)

        # Budget Constraint: Lambda * (sum w_i - 1)^2
        W_vec = np.tile(bit_weights, self.num_assets)
        Q_penalty = penalty_lambda * np.outer(W_vec, W_vec)
        diag_adjust = penalty_lambda * (W_vec**2 - 2.0 * W_vec)
        np.fill_diagonal(Q_penalty, np.diag(Q_penalty) + diag_adjust)

        # 4.5. Quadratic Transaction Costs (Market Impact)
        if current_weights is not None and c_penalty > 0.0:
            current_weights = np.array(current_weights, dtype=np.float64)
            
            # Quadratic term: c * w_i^2 (Block diagonal matrix, intra-asset crosses only)
            Q_trans = c_penalty * np.kron(np.eye(self.num_assets), weight_outer)
            
            # Linear term: -2c * w_i * w_i^{(old)}
            W_old_vec = np.repeat(current_weights, self.bits_per_asset)
            linear_trans = -2.0 * c_penalty * W_old_vec * W_vec
            
            # Apply linear shift to the diagonal
            np.fill_diagonal(Q_trans, np.diag(Q_trans) + linear_trans)
        else:
            Q_trans = 0.0

        # Combine into Unnormalized QUBO (Float64 precision)
        Q_raw = Q_risk + Q_return + Q_penalty + Q_trans

        # 5. Physical Matrix Normalization (Crucial for D-Wave / Optiqon DAC limits)
        max_abs_val = np.max(np.abs(Q_raw))
        if max_abs_val > 0:
            Q_norm = (Q_raw / max_abs_val) * 10.0
        else:
            Q_norm = Q_raw

        return Q_norm, Q_raw, penalty_lambda, max_abs_val

    def simulated_annealing(self, Q, num_reads=250, num_sweeps=100):
        """Classical baseline (paper Sec. III-B): dwave-neal Simulated Annealing, Metropolis updates
        in sequential (typewriter) order. Returns (state, energy, seconds); the timer covers the
        sampler call only (pure time-to-solution)."""
        import dimod
        from neal import SimulatedAnnealingSampler

        bqm = dimod.BinaryQuadraticModel(Q, "BINARY")
        t0 = time.time()
        sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
        elapsed = time.time() - t0
        best = sampleset.first
        state = np.array([best.sample[i] for i in range(Q.shape[0])], dtype=np.int8)
        return state, float(best.energy), elapsed
