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
#  File        : optiqon_solver.py
#  Author      : B R Nikilesh
#  Description : Unified QUBO solver bridge — four backends supported:
#
#    "simulator" → dimod SimulatedAnnealingSampler  (local, no account)
#    "dwave"     → D-Wave LeapHybridSampler          (cloud, needs API token)
#    "dwave_qpu" → D-Wave DWaveSampler + Embedding   (direct QPU, needs token)
#    "optiqon"   → Quanfluence Optiqon machine        (HTTP POST, local network)
#
#  All quantum_core modules call solve_qubo() only.
#  Switching hardware = change EXECUTION_MODE in .env. Nothing else.
#------------------------------------------------------------------------------

import dimod
import requests
from config import (EXECUTION_MODE,
                    OPTIQON_URL, OPTIQON_TIMEOUT,
                    DWAVE_API_TOKEN, DWAVE_SOLVER, DWAVE_NUM_READS)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  —  this is the only function quantum_core modules call
# ─────────────────────────────────────────────────────────────────────────────

def solve_qubo(Q: dict, label: str = "QUBO") -> dict:
    """
    Solve a QUBO problem on the configured backend.

    Parameters
    ----------
    Q     : QUBO dict  {(i, j): coefficient}  upper-triangular or full
    label : human-readable name shown in logs

    Returns
    -------
    {
        "solution"    : {var_index: 0|1, ...},
        "energy"      : float,
        "solver_mode" : "simulator" | "dwave" | "dwave_qpu" | "optiqon",
        "timing"      : dict   (available for D-Wave)
    }
    """
    n_vars = _count_vars(Q)
    print(f"\n[Solver] '{label}' | backend={EXECUTION_MODE} | variables={n_vars}")

    if EXECUTION_MODE == "dwave":
        return _solve_dwave_hybrid(Q, label)
    elif EXECUTION_MODE == "dwave_qpu":
        return _solve_dwave_qpu(Q, label)
    elif EXECUTION_MODE == "optiqon":
        return _solve_optiqon(Q, label)
    else:
        return _solve_simulator(Q, label)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 1a — D-Wave Leap Hybrid  (EXECUTION_MODE=dwave)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_dwave_hybrid(Q: dict, label: str) -> dict:
    """
    Submit QUBO to D-Wave LeapHybridSampler.

    LeapHybridSampler is the correct solver for most use cases:
      - No solver name needed — auto-picks best available hybrid solver
      - Handles large QUBOs (thousands of variables)
      - Manages its own time budget internally
      - No embedding needed (hybrid handles it)

    Use this unless you specifically need raw QPU access (dwave_qpu mode).
    """
    try:
        from dwave.system import LeapHybridSampler
    except ImportError:
        raise ImportError(
            "D-Wave Ocean SDK not installed.\n"
            "Run: pip install dwave-ocean-sdk"
        )

    if not DWAVE_API_TOKEN or DWAVE_API_TOKEN == "YOUR_DWAVE_API_TOKEN":
        raise ValueError(
            "D-Wave API token not set.\n"
            "Get yours at: https://cloud.dwavesys.com\n"
            "Then set DWAVE_API_TOKEN in your .env file."
        )

    # Build sampler — only pass solver if explicitly set in .env
    sampler_kwargs = {"token": DWAVE_API_TOKEN}
    if DWAVE_SOLVER:
        sampler_kwargs["solver"] = DWAVE_SOLVER

    sampler = LeapHybridSampler(**sampler_kwargs)
    print(f"[D-Wave] LeapHybridSampler ready | solver={sampler.solver.name}")
    print(f"[D-Wave] Submitting QUBO | {len(Q)} terms")

    sampleset = sampler.sample_qubo(Q, label=label)

    best   = sampleset.first
    sol    = dict(best.sample)
    energy = float(best.energy)

    print(f"[D-Wave] ✓ Solution found | energy={energy:.4f}")

    return {
        "solution":    {int(k): int(v) for k, v in sol.items()},
        "energy":      energy,
        "solver_mode": "dwave",
        "timing": {
            "solver": sampler.solver.name,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 1b — D-Wave QPU direct  (EXECUTION_MODE=dwave_qpu)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_dwave_qpu(Q: dict, label: str) -> dict:
    """
    Submit QUBO directly to a D-Wave QPU chip via EmbeddingComposite.

    Only use this if you specifically need raw QPU access.
    Requires DWAVE_SOLVER to be set to a valid QPU name, e.g.:
        DWAVE_SOLVER=Advantage_system4.1
    Burns through QPU quota much faster than hybrid mode.
    """
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
    except ImportError:
        raise ImportError(
            "D-Wave Ocean SDK not installed.\n"
            "Run: pip install dwave-ocean-sdk"
        )

    if not DWAVE_API_TOKEN or DWAVE_API_TOKEN == "YOUR_DWAVE_API_TOKEN":
        raise ValueError(
            "D-Wave API token not set.\n"
            "Get yours at: https://cloud.dwavesys.com\n"
            "Then set DWAVE_API_TOKEN in your .env file."
        )

    sampler_kwargs = {"token": DWAVE_API_TOKEN}
    if DWAVE_SOLVER:
        sampler_kwargs["solver"] = DWAVE_SOLVER

    qpu     = DWaveSampler(**sampler_kwargs)
    sampler = EmbeddingComposite(qpu)

    print(f"[D-Wave] QPU ready | solver={qpu.solver.name}")
    print(f"[D-Wave] Submitting QUBO | {len(Q)} terms | num_reads={DWAVE_NUM_READS}")

    bqm       = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampleset = sampler.sample(bqm, num_reads=DWAVE_NUM_READS, label=label)

    best   = sampleset.first
    sol    = dict(best.sample)
    energy = float(best.energy)

    timing          = sampleset.info.get("timing", {})
    qpu_access_time = timing.get("qpu_access_time", "N/A")

    print(f"[D-Wave] ✓ Solution found | energy={energy:.4f}")
    print(f"[D-Wave] QPU access time: {qpu_access_time} μs")

    return {
        "solution":    {int(k): int(v) for k, v in sol.items()},
        "energy":      energy,
        "solver_mode": "dwave_qpu",
        "timing": {
            "qpu_access_time_us": qpu_access_time,
            "solver":             qpu.solver.name,
            "num_reads":          DWAVE_NUM_READS,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 2 — Quanfluence Optiqon (local network HTTP)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_optiqon(Q: dict, label: str) -> dict:
    """
    Send QUBO to the Quanfluence Optiqon machine.
    Format matches the official run_qubo.py demo exactly.
    """
    bqm              = dimod.BinaryQuadraticModel.from_qubo(Q)
    bqm_serializable = bqm.to_serializable()
    payload          = {"bqm": bqm_serializable}

    print(f"[Optiqon] POST → {OPTIQON_URL}")
    response = requests.post(OPTIQON_URL, json=payload, timeout=OPTIQON_TIMEOUT)

    if response.status_code == 200:
        result   = response.json()
        solution = result["S"]
        energy   = result["energy"]
        print(f"[Optiqon] ✓ Solution found | energy={energy:.4f}")
        return {
            "solution":    _to_dict(solution),
            "energy":      float(energy),
            "solver_mode": "optiqon",
            "timing":      {},
        }
    else:
        raise RuntimeError(
            f"[Optiqon] Hardware call failed | "
            f"HTTP {response.status_code} | {response.text}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Backend 3 — Simulator (local, no account needed)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_simulator(Q: dict, label: str) -> dict:
    """
    Solve QUBO locally using dimod SimulatedAnnealingSampler.
    Exact drop-in for hardware during development.
    """
    bqm       = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler   = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=500, num_sweeps=1000)

    best   = sampleset.first
    sol    = dict(best.sample)
    energy = float(best.energy)

    print(f"[Simulator] ✓ Solution found | energy={energy:.4f}")
    return {
        "solution":    {int(k): int(v) for k, v in sol.items()},
        "energy":      energy,
        "solver_mode": "simulator",
        "timing":      {},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_vars(Q: dict) -> int:
    indices = set()
    for i, j in Q.keys():
        indices.add(i)
        indices.add(j)
    return len(indices)


def _to_dict(solution) -> dict:
    """Normalize solution to {int: int} dict regardless of hardware format."""
    if isinstance(solution, dict):
        return {int(k): int(v) for k, v in solution.items()}
    elif isinstance(solution, list):
        return {i: int(v) for i, v in enumerate(solution)}
    return solution


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Q_demo = {
        (0, 0):  1,   (0, 1): -1,
        (1, 1):  2,   (0, 2):  0.5,  (1, 2): -0.5,
        (2, 2):  1,   (0, 3): -1,    (1, 3):  0.5,
        (2, 3): -0.5, (3, 3):  1.5,
    }
    result = solve_qubo(Q_demo, label="Demo QUBO")
    print("\nSolution :", result["solution"])
    print("Energy   :", result["energy"])
    print("Backend  :", result["solver_mode"])
    if result["timing"]:
        print("Timing   :", result["timing"])