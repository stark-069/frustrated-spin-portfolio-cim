#------------------------------------------------------------------------------
#
#            (c) Copyright 2026 by QUANFLUENCE PRIVATE LIMITED
#                          All rights reserved.
#
#------------------------------------------------------------------------------
#  File        : config.py
#  Description : Central configuration + solver dispatcher for the
#                Quantum Finance Vertical Pipeline.
#
#  Execution modes (set EXECUTION_MODE in .env):
#    simulator   — SimulatedAnnealingSampler  (local, no credentials)
#    tabu        — TabuSampler                (local, faster than SA)
#    steepest    — SteepestDescentSampler     (local, deterministic)
#    exact       — ExactSolver               (local, tiny problems ≤ ~20 vars)
#    dwave       — LeapHybridSampler          (D-Wave Leap cloud)
#    dwave_qpu   — DWaveSampler              (QPU direct, needs embedding)
#    optiqon     — OptiqonSampler            (Optiqon machine via REST)
#------------------------------------------------------------------------------

from dotenv import load_dotenv
import os

load_dotenv()

# ─── EXECUTION MODE ───────────────────────────────────────────────────────────
# Controls which quantum / classical sampler get_solver() returns.
# Override at runtime:  EXECUTION_MODE=dwave python main.py "..."
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "simulator").lower().strip()

# ─── D-WAVE LEAP ──────────────────────────────────────────────────────────────
DWAVE_API_TOKEN  = os.getenv("DWAVE_API_TOKEN", "")
# Leave blank to let the Leap client pick the best available solver automatically.
DWAVE_SOLVER     = os.getenv("DWAVE_SOLVER") or None
# Number of reads sent to the QPU (dwave_qpu mode only).
DWAVE_NUM_READS  = int(os.getenv("DWAVE_NUM_READS", 1000))

# ─── OPTIQON MACHINE ──────────────────────────────────────────────────────────
OPTIQON_URL      = os.getenv("OPTIQON_URL", "")
OPTIQON_API_KEY  = os.getenv("OPTIQON_API_KEY", "")
OPTIQON_TIMEOUT  = (
    int(os.getenv("OPTIQON_CONNECT_TIMEOUT", 10)),
    int(os.getenv("OPTIQON_READ_TIMEOUT",    3600)),
)

# ─── LOCAL SAMPLER PARAMS ─────────────────────────────────────────────────────
# Simulated annealing
SA_NUM_READS     = int(os.getenv("SA_NUM_READS",    1000))
SA_NUM_SWEEPS    = int(os.getenv("SA_NUM_SWEEPS",   1000))
# Tabu search
TABU_NUM_READS   = int(os.getenv("TABU_NUM_READS",  1000))
TABU_TENURE      = os.getenv("TABU_TENURE")   # None → auto
TABU_TIMEOUT     = int(os.getenv("TABU_TIMEOUT_MS", 20))   # milliseconds

# ─── API KEYS ─────────────────────────────────────────────────────────────────
FRED_API_KEY     = os.getenv("FRED_API_KEY",  "")
NEWS_API_KEY     = os.getenv("NEWS_API_KEY",  "")

# ─── DATA SOURCE TOGGLES ──────────────────────────────────────────────────────
USE_MACRO_DATA   = os.getenv("USE_MACRO_DATA",  "True").lower()  == "true"
USE_CRYPTO_DATA  = os.getenv("USE_CRYPTO_DATA", "False").lower() == "true"
USE_NEWS_DATA    = os.getenv("USE_NEWS_DATA",   "True").lower()  == "true"

# ─── DATA FETCH DEFAULTS ──────────────────────────────────────────────────────
DEFAULT_PERIOD   = os.getenv("DEFAULT_PERIOD",   "1y")
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1d")

# ─── QUBO PARAMS ──────────────────────────────────────────────────────────────
N_BINS          = int(os.getenv("N_BINS",         4))
RISK_PENALTY    = float(os.getenv("RISK_PENALTY", 2.0))
CONSTRAINT_PEN  = float(os.getenv("CONSTRAINT_PEN", 5.0))

# ─── FEATURE COLUMNS ──────────────────────────────────────────────────────────
BASE_FEATURES   = ["returns", "volatility", "rsi", "macd"]
MACRO_FEATURES  = ["fed_rate", "vix", "us10y"]
NEWS_FEATURES   = ["news_sentiment"]


# ─────────────────────────────────────────────────────────────────────────────
#  Solver dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def get_solver(mode: str | None = None):
    """
    Return a dimod-compatible sampler for the requested execution mode.

    Parameters
    ----------
    mode : str, optional
        Override EXECUTION_MODE for this call only.
        Useful for testing or running two solvers in parallel.

    Returns
    -------
    A sampler that exposes .sample_qubo() / .sample_ising() / .sample().

    Raises
    ------
    ValueError   — unknown mode or missing credentials.
    ImportError  — required package not installed.

    Usage in quantum modules
    ------------------------
        from config import get_solver

        sampler    = get_solver()
        sampleset  = sampler.sample_qubo(Q, **get_solver_params())
    """
    target = (mode or EXECUTION_MODE).lower().strip()

    # ── Local simulators (no credentials needed) ──────────────────────────────

    if target == "simulator":
        try:
            from neal import SimulatedAnnealingSampler
        except ImportError:
            from dimod import SimulatedAnnealingSampler   # dimod fallback
        return SimulatedAnnealingSampler()

    if target == "tabu":
        try:
            from tabu import TabuSampler
        except ImportError as e:
            raise ImportError(
                "TabuSampler requires the 'dwave-tabu' package. "
                "Install it with:  pip install dwave-tabu"
            ) from e
        tenure = int(TABU_TENURE) if TABU_TENURE else None
        return TabuSampler(tenure=tenure)

    if target == "steepest":
        try:
            from greedy import SteepestDescentSampler
        except ImportError as e:
            raise ImportError(
                "SteepestDescentSampler requires the 'dwave-greedy' package. "
                "Install it with:  pip install dwave-greedy"
            ) from e
        return SteepestDescentSampler()

    if target == "exact":
        from dimod import ExactSolver
        print(
            "[config] WARNING: ExactSolver is exponential — "
            "safe only for QUBO problems with ≤ ~20 variables."
        )
        return ExactSolver()

    # ── D-Wave Leap (cloud) ───────────────────────────────────────────────────

    if target == "dwave":
        _require_token("DWAVE_API_TOKEN", DWAVE_API_TOKEN, "dwave")
        try:
            from dwave.system import LeapHybridSampler
        except ImportError as e:
            raise ImportError(
                "D-Wave cloud requires 'dwave-system'. "
                "Install with:  pip install dwave-system"
            ) from e
        kwargs = dict(token=DWAVE_API_TOKEN)
        if DWAVE_SOLVER:
            kwargs["solver"] = DWAVE_SOLVER
        sampler = LeapHybridSampler(**kwargs)
        print(f"[config] D-Wave LeapHybridSampler ready  "
              f"(solver={sampler.solver.name})")
        return sampler

    if target == "dwave_qpu":
        _require_token("DWAVE_API_TOKEN", DWAVE_API_TOKEN, "dwave_qpu")
        try:
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError as e:
            raise ImportError(
                "D-Wave QPU requires 'dwave-system'. "
                "Install with:  pip install dwave-system"
            ) from e
        kwargs = dict(token=DWAVE_API_TOKEN)
        if DWAVE_SOLVER:
            kwargs["solver"] = DWAVE_SOLVER
        qpu     = DWaveSampler(**kwargs)
        sampler = EmbeddingComposite(qpu)
        print(f"[config] D-Wave QPU ready  "
              f"(solver={qpu.solver.name}, "
              f"qubits={len(qpu.nodelist)})")
        return sampler

    # ── Optiqon machine (REST) ────────────────────────────────────────────────

    if target == "optiqon":
        if not OPTIQON_URL:
            raise ValueError(
                "EXECUTION_MODE=optiqon requires OPTIQON_URL in .env"
            )
        try:
            from optiqon_solver import OptiqonSampler
        except ImportError as e:
            raise ImportError(
                "Optiqon mode requires optiqon_solver.py in your project root. "
                "Contact Optiqon / Quanfluence for the SDK."
            ) from e
        sampler = OptiqonSampler(
            url     = OPTIQON_URL,
            api_key = OPTIQON_API_KEY or None,
            timeout = OPTIQON_TIMEOUT,
        )
        print(f"[config] Optiqon sampler ready  (url={OPTIQON_URL})")
        return sampler

    # ── Unknown mode ──────────────────────────────────────────────────────────

    valid = ["simulator", "tabu", "steepest", "exact",
             "dwave", "dwave_qpu", "optiqon"]
    raise ValueError(
        f"Unknown EXECUTION_MODE: '{target}'. "
        f"Valid options: {', '.join(valid)}"
    )


def get_solver_params(mode: str | None = None) -> dict:
    """
    Return sampler-specific keyword arguments for .sample_qubo() / .sample().

    Pass these as **get_solver_params() alongside the QUBO dict so each
    backend gets the right knobs without the calling module knowing which
    backend is active.

    Example
    -------
        sampler   = get_solver()
        params    = get_solver_params()
        sampleset = sampler.sample_qubo(Q, **params)
    """
    target = (mode or EXECUTION_MODE).lower().strip()

    if target == "simulator":
        return dict(num_reads=SA_NUM_READS, num_sweeps=SA_NUM_SWEEPS)

    if target == "tabu":
        return dict(num_reads=TABU_NUM_READS, timeout=TABU_TIMEOUT)

    if target == "steepest":
        return dict(num_reads=SA_NUM_READS)

    if target == "exact":
        return {}   # ExactSolver ignores extra kwargs

    if target == "dwave":
        return {}   # LeapHybridSampler manages its own time budget

    if target == "dwave_qpu":
        return dict(num_reads=DWAVE_NUM_READS)

    if target == "optiqon":
        return {}   # Optiqon SDK uses constructor-level config

    return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_token(env_name: str, value: str, mode: str) -> None:
    if not value:
        raise ValueError(
            f"EXECUTION_MODE={mode} requires {env_name} to be set in .env"
        )