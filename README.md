# Dynamic Portfolio Optimization on a Coherent Ising Machine: validation repository

Code, market data, result files and figures for the paper
**"Bypassing Classical Bottlenecks in Frustrated Spin Networks: Dynamic Portfolio Optimization on a Coherent Ising Machine"**
(A. Gohar, A. Bindod, S. Gupta; Dept. of Physics, IIT Indore).

This repository contains what is needed to check the paper's results independently.

## Access to the Coherent Ising Machine (CIM)

The Optiqon CIM is accessible via **https://.....**
<!-- replace with the CIM access URL -->

Everything except the CIM runs themselves can be reproduced on an ordinary computer. CIM scripts read their credentials from environment variables (see `.env.example`); no keys are stored in the code.

## Paper to repository map

| Paper section | What it covers | Code | Output / reference values |
|---|---|---|---|
| II. QUBO formulation (Eqs. 1-11) | 4-bit radix-2 expansion, market-impact penalty, dynamic penalty lambda, [-10, 10] normalisation | `quantum_core/qubo_engine.py` | n/a |
| III-A. CIM interface | `cim_cloud` SDK | `cim_cloud.py` | n/a |
| III-B. Classical baseline | `dwave-neal` Simulated Annealing, 250 reads x 100 sweeps | `quantum_core/qubo_engine.py` (`simulated_annealing`) | n/a |
| III-C / IV-C. Data ingestion | Yahoo download, gap filling, mu and Sigma | `market_data.py` (snapshots), `real_market_test.py`, `100assetindia.py` | `data/` |
| IV-A. 50-asset baseline | Tables I-II, Figs. 1-2 | `benchmark_classical.py`, `benchmark_dwave.py`, `benchmark_optiqon.py`, `generate_comparison_reports.py` | `paper_tables/table1_2_baseline_50_assets.csv`, `reports/results_*.json` (written when you run the scripts), `figures/fig1*`, `fig2*` |
| IV-B. Scaling 500-1,500 assets | Figs. 3-4 | not included (published figures only) | `paper_tables/fig3_fig4_scaling_balanced_standard.csv`, `figures/fig3*`, `fig4*` |
| IV-C. 5,000-asset ceiling | payload stress test | `test_20k_dwave.py`, 5,000-asset block of `benchmark_*.py` | `reports/results_classical.json`, `reports/results_dwave.json` |
| V-A, V-B. US and NIFTY 50 case studies | Tables III-IV | `real_market_test.py` (classical), `cim_real_market_test.py` (CIM) | `paper_tables/table3_*.csv`, `table4_*.csv` |
| VI. NIFTY-100 friction study | Table V, Fig. 5 | `100assetindia.py`, `plot_nifty_results.py` | `paper_tables/table5_*.csv`, `figures/fig5*` |

`paper_tables/` holds the values as printed in the paper; compare your own run against them.
`reports/` is filled by the benchmark scripts; `figures/` holds the figures as printed in the paper.

Supporting modules (not paper results): `main.py`, `intent_parser.py`, `config.py`, `data_fetcher.py`, `optiqon_solver.py` and `quantum_core/{portfolio_optimization,stock_prediction,options_pricing,risk_modeling}.py` form the prompt-to-solver pipeline the CIM bridge was built on.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # add CIM_API_KEY / OPTIQON_URL / DWAVE_API_TOKEN only if you have access
```

## Reproducing the results

| Paper item | Command | Needs |
|---|---|---|
| Tables I-II, Figs. 1-2 | `python benchmark_classical.py`, `benchmark_dwave.py`, `benchmark_optiqon.py`, then `python generate_comparison_reports.py` | CPU; D-Wave token; `CIM_API_KEY` |
| 5,000-asset ceiling | `python test_20k_dwave.py`; `python benchmark_optiqon.py --stress-5000` | D-Wave token; `CIM_API_KEY` |
| Tables III-IV | `python real_market_test.py`; `python cim_real_market_test.py` | CPU; `CIM_API_KEY` for the CIM run |
| Table V, Fig. 5 | `python 100assetindia.py`; `python plot_nifty_results.py` redraws Fig. 5 from the recorded values | CPU; `CIM_API_KEY` for the CIM column |

Classical results are fully reproducible from this repository. D-Wave and CIM results need access to those services.

## Data

* Source: Yahoo Finance via `yfinance`, adjusted daily closes. Live-market studies use 2024-01-01 to 2026-01-01; the 50-asset benchmarks use a 1-year window (`TICKERS_50` in `benchmark_*.py`).
* Ingestion (Sec. IV-C): one isolated request per ticker, tickers with no data dropped, gaps forward-filled then backward-filled (`market_data.py`).
* Statistics (Eq. 16): daily log returns `r = ln(P_t / P_{t-1})`; `mu` and `Sigma` are the sample mean and covariance (`1/(T-1)`), annualised with 252 trading days.
* Universes: the 20 US and 20 NIFTY tickers are listed in `real_market_test.py`; the NIFTY-100 list is in `100assetindia.py` (tickers Yahoo cannot serve are dropped).
* Snapshots: the first run of any script saves its download to `data/`; later runs reuse it, so everyone validates against identical prices. Delete a file in `data/` to force a fresh download.

## Solver settings

| Setting | Value |
|---|---|
| Encoding | 4 bits per asset, `w_i = (1/15) * sum_q 2^q x_{i,q}` |
| Budget penalty | `lambda = 2.5 * max(max abs mu, max abs Sigma)` |
| Market impact | `c_penalty * sum (w_i - w_i^old)^2`, `w^old` = equal weight; 0.5 % in the 20-asset studies, 0 / 0.5 / 2 / 5 % in the NIFTY-100 study |
| Normalisation | whole QUBO scaled so that the largest absolute entry is 10 |
| Classical solver | `dwave-neal` Simulated Annealing, 250 reads, 100 sweeps per read; timer covers the sampler call only |
| CIM | `cim_cloud` SDK; reported time is hardware execution time, excluding API transmission |
| Risk aversion `gamma` | 2.5 in the live studies; benchmark profiles 0.1, 0.5, 2, 5, 8, 15 (Pure Alpha ... Min Volatility) |

Timings depend on hardware (the paper's classical runs used an Apple M3 with 16 GB RAM).

## Citation

If you use this repository, please cite the paper (reference to be added on publication).
