"""Data ingestion (paper Sec. IV-C) with frozen snapshots.

* One isolated Yahoo Finance request per ticker (no batch calls); tickers with no data are dropped.
* ``moments`` fills gaps forward then backward, computes daily logarithmic returns
  r = ln(P_t / P_{t-1}) and returns the annualised mean vector mu and covariance matrix Sigma
  (Eq. 16, 1/(T-1) normalisation, 252 trading days).
* The first download is saved to ``data/``; later runs reuse it so every validator uses identical
  prices. Delete a file in ``data/`` to force a fresh download.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRADING_DAYS = 252


def _cache(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, name)


def _download_one(ticker, **kwargs):
    import yfinance as yf

    data = yf.download(ticker, progress=False, **kwargs)
    if data.empty:
        return None
    if "Adj Close" in data.columns:
        s = data["Adj Close"]
    elif "Close" in data.columns:
        s = data["Close"]
    else:
        s = data.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = ticker
    return s


def _sequential(tickers, **kwargs):
    series = []
    for t in tickers:
        try:
            s = _download_one(t, **kwargs)
        except Exception as exc:  # 404 / rate limit / structure failure: drop the ticker
            print(f"[data] {t}: dropped ({exc})")
            continue
        if s is None:
            print(f"[data] {t}: dropped (no data)")
        else:
            series.append(s)
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1)


def load_close(tickers, tag, period="1y"):
    """50-asset benchmark data, cached as data/<tag>_close.csv."""
    path = _cache(f"{tag}_close.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = _sequential(tickers, period=period, interval="1d")
    df.to_csv(path)
    print(f"[data] snapshot saved to {path}")
    return df


def load_prices(tickers, start, end):
    """Live-market data for a date window, cached in data/."""
    path = _cache(f"prices_{tickers[0]}-{tickers[-1]}_{len(tickers)}_{start}_{end}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = _sequential(tickers, start=start, end=end)
    df.to_csv(path)
    print(f"[data] snapshot saved to {path}")
    return df


def moments(prices, periods=TRADING_DAYS):
    """Annualised expected log-return vector, covariance matrix, and surviving tickers."""
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all").ffill().bfill()
    r = np.log(prices / prices.shift(1)).iloc[1:]
    return (r.mean() * periods).values, (r.cov() * periods).values, prices.columns.tolist()
