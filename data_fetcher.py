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
#  File        : data_fetcher.py
#  Description : Multi-source financial data fetcher.
#
#  Sources pulled:
#    1. Yahoo Finance   — OHLCV price history        (always)
#    2. FRED            — Macro indicators            (interest rate, inflation, VIX)
#    3. Binance API     — Crypto prices               (optional, no key needed)
#    4. NewsAPI         — News headlines sentiment    (needs free API key)
#    5. NSE India       — Indian market data          (for .NS / .BO tickers)
#    6. Yahoo RSS       — Ticker-specific news        (always, no key needed)
#
#  All sources degrade gracefully — if one fails, the rest still work.
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

from config import DEFAULT_PERIOD, DEFAULT_INTERVAL, NEWS_API_KEY, FRED_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
#  1. Yahoo Finance — Price / OHLCV
# ─────────────────────────────────────────────────────────────────────────────

def _clean_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns, strip timezone, drop NaNs."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def fetch_stock_data(ticker: str,
                     period: str   = DEFAULT_PERIOD,
                     interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """
    Fetch OHLCV price history from Yahoo Finance.

    Uses two methods so Indian BSE/NSE tickers work reliably:
      Method 1 — yf.Ticker().history()   (better for .BO / .NS)
      Method 2 — yf.download()           (fallback, works for most US tickers)
    """
    if not HAS_YF:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    print(f"[Yahoo] Fetching {ticker} | period={period} | interval={interval}")

    is_indian = ticker.upper().endswith(".BO") or ticker.upper().endswith(".NS")
    df = pd.DataFrame()

    # ── Method 1: Ticker.history — more reliable for Indian exchanges ─────────
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty:
            df = _clean_yf_df(df)
            print(f"[Yahoo] ✓ {len(df)} bars via Ticker.history | "
                  f"{df.index[0].date()} → {df.index[-1].date()}")
            return df
    except Exception as e:
        print(f"[Yahoo] Ticker.history failed for {ticker}: {e}")

    # ── Method 2: yf.download — standard fallback ────────────────────────────
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if not df.empty:
            df = _clean_yf_df(df)
            print(f"[Yahoo] ✓ {len(df)} bars via download | "
                  f"{df.index[0].date()} → {df.index[-1].date()}")
            return df
    except Exception as e:
        print(f"[Yahoo] download failed for {ticker}: {e}")

    # ── Both methods failed ───────────────────────────────────────────────────
    raise ValueError(f"[Yahoo] No data for: {ticker}")


def fetch_multiple_stocks(tickers: list, period: str = DEFAULT_PERIOD) -> dict:
    """Fetch OHLCV for multiple tickers. Returns {ticker: DataFrame}."""
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_stock_data(t, period=period)
        except Exception as e:
            print(f"[Yahoo] Warning: {t} failed — {e}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  2. FRED — Macro Indicators
# ─────────────────────────────────────────────────────────────────────────────

FRED_SERIES = {
    "fed_rate":    "DFF",              # Federal Funds Rate (daily)
    "inflation":   "CPIAUCSL",         # CPI (monthly)
    "vix":         "VIXCLS",           # VIX Volatility Index (daily)
    "us10y":       "DGS10",            # 10-Year Treasury Yield (daily)
    "unemployment":"UNRATE",           # Unemployment Rate (monthly)
    "gdp_growth":  "A191RL1Q225SBEA",  # Real GDP growth (quarterly)
}

def fetch_fred_macro(series_ids: dict = None, api_key: str = None) -> pd.DataFrame:
    """
    Fetch macro indicators from FRED (Federal Reserve Economic Data).

    Parameters
    ----------
    series_ids : dict of {name: FRED_series_id} — defaults to FRED_SERIES
    api_key    : FRED API key (free at fred.stlouisfed.org) — uses config if omitted

    Returns
    -------
    DataFrame with macro indicators aligned to business days, forward-filled.
    """
    key      = api_key or FRED_API_KEY
    series   = series_ids or FRED_SERIES
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    frames   = {}

    for name, series_id in series.items():
        try:
            params = {
                "series_id": series_id,
                "api_key":   key,
                "file_type": "json",
                "sort_order":"desc",
                "limit":     365,
            }
            r = requests.get(base_url, params=params, timeout=10)
            if r.status_code != 200:
                print(f"[FRED] {name} ({series_id}) → HTTP {r.status_code} — skipped")
                continue

            obs = r.json().get("observations", [])
            if not obs:
                continue

            df_s = pd.DataFrame(obs)[["date", "value"]]
            df_s["date"]  = pd.to_datetime(df_s["date"])
            df_s["value"] = pd.to_numeric(df_s["value"], errors="coerce")
            df_s.set_index("date", inplace=True)
            df_s.rename(columns={"value": name}, inplace=True)
            df_s.sort_index(inplace=True)
            frames[name] = df_s
            print(f"[FRED] ✓ {name:15s} ({series_id}) | {len(df_s)} observations")

        except Exception as e:
            print(f"[FRED] {name} failed — {e}")

    if not frames:
        print("[FRED] No macro data fetched — continuing without it")
        return pd.DataFrame()

    macro = pd.concat(frames.values(), axis=1)
    macro = macro.resample("B").last().ffill()
    return macro


# ─────────────────────────────────────────────────────────────────────────────
#  3. Binance — Crypto Prices (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

CRYPTO_SYMBOLS = {
    "bitcoin":  "BTCUSDT",
    "ethereum": "ETHUSDT",
    "bnb":      "BNBUSDT",
}

def fetch_crypto(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """
    Fetch crypto OHLCV from Binance public API. No API key required.
    """
    url    = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol":   symbol.upper(),
        "interval": "1d",
        "limit":    min(days, 1000),
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"[Binance] {symbol} → HTTP {r.status_code} — skipped")
            return pd.DataFrame()

        cols = ["open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_base",
                "taker_quote","ignore"]
        df = pd.DataFrame(r.json(), columns=cols)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("date", inplace=True)
        df = df[["open","high","low","close","volume"]].astype(float)
        df.sort_index(inplace=True)
        print(f"[Binance] ✓ {symbol} | {len(df)} days")
        return df

    except Exception as e:
        print(f"[Binance] {symbol} failed — {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  4. News Sentiment — Yahoo RSS + NewsAPI
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_sentiment(ticker: str, api_key: str = None) -> dict:
    """
    Fetch recent news headlines and compute a simple sentiment score.

    Tries NewsAPI first (needs free key), falls back to Yahoo Finance RSS.
    """
    key = api_key or NEWS_API_KEY
    if key and key != "YOUR_NEWSAPI_KEY":
        result = _fetch_newsapi(ticker, key)
        if result["n_articles"] > 0:
            return result
    return _fetch_yahoo_rss(ticker)


def _fetch_newsapi(ticker: str, api_key: str) -> dict:
    url    = "https://newsapi.org/v2/everything"
    params = {
        "q":        ticker,
        "apiKey":   api_key,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 20,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {"sentiment_score": 0, "n_articles": 0, "headlines": [], "source": "none"}
        articles  = r.json().get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]
        score     = _score_headlines(headlines)
        print(f"[NewsAPI] ✓ {ticker} | {len(headlines)} articles | sentiment={score:.3f}")
        return {"sentiment_score": round(score, 4), "n_articles": len(headlines),
                "headlines": headlines[:5], "source": "newsapi"}
    except Exception as e:
        print(f"[NewsAPI] Failed — {e}")
        return {"sentiment_score": 0, "n_articles": 0, "headlines": [], "source": "none"}


def _fetch_yahoo_rss(ticker: str) -> dict:
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return {"sentiment_score": 0, "n_articles": 0, "headlines": [], "source": "none"}
        import re
        titles    = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
        titles   += re.findall(r"<title>(.*?)</title>", r.text)
        headlines = [t for t in titles if ticker.upper() not in t and "Yahoo" not in t][:15]
        score     = _score_headlines(headlines)
        print(f"[Yahoo RSS] ✓ {ticker} | {len(headlines)} headlines | sentiment={score:.3f}")
        return {"sentiment_score": round(score, 4), "n_articles": len(headlines),
                "headlines": headlines[:5], "source": "yahoo_rss"}
    except Exception as e:
        print(f"[Yahoo RSS] Failed — {e}")
        return {"sentiment_score": 0, "n_articles": 0, "headlines": [], "source": "none"}


def _score_headlines(headlines: list) -> float:
    positive = ["surge", "rally", "gain", "rise", "beat", "profit", "growth",
                "record", "strong", "up", "high", "positive", "bull", "buy",
                "upgrade", "outperform", "boost", "soar", "jump", "exceed"]
    negative = ["fall", "drop", "loss", "decline", "miss", "weak", "down",
                "low", "negative", "bear", "sell", "downgrade", "crash",
                "plunge", "slump", "cut", "reduce", "risk", "concern", "fear"]
    pos_count = neg_count = 0
    for h in headlines:
        h_lower = h.lower()
        pos_count += sum(w in h_lower for w in positive)
        neg_count += sum(w in h_lower for w in negative)
    total = pos_count + neg_count + 1e-9
    return (pos_count - neg_count) / total


# ─────────────────────────────────────────────────────────────────────────────
#  5. NSE India — live quote data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nse_data(symbol: str) -> dict:
    """
    Fetch live quote from NSE India. Works for both .BO and .NS tickers —
    strip the suffix before passing (handled automatically in fetch_all).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":     "application/json",
        "Referer":    "https://www.nseindia.com",
    }
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=8)
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol.upper()}"
        r   = session.get(url, headers=headers, timeout=8)
        if r.status_code != 200:
            print(f"[NSE] {symbol} → HTTP {r.status_code} — skipped")
            return {}
        data    = r.json()
        pd_data = data.get("priceInfo", {})
        result  = {
            "last_price":  pd_data.get("lastPrice"),
            "change":      pd_data.get("change"),
            "pct_change":  pd_data.get("pChange"),
            "week52_high": pd_data.get("weekHighLow", {}).get("max"),
            "week52_low":  pd_data.get("weekHighLow", {}).get("min"),
            "volume":      data.get("securityInfo", {}).get("issuedSize"),
        }
        print(f"[NSE] ✓ {symbol} | ₹{result['last_price']} ({result['pct_change']}%)")
        return result
    except Exception as e:
        print(f"[NSE] {symbol} failed — {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  6. Master fetcher — pulls everything and merges
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all(ticker: str, period: str = DEFAULT_PERIOD,
              include_macro: bool  = True,
              include_crypto: bool = False,
              include_news: bool   = True,
              include_nse: bool    = None) -> dict:
    """
    Master data fetcher — pulls from all available sources and merges.
    """
    result = {
        "price_df":  pd.DataFrame(),
        "macro_df":  pd.DataFrame(),
        "sentiment": {"sentiment_score": 0, "n_articles": 0, "source": "none"},
        "nse_live":  {},
        "crypto_df": pd.DataFrame(),
        "merged_df": pd.DataFrame(),
    }

    # ── 1. Price ─────────────────────────────────────────────
    try:
        result["price_df"] = fetch_stock_data(ticker, period=period)
    except Exception as e:
        print(f"[DataFetcher] Price fetch failed: {e}")
        return result

    # ── 2. Macro ─────────────────────────────────────────────
    if include_macro:
        result["macro_df"] = fetch_fred_macro()

    # ── 3. News Sentiment ─────────────────────────────────────
    if include_news:
        # Strip exchange suffix for news search — "ZOMATO.BO" → "ZOMATO"
        clean_ticker = ticker.replace(".BO", "").replace(".NS", "").replace("^", "")
        result["sentiment"] = fetch_news_sentiment(clean_ticker)

    # ── 4. NSE live (auto for Indian tickers) ────────────────
    is_indian = ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")
    if include_nse or (include_nse is None and is_indian):
        clean = ticker.replace(".NS", "").replace(".BO", "").replace("^", "")
        result["nse_live"] = fetch_nse_data(clean)

    # ── 5. Crypto (BTC as market sentiment proxy) ────────────
    if include_crypto:
        result["crypto_df"] = fetch_crypto("BTCUSDT", days=365)

    # ── 6. Merge everything ───────────────────────────────────
    result["merged_df"] = _merge_all(
        result["price_df"],
        result["macro_df"],
        result["crypto_df"],
        result["sentiment"],
    )

    return result


def _merge_all(price_df: pd.DataFrame, macro_df: pd.DataFrame,
               crypto_df: pd.DataFrame, sentiment: dict) -> pd.DataFrame:
    """Merge all data sources onto the price_df date index."""
    merged = price_df.copy()

    if not macro_df.empty:
        macro_aligned = macro_df.reindex(merged.index, method="ffill")
        merged = merged.join(macro_aligned, how="left")

    if not crypto_df.empty:
        btc = crypto_df[["close"]].rename(columns={"close": "btc_close"})
        btc_aligned = btc.reindex(merged.index, method="ffill")
        merged = merged.join(btc_aligned, how="left")

    merged["news_sentiment"] = sentiment.get("sentiment_score", 0)
    merged.ffill(inplace=True)
    merged.dropna(inplace=True)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to price DataFrame."""
    df = df.copy()

    df["returns"]    = df["close"].pct_change()
    df["ma_5"]       = df["close"].rolling(5).mean()
    df["ma_20"]      = df["close"].rolling(20).mean()
    df["volatility"] = df["returns"].rolling(10).std()

    delta       = df["close"].diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12       = df["close"].ewm(span=12).mean()
    ema26       = df["close"].ewm(span=26).mean()
    df["macd"]  = ema12 - ema26

    sma20           = df["close"].rolling(20).mean()
    std20           = df["close"].rolling(20).std()
    df["bb_width"]  = (2 * std20) / (sma20 + 1e-9)

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Scenario engine
# ─────────────────────────────────────────────────────────────────────────────

def apply_scenario(df: pd.DataFrame, scenario: dict | None) -> pd.DataFrame:
    """Override last row of DataFrame with what-if values."""
    if not scenario:
        return df
    df  = df.copy()
    idx = df.index[-1]
    for var, val in scenario.items():
        if var in df.columns:
            print(f"[Scenario] {var}: {df.loc[idx, var]:.4f} → {val}")
            df.loc[idx, var] = val
        else:
            print(f"[Scenario] '{var}' not found — skipped")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Normalize feature columns to [0, 1]."""
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError(
            f"None of {feature_cols} found in DataFrame columns: {df.columns.tolist()}"
        )
    data    = df[cols].values.astype(float)
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    return (data - col_min) / (col_max - col_min + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Testing Indian stock fetch ===")
    try:
        df = fetch_stock_data("ZOMATO.BO")
        print(f"ZOMATO.BO: {len(df)} bars, latest close = {df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"ZOMATO.BO fetch failed: {e}")

    print("\n=== Testing sentiment scoring (no network needed) ===")
    test_headlines = [
        "Apple stock surges to record high on strong earnings beat",
        "Markets rally as Fed signals rate cuts",
        "Tesla shares drop on weak delivery numbers",
    ]
    score = _score_headlines(test_headlines)
    print(f"Sentiment score: {score:.3f}  (positive=good)")

    print("\n=== Testing feature engineering (synthetic data) ===")
    np.random.seed(42)
    dates  = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = 150 + np.cumsum(np.random.randn(100))
    df_syn = pd.DataFrame({
        "open": prices * 0.99, "high": prices * 1.01,
        "low":  prices * 0.98, "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, 100).astype(float),
    }, index=dates)
    df_feat = add_features(df_syn)
    print(f"Features: {[c for c in df_feat.columns if c not in ['open','high','low','close','volume']]}")
    print(f"Shape: {df_feat.shape}")

    X = normalize_features(df_feat, ["returns", "volatility", "rsi", "macd"])
    print(f"Normalized shape: {X.shape} | range: {X.min():.3f} - {X.max():.3f}")
    print("data_fetcher.py OK")