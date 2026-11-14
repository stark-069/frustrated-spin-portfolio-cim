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
#  File        : intent_parser.py
#  Author      : B R Nikilesh
#  Description : Parse a natural language prompt into a structured intent dict.
#                Detects: task type, ticker symbol, time horizon, what-if overrides.
#------------------------------------------------------------------------------

import re
import yfinance as yf

# ─── Supported tasks and their trigger keywords ───────────────────────────────
TASKS = {
    "stock_prediction":       ["predict", "price", "stock", "stocks", "forecast",
                                "trend", "movement", "will go up", "will go down",
                                "next week", "next day", "direction", "prize", "share",
                                "shares", "worth", "value", "analyse", "analyze"],
    "portfolio_optimization": ["portfolio", "optimize", "allocate", "weights",
                                "diversify", "best mix", "allocation", "select assets"],
    "options_pricing":        ["option", "call", "put", "derivative", "black scholes",
                                "expiry", "strike", "contract"],
    "risk_modeling":          ["risk", "var", "value at risk", "volatility", "drawdown",
                                "sharpe", "exposure", "sortino", "cvar"],
    "systemic_risk":          ["contagion", "systemic", "cascade", "collapse", "default", 
                                "bank failure"]
}

# ─── Fast-path: common company names → ticker (no network call needed) ────────
# NOTE: Indian stocks use .BO (BSE) — Yahoo Finance has better coverage than .NS
KNOWN_TICKERS = {
    # ── US ────────────────────────────────────────────────────────────────────
    "apple":        "AAPL",     "microsoft":    "MSFT",     "google":       "GOOGL",
    "alphabet":     "GOOGL",    "amazon":       "AMZN",     "tesla":        "TSLA",
    "nvidia":       "NVDA",     "meta":         "META",     "netflix":      "NFLX",
    "amd":          "AMD",      "intel":        "INTC",     "salesforce":   "CRM",
    "uber":         "UBER",     "airbnb":       "ABNB",     "spotify":      "SPOT",
    "jpmorgan":     "JPM",      "goldman":      "GS",       "visa":         "V",
    # ── Indian (BSE — .BO has better Yahoo Finance coverage than .NS) ─────────
    "zomato":           "ZOMATO.BO",
    "reliance":         "RELIANCE.BO",
    "tcs":              "TCS.BO",
    "infosys":          "INFY.BO",
    "wipro":            "WIPRO.BO",
    "hdfc":             "HDFCBANK.BO",
    "icici":            "ICICIBANK.BO",
    "sbi":              "SBIN.BO",
    "kotak":            "KOTAKBANK.BO",
    "adani":            "ADANIENT.BO",
    "tata motors":      "TATAMOTORS.BO",
    "tatamotors":       "TATAMOTORS.BO",
    "tata steel":       "TATASTEEL.BO",
    "bajaj":            "BAJFINANCE.BO",
    "bajaj finance":    "BAJFINANCE.BO",
    "ongc":             "ONGC.BO",
    "ntpc":             "NTPC.BO",
    "powergrid":        "POWERGRID.BO",
    "sun pharma":       "SUNPHARMA.BO",
    "sunpharma":        "SUNPHARMA.BO",
    "dr reddy":         "DRREDDY.BO",
    "cipla":            "CIPLA.BO",
    "hul":              "HINDUNILVR.BO",
    "hindustan unilever": "HINDUNILVR.BO",
    "itc":              "ITC.BO",
    "maruti":           "MARUTI.BO",
    "paytm":            "PAYTM.BO",
    "nykaa":            "NYKAA.BO",
    "swiggy":           "SWIGGY.BO",
    "airtel":           "BHARTIARTL.BO",
    "bharti airtel":    "BHARTIARTL.BO",
    "asian paints":     "ASIANPAINT.BO",
    "ltimindtree":      "LTIM.BO",
    "hcl":              "HCLTECH.BO",
    "tech mahindra":    "TECHM.BO",
    "axis bank":        "AXISBANK.BO",
    "ultratech":        "ULTRACEMCO.BO",
    "nestle":           "NESTLEIND.BO",
    "dmart":            "DMART.BO",
    # ── Indices & commodities ─────────────────────────────────────────────────
    "nifty":        "^NSEI",
    "sensex":       "^BSESN",
    "gold":         "GC=F",
    "crude":        "CL=F",
    "oil":          "CL=F",
    "silver":       "SI=F",
}

# Words to strip before sending a name to Yahoo Search
_NOISE = re.compile(
    r'\b(predict|price|stock|stocks|share|shares|analyze|analysis|analyse|'
    r'risk|forecast|trend|what is|tell me|about|the|of|for|'
    r'prize|value|worth|today|tomorrow|next|week|month|year|me|'
    r'give|show|check|fetch|get|how|much|is|are|will|going|to)\b',
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  Live ticker resolver  (uses yfinance — already a dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ticker_live(query: str) -> str | None:
    """
    Ask Yahoo Finance to resolve a free-text company name to a ticker.
    Returns the best-match symbol or None if nothing found.
    Called only when KNOWN_TICKERS has no match — one network call.
    """
    query = query.strip()
    if not query or len(query) < 2:
        return None
    try:
        results = yf.Search(query, max_results=5).quotes
        if not results:
            return None
        # Prefer equity results over crypto / futures / index
        for r in results:
            if r.get("quoteType", "") in ("EQUITY", "ETF"):
                ticker = r.get("symbol", "")
                name   = r.get("shortname", r.get("longname", ""))
                print(f"[Intent] '{query}' → {ticker} ({name})")
                return ticker
        # Fallback: take the first result regardless of type
        ticker = results[0].get("symbol", "")
        print(f"[Intent] '{query}' → {ticker} (best guess)")
        return ticker
    except Exception as e:
        print(f"[Intent] Live ticker lookup failed for '{query}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Main parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_intent(prompt: str) -> dict:
    """
    Parse a natural language prompt.

    Returns
    -------
    {
        "task"       : str,           # one of TASKS keys
        "ticker"     : str,           # primary stock symbol
        "tickers"    : list[str],     # all tickers found (for portfolio)
        "horizon"    : str,           # "1d" | "1w" | "1mo" | "1y"
        "scenario"   : dict | None,   # {"close": 150.0, ...} or None
        "option_type": str,           # "call" | "put"
        "strike"     : float | None,  # strike price if mentioned
    }
    """
    lower = prompt.lower()
    return {
        "task":         _detect_task(lower),
        "ticker":       _extract_primary_ticker(prompt),
        "tickers":      _extract_all_tickers(prompt),
        "horizon":      _extract_horizon(lower),
        "scenario":     _extract_scenario(lower),
        "option_type":  "put" if "put" in lower else "call",
        "strike":       _extract_strike(lower),
        "risk_penalty": _extract_risk_penalty(lower), # Add this line
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Sub-parsers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_task(lower: str) -> str:
    scores = {task: 0 for task in TASKS}
    for task, keywords in TASKS.items():
        for kw in keywords:
            if kw in lower:
                scores[task] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "stock_prediction"


def _extract_primary_ticker(prompt: str) -> str:
    lower = prompt.lower()

    # 1. Hardcoded map — instant, no network
    for name, ticker in KNOWN_TICKERS.items():
        if name in lower:
            return ticker

    # 2. Explicit uppercase ticker typed by user (e.g. AAPL, TSLA)
    match = re.search(r'\b([A-Z]{2,5})\b', prompt)
    if match:
        sym = match.group(1)
        if sym not in ("IF", "OR", "AND", "FOR", "THE", "IS", "IN", "ON"):
            return sym

    # 3. Strip noise words and send remainder to Yahoo Search
    cleaned = _NOISE.sub("", lower).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned:
        ticker = _resolve_ticker_live(cleaned)
        if ticker:
            return ticker

    # 4. Last resort default
    return "AAPL"


def _extract_all_tickers(prompt: str) -> list:
    found = []
    lower = prompt.lower()

    # From hardcoded map
    for name, ticker in KNOWN_TICKERS.items():
        if name in lower and ticker not in found:
            found.append(ticker)

    # Explicit uppercase symbols in prompt
    for m in re.finditer(r'\b([A-Z]{2,5})\b', prompt):
        sym = m.group(1)
        if sym not in found and sym not in ("IF", "OR", "AND", "FOR", "THE", "IS"):
            found.append(sym)

    # Nothing found — try live resolve
    if not found:
        ticker = _extract_primary_ticker(prompt)
        if ticker:
            found.append(ticker)

    return found if found else ["AAPL", "MSFT", "GOOGL", "AMZN"]


def _extract_horizon(lower: str) -> str:
    if any(w in lower for w in ["next week", "1 week", "weekly", "7 day"]):
        return "1w"
    if any(w in lower for w in ["next month", "1 month", "monthly", "30 day"]):
        return "1mo"
    if any(w in lower for w in ["next year", "1 year", "yearly", "annual"]):
        return "1y"
    if any(w in lower for w in ["tomorrow", "next day", "daily", "1 day"]):
        return "1d"
    return "1w"


def _extract_scenario(lower: str) -> dict | None:
    """Detect what-if variable overrides."""
    scenario = {}
    patterns = [
        r"what if (\w+)\s+(?:is|becomes|drops to|rises to|goes to|hits|reaches)\s+([\d.]+)",
        r"if (\w+)\s+(?:is|becomes|drops to|rises to|goes to|hits|reaches)\s+([\d.]+)",
        r"assume (\w+)\s+(?:is|=|equals)\s+([\d.]+)",
        r"set (\w+)\s+to\s+([\d.]+)",
    ]
    for pat in patterns:
        for var, val in re.findall(pat, lower):
            scenario[var.strip()] = float(val)
    return scenario if scenario else None


def _extract_strike(lower: str) -> float | None:
    m = re.search(r'strike\s+(?:price\s+)?(?:of\s+)?([\d.]+)', lower)
    if m:
        return float(m.group(1))
    m = re.search(r'([\d.]+)\s+strike', lower)
    return float(m.group(1)) if m else None


def _extract_risk_penalty(lower: str) -> float:
    """
    Extract risk aversion factor (gamma). 
    Defaults to 2.0 if not specified.
    """
    # Look for explicit definitions like gamma=5 or risk penalty of 4.5
    m = re.search(r'(?:gamma|penalty|aversion)\s*(?:=|\bof\b)\s*([\d.]+)', lower)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
            
    # Look for profile keywords
    if any(w in lower for w in ["conservative", "safe", "low risk", "low volatility"]):
        return 8.0
    if any(w in lower for w in ["aggressive", "high risk", "growth optimized"]):
        return 0.5
        
    return 2.0
# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "zomato stocks",
        "swiggy share price",
        "volkswagen stock forecast",
        "samsung share value",
        "What is the risk of holding Tesla and NVDA?",
        "Optimize my portfolio with AAPL MSFT GOOGL AMZN",
        "Price a call option on TSLA with strike 250",
        "What if close drops to 150, will AAPL go up?",
        "Analyze risk for Reliance",
        "louis vuitton stock",
        "sony share price",
        "Simulate systemic contagion if Silicon Valley Bank defaults" # NEW TEST
    ]
    for t in tests:
        print(f"\nPrompt : {t}")
        print(f"Parsed : {parse_intent(t)}")