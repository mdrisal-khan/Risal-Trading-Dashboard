import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# RISAL TRADING DASHBOARD V9.0
# Spot-only • Multi-Timeframe • Regime Filter • A+ Trade Engine
# Binance public API • Narrative Breadth • ATR Risk/Reward
# ============================================================

st.set_page_config(
    page_title="Risal Trading Dashboard V9",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.stApp { background:#0b0e14; color:#d7dee8; }
section[data-testid="stSidebar"] { background:#11161d; }
div[data-testid="stMetric"] {
    background:#151b23; border:1px solid #2b3440;
    border-radius:10px; padding:12px 16px;
}
div[data-testid="stDataFrame"] {
    border:1px solid #2b3440; border-radius:10px; overflow:hidden;
}
.stButton>button { border-radius:8px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Risal Trading Dashboard V9")

st.markdown("### 💰 Risk & Position Calculator")
top_c1, top_c2, top_c3 = st.columns(3)

portfolio_balance = top_c1.number_input(
    "💰 Portfolio Balance (USDT)",
    min_value=1.0,
    value=500.0,
    step=10.0,
    key="portfolio_balance_main",
    help="Enter your current total spot portfolio balance. You can change it anytime.",
)

max_allowed_risk = 2.0 if portfolio_balance < 1000 else 1.0

risk_pct = top_c2.number_input(
    "🎯 Risk per Trade (%)",
    min_value=0.1,
    max_value=max_allowed_risk,
    value=max_allowed_risk,
    step=0.1,
    key="risk_pct_main",
    help="Hard limit: below $1,000 max 2%; $1,000 or above max 1%.",
)

max_risk_usdt = portfolio_balance * risk_pct / 100.0

top_c3.metric("🛑 Maximum Loss at SL", f"${max_risk_usdt:,.2f}")

st.caption(
    f"Discipline rule: portfolio < $1,000 → maximum {max_allowed_risk:.0f}% risk | "
    f"portfolio ≥ $1,000 → maximum 1% risk."
)

st.caption(
    "Spot Trade Radar • Multi-Timeframe Confirmation • Narrative Intelligence • "
    "Pre-Breakout • ATR Risk Engine"
)

# ============================================================
# SETTINGS
# ============================================================

BINANCE_BASE_URLS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://data-api.binance.vision",
]

TICKER_PATH = "/api/v3/ticker/24hr"
KLINE_PATH = "/api/v3/klines"
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

REQUEST_TIMEOUT = 12
API_RETRIES = 2
API_BACKOFF = 0.8
MAX_WORKERS = 8

MIN_QUOTE_VOLUME = 2_000_000
DEEP_SCAN_LIMIT = 80

SMALL_CAP_MIN = 10_000_000
SMALL_CAP_MAX = 500_000_000

SIGNAL_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "signal_history_v9.csv",
)

A_PLUS_SCORE = 85
A_SCORE = 78
WATCH_SCORE = 68
MIN_RR = 2.0

NARRATIVE_MAP = {
    "Artificial Intelligence (AI)": [
        "FET", "RENDER", "TAO", "NEAR", "AKT", "AR", "IO",
        "AIOZ", "GRASS", "VIRTUAL", "WLD",
    ],
    "Real World Assets (RWA)": [
        "ONDO", "PENDLE", "MKR", "LINK", "POLYX", "CPOOL", "TRU", "XDC",
    ],
    "DeFi": [
        "UNI", "AAVE", "MKR", "CRV", "LDO", "COMP", "SNX",
        "DYDX", "PENDLE", "JUP", "RAY", "SUSHI",
    ],
    "DEX": [
        "UNI", "JUP", "DYDX", "RAY", "SUSHI", "CAKE", "GMX", "ORCA",
    ],
    "Layer 1": [
        "ETH", "SOL", "BNB", "ADA", "AVAX", "SUI", "APT",
        "NEAR", "ATOM", "TRX", "SEI", "TON", "XRP",
    ],
    "Layer 2 / Rollup": [
        "ARB", "OP", "STRK", "ZK", "MANTA", "METIS", "IMX", "MNT",
    ],
    "DePIN": [
        "RENDER", "FIL", "AR", "HNT", "THETA", "AKT", "IO", "GRASS", "AIOZ",
    ],
    "Gaming / GameFi": [
        "IMX", "GALA", "SAND", "MANA", "AXS", "BEAM", "RON", "PIXEL", "SUPER",
    ],
    "Meme": [
        "DOGE", "SHIB", "PEPE", "BONK", "FLOKI", "WIF", "BRETT", "MEME",
    ],
    "Oracle": ["LINK", "PYTH", "API3", "BAND", "UMA"],
    "Liquid Staking / Restaking": ["LDO", "RPL", "ETHFI", "EIGEN", "REZ"],
    "Infrastructure": ["LINK", "FIL", "AR", "ICP", "TIA", "ATOM", "QNT", "GRT"],
}

# ============================================================
# HELPERS
# ============================================================

def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def get_bd_time():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Dhaka"))
    except Exception:
        return datetime.now()


def fmt_price(x):
    x = safe_float(x, 0)
    if x >= 1000:
        return f"${x:,.2f}"
    if x >= 1:
        return f"${x:,.4f}"
    return f"${x:,.6f}"


def request_binance(path, params=None, retries=API_RETRIES):
    diagnostics = []

    for base in BINANCE_BASE_URLS:
        url = base + path

        for attempt in range(retries + 1):
            try:
                r = requests.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": "Risal-Trading-Dashboard-V9/1.0"},
                )

                diagnostics.append(f"{base} → HTTP {r.status_code}")

                if 200 <= r.status_code < 300:
                    try:
                        return r.json(), diagnostics
                    except Exception:
                        break

                if r.status_code in (403, 418):
                    break

                retryable = r.status_code in (429, 500, 502, 503, 504)

                if not retryable or attempt >= retries:
                    break

                retry_after = safe_float(
                    r.headers.get("Retry-After"),
                    np.nan,
                )

                wait = (
                    retry_after
                    if np.isfinite(retry_after)
                    else API_BACKOFF * (2 ** attempt)
                )

                time.sleep(min(max(wait, 0.25), 6.0))

            except requests.RequestException as e:
                diagnostics.append(f"{base} → {type(e).__name__}")

                if attempt < retries:
                    time.sleep(min(API_BACKOFF * (2 ** attempt), 6.0))

            except Exception as e:
                diagnostics.append(f"{base} → {type(e).__name__}")
                break

    return None, diagnostics


# ============================================================
# INDICATORS
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))

    return out.fillna(50)


def macd(s):
    fast = ema(s, 12)
    slow = ema(s, 26)

    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal

    return line, signal, hist


def atr(df, n=14):
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1/n,
        adjust=False,
        min_periods=n,
    ).mean()


def adx(df, n=14):
    high, low, close = df["high"], df["low"], df["close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where(
        (up > down) & (up > 0),
        up,
        0.0,
    )

    minus_dm = np.where(
        (down > up) & (down > 0),
        down,
        0.0,
    )

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_n = tr.ewm(
        alpha=1/n,
        adjust=False,
    ).mean()

    plus_di = (
        100
        * pd.Series(
            plus_dm,
            index=df.index,
        ).ewm(
            alpha=1/n,
            adjust=False,
        ).mean()
        / atr_n
    )

    minus_di = (
        100
        * pd.Series(
            minus_dm,
            index=df.index,
        ).ewm(
            alpha=1/n,
            adjust=False,
        ).mean()
        / atr_n
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1/n,
        adjust=False,
    ).mean().fillna(0)


def enrich(df):
    d = df.copy()

    d["EMA20"] = ema(d["close"], 20)
    d["EMA50"] = ema(d["close"], 50)
    d["EMA200"] = ema(d["close"], 200)

    d["RSI"] = rsi(d["close"], 14)
    d["ATR"] = atr(d, 14)
    d["ATR%"] = d["ATR"] / d["close"] * 100

    d["ADX"] = adx(d, 14)

    d["MACD"], d["MACDSignal"], d["MACDHist"] = macd(d["close"])

    d["VolMA20"] = d["volume"].rolling(20).mean()
    d["VolumeRatio"] = d["volume"] / d["VolMA20"].replace(0, np.nan)

    return d


# ============================================================
# BINANCE DATA
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def get_tickers():
    data, diag = request_binance(TICKER_PATH)

    if not isinstance(data, list):
        return pd.DataFrame(), diag

    rows = []

    for x in data:
        symbol = str(x.get("symbol", "")).upper()

        if not symbol.endswith("USDT"):
            continue

        price = safe_float(x.get("lastPrice"))
        change = safe_float(x.get("priceChangePercent"))
        qv = safe_float(x.get("quoteVolume"))
        trades = safe_float(x.get("count"), 0)

        if not all(np.isfinite(v) for v in [price, change, qv]):
            continue

        if qv < MIN_QUOTE_VOLUME:
            continue

        rows.append({
            "Symbol": symbol,
            "Coin": symbol[:-4],
            "Price": price,
            "24H %": change,
            "24H Volume": qv,
            "Trades": trades,
        })

    return pd.DataFrame(rows), diag


@st.cache_data(ttl=300, show_spinner=False)
def get_market_caps():
    rows = []

    for page in range(1, 4):
        data = None

        try:
            r = requests.get(
                COINGECKO_URL,
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                    "sparkline": "false",
                },
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Risal-Trading-Dashboard-V9/1.0"},
            )

            if 200 <= r.status_code < 300:
                data = r.json()

        except Exception:
            pass

        if not isinstance(data, list) or not data:
            break

        for x in data:
            coin = str(x.get("symbol", "")).upper().strip()
            cap = safe_float(x.get("market_cap"))

            if coin and np.isfinite(cap) and cap > 0:
                rows.append({
                    "Coin": coin,
                    "Market Cap": cap,
                })

        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(columns=["Coin", "Market Cap"])

    return (
        pd.DataFrame(rows)
        .groupby("Coin", as_index=False)["Market Cap"]
        .max()
    )


@st.cache_data(ttl=30, show_spinner=False)
def get_klines(symbol, interval="1h", limit=250):
    data, _ = request_binance(
        KLINE_PATH,
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
        retries=1,
    )

    if not isinstance(data, list) or len(data) < 60:
        return pd.DataFrame()

    rows = []

    for k in data:
        try:
            rows.append({
                "open_time": pd.to_datetime(
                    int(k[0]),
                    unit="ms",
                ),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        except Exception:
            continue

    d = pd.DataFrame(rows)

    if len(d) < 60:
        return pd.DataFrame()

    # Remove current incomplete candle.
    return d.iloc[:-1].reset_index(drop=True)


# ============================================================
# LIVE BTC PRICE
# ============================================================

def get_live_btc_price():
    data, _ = request_binance(
        TICKER_PATH,
        {"symbol": "BTCUSDT"},
        retries=1,
    )

    if isinstance(data, dict):
        return safe_float(data.get("lastPrice"), np.nan)

    return np.nan


# ============================================================
# MARKET REGIME
# ============================================================

def analyze_regime(btc_4h):
    if btc_4h.empty or len(btc_4h) < 80:
        return {
            "Regime": "UNKNOWN",
            "Score": 50,
            "Reason": "BTC 4H data unavailable",
        }

    d = enrich(btc_4h)
    x = d.iloc[-1]

    score = 50
    reasons = []

    if x["close"] > x["EMA200"]:
        score += 15
        reasons.append("BTC > EMA200")
    else:
        score -= 15
        reasons.append("BTC < EMA200")

    if x["EMA20"] > x["EMA50"]:
        score += 10
        reasons.append("EMA20 > EMA50")
    else:
        score -= 10
        reasons.append("EMA20 < EMA50")

    if x["MACDHist"] > 0:
        score += 8
        reasons.append("MACD positive")
    else:
        score -= 8
        reasons.append("MACD negative")

    if x["RSI"] >= 55:
        score += 7
        reasons.append("RSI bullish")
    elif x["RSI"] < 45:
        score -= 7
        reasons.append("RSI weak")

    if x["ADX"] >= 20:
        score += 5
        reasons.append("ADX trend active")

    score = max(0, min(100, score))

    if score >= 72:
        regime = "🟢 RISK-ON"
    elif score <= 38:
        regime = "🔴 RISK-OFF"
    else:
        regime = "🟡 NEUTRAL"

    return {
        "Regime": regime,
        "Score": score,
        "Reason": " • ".join(reasons),
        "Price": x["close"],
        "RSI": x["RSI"],
        "ADX": x["ADX"],
    }


# ============================================================
# MULTI-TIMEFRAME TRADE ENGINE
# ============================================================

def timeframe_score(d, direction="LONG"):
    if d.empty or len(d) < 60:
        return 0, ["No data"]

    x = d.iloc[-1]
    score = 0
    reasons = []

    if direction == "LONG":
        if x["close"] > x["EMA20"]:
            score += 5
            reasons.append("price > EMA20")

        if x["EMA20"] > x["EMA50"]:
            score += 5
            reasons.append("EMA20 > EMA50")

        if x["close"] > x["EMA200"]:
            score += 5
            reasons.append("price > EMA200")

        if x["MACDHist"] > 0:
            score += 5
            reasons.append("MACD+")

        if 52 <= x["RSI"] <= 72:
            score += 5
            reasons.append("healthy RSI")

        if x["ADX"] >= 18:
            score += 5
            reasons.append("trend strength")

        if x["VolumeRatio"] >= 1.05:
            score += 5
            reasons.append("volume active")

    else:
        if x["close"] < x["EMA20"]:
            score += 5

        if x["EMA20"] < x["EMA50"]:
            score += 5

        if x["close"] < x["EMA200"]:
            score += 5

        if x["MACDHist"] < 0:
            score += 5

        if 28 <= x["RSI"] <= 48:
            score += 5

        if x["ADX"] >= 18:
            score += 5

        if x["VolumeRatio"] >= 1.05:
            score += 5

    return score, reasons


def swing_levels(d, lookback=30):
    if d.empty:
        return np.nan, np.nan

    recent = d.tail(lookback)

    support = recent["low"].min()
    resistance = recent["high"].max()

    return support, resistance


def detect_long_setup(d1h, d15m):
    if d1h.empty or d15m.empty:
        return {
            "Trigger": "WAIT",
            "SetupScore": 0,
            "Entry": np.nan,
            "Stop": np.nan,
            "TP1": np.nan,
            "TP2": np.nan,
            "RR": np.nan,
            "Reason": "Insufficient data",
        }

    h = enrich(d1h)
    m = enrich(d15m)

    hx = h.iloc[-1]
    mx = m.iloc[-1]

    support, resistance = swing_levels(h, 35)
    atr_v = safe_float(hx["ATR"])

    if not np.isfinite(atr_v) or atr_v <= 0:
        return {
            "Trigger": "WAIT",
            "SetupScore": 0,
            "Entry": np.nan,
            "Stop": np.nan,
            "TP1": np.nan,
            "TP2": np.nan,
            "RR": np.nan,
            "Reason": "ATR unavailable",
        }

    price = hx["close"]

    score = 0
    reasons = []

    # 1H structure
    if price > hx["EMA20"] > hx["EMA50"]:
        score += 15
        reasons.append("1H bullish EMA structure")

    if price > hx["EMA200"]:
        score += 8
        reasons.append("above 1H EMA200")

    if hx["MACDHist"] > 0:
        score += 8
        reasons.append("1H MACD positive")

    if 52 <= hx["RSI"] <= 70:
        score += 7
        reasons.append("1H RSI healthy")

    if hx["ADX"] >= 20:
        score += 7
        reasons.append("1H ADX trend")

    # 15m trigger context
    if mx["close"] > mx["EMA20"]:
        score += 5
        reasons.append("15m above EMA20")

    if mx["MACDHist"] > 0:
        score += 5
        reasons.append("15m MACD positive")

    if mx["RSI"] >= 50:
        score += 5
        reasons.append("15m momentum positive")

    if mx["VolumeRatio"] >= 1.15:
        score += 8
        reasons.append("15m volume expansion")

    # Breakout proximity / breakout confirmation
    distance = (
        ((resistance - price) / price) * 100
        if price
        else np.nan
    )

    breakout = price > resistance * 1.001

    near_resistance = (
        np.isfinite(distance)
        and -1.0 <= distance <= 2.5
    )

    trigger = "WAIT"
    entry = np.nan
    stop = np.nan
    tp1 = np.nan
    tp2 = np.nan
    rr = np.nan

    if (
        breakout
        and mx["close"] > mx["EMA20"]
        and mx["VolumeRatio"] >= 1.15
    ):
        score += 15
        reasons.append("confirmed breakout + volume")

        trigger = "🟢 BREAKOUT ENTRY"
        entry = price

    elif (
        near_resistance
        and mx["close"] > mx["EMA20"]
        and mx["MACDHist"] > 0
    ):
        score += 10
        reasons.append("coiling near resistance")

        trigger = "🟡 BREAKOUT WATCH"
        entry = resistance * 1.002

    elif (
        price > hx["EMA20"]
        and abs(price - hx["EMA20"]) / price < 0.012
    ):
        score += 8
        reasons.append("pullback near EMA20")

        trigger = "🟡 PULLBACK WATCH"
        entry = price

    if np.isfinite(entry):
        stop = min(
            entry - 1.20 * atr_v,
            support * 0.995
            if np.isfinite(support)
            else entry - 1.20 * atr_v,
        )

        risk = entry - stop

        if risk > 0:
            tp1 = entry + 2.0 * risk
            tp2 = entry + 3.0 * risk
            rr = (tp1 - entry) / risk

            risk_pct = risk / entry * 100

            if risk_pct > 6.0:
                score -= 12
                reasons.append("risk too wide")

    score = max(0, min(100, score))

    if (
        score >= A_PLUS_SCORE
        and trigger == "🟢 BREAKOUT ENTRY"
        and np.isfinite(rr)
        and rr >= MIN_RR
    ):
        label = "🔥 A+ SPOT SETUP"

    elif (
        score >= A_SCORE
        and trigger != "WAIT"
        and np.isfinite(rr)
        and rr >= MIN_RR
    ):
        label = "🟢 A SETUP"

    elif score >= WATCH_SCORE:
        label = "🟡 WATCH"

    else:
        label = "⚪ NO TRADE"

    return {
        "Trigger": label,
        "SetupScore": score,
        "Entry": entry,
        "Stop": stop,
        "TP1": tp1,
        "TP2": tp2,
        "RR": rr,
        "Resistance": resistance,
        "Support": support,
        "ATR": atr_v,
        "Reason": " • ".join(reasons),
    }


@st.cache_data(ttl=45, show_spinner=False)
def analyze_symbol(symbol):
    d15 = get_klines(symbol, "15m", 220)
    d1h = get_klines(symbol, "1h", 220)
    d4h = get_klines(symbol, "4h", 220)
    d1d = get_klines(symbol, "1d", 220)

    if min(len(d15), len(d1h), len(d4h), len(d1d)) < 60:
        return None

    s15, _ = timeframe_score(enrich(d15), "LONG")
    s1h, _ = timeframe_score(enrich(d1h), "LONG")
    s4h, _ = timeframe_score(enrich(d4h), "LONG")
    s1d, _ = timeframe_score(enrich(d1d), "LONG")

    setup = detect_long_setup(d1h, d15)

    e1 = enrich(d1h).iloc[-1]
    e4 = enrich(d4h).iloc[-1]
    ed = enrich(d1d).iloc[-1]

    mtf_score = (
        s15 * 0.15
        + s1h * 0.30
        + s4h * 0.30
        + s1d * 0.25
    )

    high_tf_bull = (
        e4["close"] > e4["EMA50"]
        and ed["close"] > ed["EMA50"]
    )

    if not high_tf_bull:
        mtf_score *= 0.75

    final_score = round(
        min(
            100,
            max(
                0,
                mtf_score * 0.55
                + setup["SetupScore"] * 0.45,
            ),
        ),
        1,
    )

    if (
        final_score >= A_PLUS_SCORE
        and setup["Trigger"] == "🔥 A+ SPOT SETUP"
    ):
        grade = "🔥 A+"

    elif (
        final_score >= A_SCORE
        and setup["Trigger"]
        in ("🟢 A SETUP", "🔥 A+ SPOT SETUP")
    ):
        grade = "🟢 A"

    elif final_score >= WATCH_SCORE:
        grade = "🟡 WATCH"

    else:
        grade = "⚪ SKIP"

    return {
        "Symbol": symbol,
        "Price": e1["close"],
        "MTF Score": round(mtf_score, 1),
        "Setup Score": setup["SetupScore"],
        "Final Score": final_score,
        "Grade": grade,
        "15m": s15,
        "1H": s1h,
        "4H": s4h,
        "1D": s1d,
        "RSI 1H": round(e1["RSI"], 1),
        "ADX 1H": round(e1["ADX"], 1),
        "Volume Ratio 1H": round(e1["VolumeRatio"], 2),
        "ATR % 1H": round(e1["ATR%"], 2),
        "Trigger": setup["Trigger"],
        "Entry": setup["Entry"],
        "Stop": setup["Stop"],
        "TP1": setup["TP1"],
        "TP2": setup["TP2"],
        "RR": setup["RR"],
        "Support": setup["Support"],
        "Resistance": setup["Resistance"],
        "Reason": setup["Reason"],
    }


# ============================================================
# NARRATIVE ENGINE
# ============================================================

def build_narratives(scanner_df):
    rows = []

    for narrative, coins in NARRATIVE_MAP.items():
        available = scanner_df[
            scanner_df["Coin"].isin([c.upper() for c in coins])
        ].copy()

        if available.empty:
            continue

        positive = (available["24H %"] > 0).mean() * 100
        avg_momentum = available["24H %"].mean()
        avg_rs = available["RS vs BTC"].mean()
        avg_vol = available["Volume Ratio"].mean()

        leader = available.sort_values(
            ["Final Score", "RS vs BTC", "Volume Ratio"],
            ascending=False,
        ).iloc[0]

        confirmations = sum([
            positive >= 60,
            avg_rs > 0,
            avg_vol >= 1.15,
            leader["Final Score"] >= 70,
        ])

        if confirmations == 4 and positive >= 60:
            status = "🔥 STRONG"
        elif confirmations >= 3 and positive >= 50:
            status = "🟢 DEVELOPING"
        elif positive >= 60:
            status = "🟡 MOMENTUM"
        else:
            status = "⚪ WEAK"

        score = (
            min(max(positive, 0), 100) * 0.30
            + min(max(avg_rs, 0), 15) / 15 * 25
            + min(max(avg_vol, 0), 2.5) / 2.5 * 20
            + min(max(leader["Final Score"], 0), 100) * 0.25
        )

        rows.append({
            "Narrative": narrative,
            "Status": status,
            "Breadth %": round(positive, 1),
            "Avg 24H %": round(avg_momentum, 2),
            "Avg RS": round(avg_rs, 2),
            "Avg Volume": round(avg_vol, 2),
            "Leader": leader["Symbol"],
            "Leader Score": leader["Final Score"],
            "Confirmations": f"{confirmations}/4",
            "Narrative Score": round(min(score, 100), 1),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("Narrative Score", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# SIGNAL HISTORY
# ============================================================

def load_history():
    if not os.path.exists(SIGNAL_HISTORY_FILE):
        return pd.DataFrame()

    try:
        d = pd.read_csv(SIGNAL_HISTORY_FILE)

        if "Detected At" in d.columns:
            d["Detected At"] = pd.to_datetime(
                d["Detected At"],
                errors="coerce",
            )

        return d

    except Exception:
        return pd.DataFrame()


def save_signals(rows):
    if not rows:
        return

    new = pd.DataFrame(rows)
    old = load_history()

    all_df = (
        pd.concat([old, new], ignore_index=True)
        if not old.empty
        else new
    )

    if "Detected At" in all_df.columns:
        all_df["Detected At"] = all_df["Detected At"].astype(str)

    keys = [
        c
        for c in ["Detected At", "Symbol", "Grade"]
        if c in all_df.columns
    ]

    if keys:
        all_df = all_df.drop_duplicates(
            subset=keys,
            keep="last",
        )

    all_df.to_csv(
        SIGNAL_HISTORY_FILE,
        index=False,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("---")

    st.subheader("💰 Portfolio Risk")

    st.info(
        "Portfolio Balance এবং Risk উপরের main dashboard থেকে change করুন। "
        f"Current: ${portfolio_balance:,.2f} | Risk: {risk_pct:.1f}% | "
        f"Max Loss: ${max_risk_usdt:,.2f}"
    )

    st.header("⚙️ Control Panel")

    refresh = st.selectbox(
        "Refresh interval",
        [30, 60, 120],
        index=0,
        format_func=lambda x: f"{x} seconds",
    )

    min_score = st.slider(
        "Minimum trade score",
        60,
        95,
        A_SCORE,
        1,
    )

    max_symbols = st.slider(
        "Deep scan coins",
        10,
        DEEP_SCAN_LIMIT,
        35,
        5,
    )

    if st.button(
        "🔄 Refresh Now",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    st.info(
        "A+ মানে একাধিক independent confirmation একসাথে পাওয়া গেছে। "
        "এটি guaranteed profit বা financial advice নয়."
    )


# ============================================================
# LOAD MARKET
# ============================================================

df, diagnostics = get_tickers()

if df.empty:
    st.error("Binance market data পাওয়া যায়নি।")

    with st.expander("Connection diagnostics"):
        for x in diagnostics:
            st.write("•", x)

    st.stop()

caps = get_market_caps()

if not caps.empty:
    df = df.merge(
        caps,
        on="Coin",
        how="left",
    )
else:
    df["Market Cap"] = np.nan


# BTC baseline
btc_7d = np.nan

btc1d = get_klines(
    "BTCUSDT",
    "1d",
    20,
)

if not btc1d.empty and len(btc1d) >= 8:
    btc_7d = (
        btc1d["close"].iloc[-1]
        / btc1d["close"].iloc[-8]
        - 1
    ) * 100

df["7D %"] = np.nan
df["RS vs BTC"] = np.nan


# Deep scan pool: liquidity + momentum + narrative + small caps.
narrative_coins = {
    c.upper()
    for values in NARRATIVE_MAP.values()
    for c in values
}

pool = df[
    (df["24H Volume"] >= MIN_QUOTE_VOLUME)
    & (df["24H %"] >= -5)
].copy()

pool["priority"] = (
    pool["24H Volume"].rank(pct=True) * 0.30
    + pool["24H %"].clip(-10, 20).rank(pct=True) * 0.25
    + pool["Coin"].isin(narrative_coins).astype(int) * 0.20
    + (
        pool["Market Cap"]
        .between(SMALL_CAP_MIN, SMALL_CAP_MAX)
        .fillna(False)
        .astype(int)
    ) * 0.15
    + pool["Trades"].rank(pct=True) * 0.10
)

symbols = (
    pool.sort_values(
        "priority",
        ascending=False,
    )
    .head(max_symbols)["Symbol"]
    .tolist()
)


# ============================================================
# BTC REGIME
# ============================================================

btc4h = get_klines(
    "BTCUSDT",
    "4h",
    220,
)

regime = analyze_regime(btc4h)

st.divider()
st.header("🌍 Market Regime")

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "BTC Regime",
    regime["Regime"],
)

r2.metric(
    "Regime Score",
    f"{regime['Score']}/100",
)

with r3:
    @st.fragment(run_every=refresh)
    def live_btc_price_card():
        live_price = get_live_btc_price()

        if np.isfinite(live_price):
            st.metric(
                "BTC Price",
                fmt_price(live_price),
            )
        else:
            st.metric(
                "BTC Price",
                fmt_price(regime.get("Price", np.nan)),
            )

    live_btc_price_card()

r4.metric(
    "BTC RSI 4H",
    f"{regime.get('RSI', np.nan):.1f}"
    if np.isfinite(regime.get("RSI", np.nan))
    else "N/A",
)

st.caption(regime["Reason"])


# ============================================================
# SCAN
# ============================================================

st.divider()
st.header("🎯 A+ Spot Trade Radar")

st.caption(
    "Higher-timeframe trend → 1H structure → 15m trigger → volume → "
    "ATR stop → minimum 2R target."
)

results = []
progress = st.progress(0)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {
        ex.submit(analyze_symbol, s): s
        for s in symbols
    }

    for i, fut in enumerate(
        as_completed(futures),
        start=1,
    ):
        try:
            result = fut.result()

            if result:
                results.append(result)

        except Exception:
            pass

        progress.progress(
            i / max(len(futures), 1)
        )

progress.empty()

trade_df = pd.DataFrame(results)

if trade_df.empty:
    st.warning(
        "এই scan-এ পর্যাপ্ত technical data পাওয়া যায়নি।"
    )
    st.stop()


# Add market data
trade_df = trade_df.merge(
    df[
        [
            "Symbol",
            "Coin",
            "24H %",
            "24H Volume",
            "Market Cap",
            "Trades",
        ]
    ],
    on="Symbol",
    how="left",
)


# RS calculation after merge
trade_df["RS vs BTC"] = (
    trade_df["7D %"]
    if "7D %" in trade_df.columns
    else np.nan
)


# Compute 7D/RS using daily klines for scanned coins.
def get_7d(symbol):
    d = get_klines(
        symbol,
        "1d",
        12,
    )

    if len(d) >= 8:
        return (
            d["close"].iloc[-1]
            / d["close"].iloc[-8]
            - 1
        ) * 100

    return np.nan


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    fs = {
        ex.submit(get_7d, s): s
        for s in trade_df["Symbol"]
    }

    seven = {}

    for fut in as_completed(fs):
        try:
            seven[fs[fut]] = fut.result()
        except Exception:
            seven[fs[fut]] = np.nan

trade_df["7D %"] = trade_df["Symbol"].map(seven)

trade_df["RS vs BTC"] = (
    trade_df["7D %"] - btc_7d
    if np.isfinite(btc_7d)
    else np.nan
)


# Small extra quality adjustment
trade_df["Final Score"] = (
    trade_df["Final Score"]
    .astype(float)
)

trade_df.loc[
    (trade_df["RS vs BTC"] < 0)
    & trade_df["RS vs BTC"].notna(),
    "Final Score"
] -= 8

trade_df.loc[
    trade_df["24H %"] > 12,
    "Final Score"
] -= 6

trade_df["Final Score"] = (
    trade_df["Final Score"]
    .clip(0, 100)
    .round(1)
)


# Regime filter: in risk-off, only very strong setups can qualify.
if regime["Regime"] == "🔴 RISK-OFF":
    trade_df.loc[
        trade_df["Final Score"] < 90,
        "Grade"
    ] = "⚠️ RISK-OFF"


# ============================================================
# TOP SETUPS
# ============================================================

top = trade_df[
    (trade_df["Final Score"] >= min_score)
    & (trade_df["RR"].fillna(0) >= MIN_RR)
    & (
        trade_df["Grade"].isin(
            ["🔥 A+", "🟢 A"]
        )
    )
].sort_values(
    ["Final Score", "RR", "Volume Ratio 1H"],
    ascending=False,
)

if top.empty:
    st.info(
        "এখনই high-quality entry পাওয়া যাচ্ছে না। "
        "WAIT করা better than forcing a trade."
    )

else:
    st.success(
        f"🔥 {len(top)}টি high-quality spot setup পাওয়া গেছে।"
    )

    display_cols = [
        "Symbol", "Grade", "Final Score", "Trigger",
        "Price", "Entry", "Stop", "TP1", "TP2", "RR",
        "24H %", "7D %", "RS vs BTC",
        "15m", "1H", "4H", "1D",
        "RSI 1H", "ADX 1H", "Volume Ratio 1H",
    ]

    st.dataframe(
        top[display_cols].head(15),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SELECTED SETUP DETAIL
# ============================================================

st.divider()
st.header("🔬 Setup Detail")

options = (
    trade_df.sort_values(
        "Final Score",
        ascending=False,
    )["Symbol"]
    .tolist()
)

selected = st.selectbox(
    "Coin select করুন",
    options,
    index=0,
)

row = trade_df[
    trade_df["Symbol"] == selected
].iloc[0]

d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Final Score",
    f"{row['Final Score']:.1f}/100",
)

d2.metric(
    "Grade",
    row["Grade"],
)

d3.metric(
    "Trigger",
    row["Trigger"],
)

d4.metric(
    "Risk / Reward",
    f"{row['RR']:.2f}R"
    if np.isfinite(row["RR"])
    else "N/A",
)


entry_status = str(
    row.get("Entry Status", "")
)

if (
    "STRONG ENTRY" in entry_status
    or "ENTRY CONFIRMED" in entry_status
):
    st.success(f"**{entry_status}**")

elif "WAIT" in entry_status:
    st.warning(f"**{entry_status}**")

else:
    st.error(f"**{entry_status}**")


st.write(
    f"**Entry:** "
    f"{fmt_price(row['Entry']) if np.isfinite(row['Entry']) else 'WAIT'}  |  "
    f"**Stop:** "
    f"{fmt_price(row['Stop']) if np.isfinite(row['Stop']) else 'N/A'}  |  "
    f"**TP1:** "
    f"{fmt_price(row['TP1']) if np.isfinite(row['TP1']) else 'N/A'}  |  "
    f"**TP2:** "
    f"{fmt_price(row['TP2']) if np.isfinite(row['TP2']) else 'N/A'}"
)

st.caption(
    f"Why: {row['Reason']}"
)

st.markdown(
    "### 💵 Position Size by Your Risk"
)

risk_dollar = (
    portfolio_balance * risk_pct / 100
)

if (
    np.isfinite(row["Entry"])
    and np.isfinite(row["Stop"])
    and row["Entry"] > row["Stop"]
):
    per_coin_risk_pct = (
        (row["Entry"] - row["Stop"])
        / row["Entry"]
        * 100
    )

    raw_position_value = (
        risk_dollar
        / (per_coin_risk_pct / 100)
    )

    spot_position_value = min(
        raw_position_value,
        portfolio_balance,
    )

    spot_qty = (
        spot_position_value
        / row["Entry"]
    )

    pc1, pc2, pc3, pc4 = st.columns(4)

    pc1.metric(
        "Portfolio",
        f"${portfolio_balance:,.2f}",
    )

    pc2.metric(
        "Risk",
        f"{risk_pct:.2f}%",
    )

    pc3.metric(
        "Max Loss",
        f"${risk_dollar:,.2f}",
    )

    pc4.metric(
        "Position Size",
        f"${spot_position_value:,.2f}",
    )

    st.caption(
        f"SL distance: {per_coin_risk_pct:.2f}%  •  "
        f"Quantity: {spot_qty:.6f}"
    )

    if raw_position_value > portfolio_balance:
        st.warning(
            f"Exact risk-based size would be ${raw_position_value:,.2f}, "
            f"but your portfolio is ${portfolio_balance:,.2f}. "
            "Spot-only: position is capped at your available portfolio, "
            "so actual SL loss will be below the selected risk."
        )

else:
    st.info(
        "Valid Entry এবং Stop Loss না পাওয়া পর্যন্ত "
        "position size calculate হবে না."
    )


# ============================================================
# FULL SCAN
# ============================================================

st.divider()
st.subheader("📊 Full Technical Scan")

scan_cols = [
    "Symbol", "Coin", "Grade", "Final Score", "MTF Score", "Setup Score",
    "24H %", "7D %", "RS vs BTC", "Market Cap",
    "Trigger", "RR", "RSI 1H", "ADX 1H",
    "Volume Ratio 1H", "ATR % 1H",
]

st.dataframe(
    trade_df.sort_values(
        ["Final Score", "RR"],
        ascending=False,
    )[scan_cols],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# NARRATIVES
# ============================================================

st.divider()
st.header("🔥 Narrative Intelligence")

scanner = df.copy()

scanner["Final Score"] = scanner["Symbol"].map(
    trade_df.set_index("Symbol")["Final Score"].to_dict()
)

scanner["Volume Ratio"] = scanner["Symbol"].map(
    trade_df.set_index("Symbol")["Volume Ratio 1H"].to_dict()
)

scanner["RS vs BTC"] = scanner["Symbol"].map(
    trade_df.set_index("Symbol")["RS vs BTC"].to_dict()
)

narratives = build_narratives(scanner)

if narratives.empty:
    st.info("Narrative data unavailable.")

else:
    st.dataframe(
        narratives.head(15),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SMALL CAPS
# ============================================================

st.divider()
st.header("🚀 Small-Cap Leaders")

small = trade_df[
    trade_df["Market Cap"].between(
        SMALL_CAP_MIN,
        SMALL_CAP_MAX,
    )
].copy()

if small.empty:
    st.info(
        "এই scan-এ qualifying small-cap পাওয়া যায়নি।"
    )

else:
    st.dataframe(
        small.sort_values(
            ["Final Score", "RS vs BTC", "Volume Ratio 1H"],
            ascending=False,
        )[
            [
                "Symbol", "Market Cap", "Final Score", "Grade",
                "24H %", "7D %", "RS vs BTC",
                "Volume Ratio 1H", "Trigger", "RR",
            ]
        ].head(25),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PRE-BREAKOUT
# ============================================================

st.divider()
st.header("🚨 Pre-Breakout Watch")

pre = trade_df[
    trade_df["Trigger"].isin(
        [
            "🟡 BREAKOUT WATCH",
            "🟡 PULLBACK WATCH",
        ]
    )
].sort_values(
    ["Final Score", "RR"],
    ascending=False,
)

if pre.empty:
    st.info(
        "এই মুহূর্তে clean pre-breakout setup নেই।"
    )

else:
    st.dataframe(
        pre[
            [
                "Symbol", "Final Score", "Grade", "Trigger",
                "Price", "Entry", "Stop", "TP1", "TP2",
                "RR", "Resistance", "Support",
                "RSI 1H", "ADX 1H", "Volume Ratio 1H",
            ]
        ].head(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SIGNAL TRACKING
# ============================================================

st.divider()
st.header("📈 Signal Tracking")

now = get_bd_time().strftime(
    "%Y-%m-%d %H:%M:%S"
)

signal_rows = []

for _, r in top.head(10).iterrows():
    signal_rows.append({
        "Detected At": now,
        "Symbol": r["Symbol"],
        "Grade": r["Grade"],
        "Score": r["Final Score"],
        "Trigger": r["Trigger"],
        "Price": r["Price"],
        "Entry": r["Entry"],
        "Stop": r["Stop"],
        "TP1": r["TP1"],
        "TP2": r["TP2"],
        "RR": r["RR"],
        "24H %": r["24H %"],
        "7D %": r["7D %"],
        "RS vs BTC": r["RS vs BTC"],
    })

if signal_rows:
    save_signals(signal_rows)

history = load_history()

if history.empty:
    st.info(
        "এখনো qualifying signal history নেই."
    )

else:
    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Tracked Signals",
        len(history),
    )

    h2.metric(
        "Tracked Coins",
        history["Symbol"].nunique(),
    )

    h3.metric(
        "History File",
        "signal_history_v9.csv",
    )

    st.dataframe(
        history.tail(50).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    csv = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Signal History",
        data=csv,
        file_name="signal_history_v9.csv",
        mime="text/csv",
    )


# ============================================================
# DECISION FRAMEWORK
# ============================================================

st.divider()
st.header("🧠 How V9 decides")

st.markdown("""
**Trade priority:**

1. **BTC 4H regime filter** — risk-on / neutral / risk-off.
2. **1D + 4H alignment** — avoids taking a 15m signal against the larger trend.
3. **1H structure** — EMA20/50/200, MACD, RSI, ADX.
4. **15m trigger** — momentum and volume confirmation.
5. **Resistance logic** — breakout, breakout-watch or pullback-watch.
6. **ATR stop** — stop distance adapts to volatility instead of using an arbitrary percentage.
7. **Minimum 2R** — weak reward/risk setups are rejected.
8. **Relative strength vs BTC** — coins underperforming BTC are penalized.
9. **Overheated-move penalty** — already-pumped coins are treated more carefully.
10. **A+ only when multiple independent confirmations agree.**
""")

st.warning(
    "Important: a high score is a screening signal, not a guarantee of a profitable trade. "
    "Always verify the live chart, liquidity, news/event risk and execution before entering."
)

st.caption(
    f"Risal Trading Dashboard V9.0 • Spot-only • "
    f"Last update {get_bd_time():%Y-%m-%d %H:%M:%S} BD"
)
