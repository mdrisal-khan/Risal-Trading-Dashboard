import streamlit as st
import requests
import pandas as pd
import time
import math
import threading
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# RISAL TRADING DASHBOARD V7.2
# Binance Failover • Signal Tracking • Pre-Breakout Scanner
# =========================================================

st.set_page_config(
    page_title="Risal Trading Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b0e14;
    color: #c5cdd9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
h1 { color: #4da6ff !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }
div[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
div[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.85rem !important; font-weight: 500; }
div[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.5rem !important; font-weight: 700; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
.stButton>button {
    background-color: #238636 !important; color: #ffffff !important;
    border: none !important; border-radius: 6px !important; font-weight: 600 !important;
}
.stButton>button:hover { background-color: #2ea043 !important; box-shadow: 0 0 10px rgba(46, 160, 67, 0.4); }
hr { border-color: #30363d !important; margin: 1.5rem 0 !important; }
div[data-testid="stDataFrame"] {
    border: 1px solid #30363d; border-radius: 8px; overflow: hidden; background-color: #161b22;
}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Risal Trading Dashboard")
st.caption("Narrative Intelligence • Trade Radar • Small-Cap Momentum • Coin Scanner")

# =========================================================
# SETTINGS
# =========================================================

MIN_VOLUME = 2_000_000
MIN_MOMENTUM = 3.0
MIN_VOLUME_RATIO = 1.20

SMALL_CAP_MIN = 10_000_000
SMALL_CAP_MAX = 500_000_000

TOP_NARRATIVES = 15
TOP_COINS = 30
TOP_SMALL_CAP = 25

DEEP_SCAN_LIMIT = 100

MAX_VALID_7D_CHANGE = 75.0
MAX_VALID_RS = 80.0
MIN_CLEAN_COVERAGE = 0.67
MIN_STRONG_NARRATIVE_COINS = 3

REQUEST_TIMEOUT = 15
MAX_WORKERS = 8
API_RETRIES = 2
API_BACKOFF = 0.8

STRONG_LEADER_SCORE = 60
STRONG_LEADER_COIN_SCORE = 55
STRONG_LEADER_VOLUME = 1.50
STRONG_LEADER_MOMENTUM = 5.0

DEVELOPING_LEADER_SCORE = 45
DEVELOPING_LEADER_COIN_SCORE = 40
DEVELOPING_LEADER_VOLUME = 1.20
DEVELOPING_LEADER_MOMENTUM = 3.0

BREAKOUT_SCAN_LIMIT = 60
BREAKOUT_INTERVAL = "1h"
BREAKOUT_KLINE_LIMIT = 80
BREAKOUT_MIN_SCORE = 68
BREAKOUT_DEBUG = False

BREAKOUT_MAX_DISTANCE = 4.0
BREAKOUT_MAX_24H = 8.0
BREAKOUT_MIN_VOLUME_RATIO = 0.60
BREAKOUT_MIN_COMPRESSION = 8.0
BREAKOUT_MIN_HIGHER_LOW = 0.50
BREAKOUT_MIN_RESISTANCE_TESTS = 3

# IMPORTANT: __file__ is correct here (not _file_)
SIGNAL_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "signal_history.csv"
)

BINANCE_BASE_URLS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://data-api.binance.vision",
]

BINANCE_TICKER_PATH = "/api/v3/ticker/24hr"
BINANCE_KLINE_PATH = "/api/v3/klines"

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

NARRATIVE_MAP = {
    "Artificial Intelligence (AI)": ["FET","RENDER","TAO","NEAR","AKT","AR","IO","AIOZ","GRASS","VIRTUAL","WLD"],
    "Real World Assets (RWA)": ["ONDO","PENDLE","MKR","LINK","POLYX","CPOOL","TRU","XDC"],
    "Decentralized Finance (DeFi)": ["UNI","AAVE","MKR","CRV","LDO","COMP","SNX","DYDX","PENDLE","JUP","RAY","SUSHI"],
    "Decentralized Exchange (DEX)": ["UNI","JUP","DYDX","RAY","SUSHI","CAKE","GMX","ORCA"],
    "Layer 1": ["ETH","SOL","BNB","ADA","AVAX","SUI","APT","NEAR","ATOM","TRX","SEI","TON","XRP"],
    "Layer 2 / Rollup": ["ARB","OP","STRK","ZK","MANTA","METIS","IMX","MNT"],
    "DePIN": ["RENDER","FIL","AR","HNT","THETA","AKT","IO","GRASS","AIOZ"],
    "Gaming / GameFi": ["IMX","GALA","SAND","MANA","AXS","BEAM","RON","PIXEL","SUPER"],
    "Meme": ["DOGE","SHIB","PEPE","BONK","FLOKI","WIF","BRETT","MEME"],
    "Privacy": ["XMR","ZEC","DASH","SCRT","ROSE"],
    "Oracle": ["LINK","PYTH","API3","BAND","UMA"],
    "Liquid Staking / Restaking": ["LDO","RPL","ETHFI","EIGEN","REZ"],
    "Infrastructure": ["LINK","FIL","AR","ICP","TIA","ATOM","QNT","GRT"]
}

def is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False

def safe_float(value, default=0.0):
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return default

def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default

def get_bd_time():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Dhaka"))
    except Exception:
        return datetime.now()

_THREAD_LOCAL = threading.local()

def get_worker_session():
    if not hasattr(_THREAD_LOCAL, "session"):
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Risal-Trading-Dashboard/7.2",
            "Accept": "application/json"
        })
        _THREAD_LOCAL.session = session
    return _THREAD_LOCAL.session

@st.cache_resource
def get_http_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Risal-Trading-Dashboard/7.2",
        "Accept": "application/json"
    })
    return session

SESSION = get_http_session()

def request_json(url, params=None, retries=API_RETRIES):
    session = get_worker_session()
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            status = response.status_code
            if 200 <= status < 300:
                try:
                    return response.json()
                except Exception:
                    return None

            retryable = status == 429 or status in {500, 502, 503, 504}
            if not retryable or attempt >= retries:
                return None

            retry_after = response.headers.get("Retry-After")
            try:
                wait_time = float(retry_after)
            except Exception:
                wait_time = API_BACKOFF * (2 ** attempt)
            time.sleep(min(max(wait_time, 0.25), 8.0))

        except (requests.RequestException, ValueError):
            if attempt >= retries:
                return None
            time.sleep(min(API_BACKOFF * (2 ** attempt), 8.0))
        except Exception:
            return None
    return None

def request_binance_json(path, params=None, retries=API_RETRIES):
    session = get_worker_session()
    diagnostics = []

    for base_url in BINANCE_BASE_URLS:
        url = base_url + path
        for attempt in range(retries + 1):
            try:
                response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                status = response.status_code
                diagnostics.append(f"{base_url} → HTTP {status}")

                if 200 <= status < 300:
                    try:
                        return response.json(), diagnostics
                    except Exception:
                        diagnostics.append(f"{base_url} → invalid JSON")
                        break

                if status in {403, 418}:
                    break

                retryable = status == 429 or status in {500, 502, 503, 504}
                if not retryable or attempt >= retries:
                    break

                retry_after = response.headers.get("Retry-After")
                try:
                    wait_time = float(retry_after)
                except Exception:
                    wait_time = API_BACKOFF * (2 ** attempt)
                time.sleep(min(max(wait_time, 0.25), 5.0))

            except requests.RequestException as exc:
                diagnostics.append(f"{base_url} → {type(exc).__name__}")
                if attempt >= retries:
                    break
                time.sleep(min(API_BACKOFF * (2 ** attempt), 5.0))
            except Exception as exc:
                diagnostics.append(f"{base_url} → {type(exc).__name__}")
                break

    return None, diagnostics

@st.cache_data(ttl=30, show_spinner=False)
def get_binance_data():
    data, diagnostics = request_binance_json(BINANCE_TICKER_PATH)
    if not isinstance(data, list):
        return pd.DataFrame(), diagnostics

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue

        symbol = str(item.get("symbol", "")).upper()
        if not symbol.endswith("USDT"):
            continue

        price = safe_float(item.get("lastPrice"), default=float("nan"))
        change = safe_float(item.get("priceChangePercent"), default=float("nan"))
        volume = safe_float(item.get("quoteVolume"), default=float("nan"))
        trades = safe_int(item.get("count"), default=0)

        if not is_finite_number(price) or not is_finite_number(change) or not is_finite_number(volume):
            continue
        if volume < MIN_VOLUME:
            continue

        rows.append({
            "Symbol": symbol,
            "Coin": symbol[:-4].upper(),
            "Price": price,
            "24H %": change,
            "24H Volume": volume,
            "Trades": trades
        })

    if not rows:
        return pd.DataFrame(), diagnostics

    return pd.DataFrame(rows).drop_duplicates(subset=["Symbol"], keep="first"), diagnostics

@st.cache_data(ttl=900, show_spinner=False)
def get_market_caps():
    rows = []
    for page in range(1, 5):
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": "false"
        }
        data = request_json(COINGECKO_MARKETS_URL, params=params)
        if not isinstance(data, list) or not data:
            break

        for item in data:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper().strip()
            market_cap = safe_float(item.get("market_cap"), default=float("nan"))
            if not symbol or not is_finite_number(market_cap) or market_cap <= 0:
                continue
            rows.append({"Coin": symbol, "Market Cap": market_cap})

        if page < 4:
            time.sleep(0.20)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).groupby("Coin", as_index=False)["Market Cap"].max()

def fetch_7d_change(symbol):
    data, _ = request_binance_json(
        BINANCE_KLINE_PATH,
        params={"symbol": symbol, "interval": "1d", "limit": 9}
    )
    if not isinstance(data, list) or len(data) < 8:
        return symbol, float("nan"), False

    closed = data[:-1]
    if len(closed) < 8:
        return symbol, float("nan"), False

    try:
        first_close = float(closed[0][4])
        last_close = float(closed[-1][4])
    except Exception:
        return symbol, float("nan"), False

    if not is_finite_number(first_close) or not is_finite_number(last_close) or first_close <= 0 or last_close <= 0:
        return symbol, float("nan"), False

    change = ((last_close - first_close) / first_close) * 100
    return symbol, change, is_finite_number(change)

def fetch_volume_ratio(symbol):
    data, _ = request_binance_json(
        BINANCE_KLINE_PATH,
        params={"symbol": symbol, "interval": "1h", "limit": 8}
    )
    if not isinstance(data, list) or len(data) < 8:
        return symbol, float("nan"), False

    closed = data[:-1]
    if len(closed) < 7:
        return symbol, float("nan"), False

    try:
        latest_volume = float(closed[-1][5])
        previous_volumes = [float(item[5]) for item in closed[:-1]]
    except Exception:
        return symbol, float("nan"), False

    if not is_finite_number(latest_volume) or latest_volume < 0:
        return symbol, float("nan"), False

    valid_previous = [value for value in previous_volumes if is_finite_number(value) and value >= 0]
    if not valid_previous:
        return symbol, float("nan"), False

    average_volume = sum(valid_previous) / len(valid_previous)
    if not is_finite_number(average_volume) or average_volume <= 0:
        return symbol, float("nan"), False

    ratio = latest_volume / average_volume
    return symbol, ratio, is_finite_number(ratio)

def fetch_symbol_market_data(symbol):
    _, change_7d, seven_day_valid = fetch_7d_change(symbol)
    _, volume_ratio, volume_valid = fetch_volume_ratio(symbol)
    return symbol, {
        "7D %": change_7d,
        "7D Valid": seven_day_valid,
        "Volume Ratio": volume_ratio,
        "Volume Valid": volume_valid
    }

@st.cache_data(ttl=30, show_spinner=False)
def get_deep_market_data(symbols):
    symbols = tuple(sorted(set(symbols)))
    results = {}
    if not symbols:
        return results

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_symbol_market_data, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            original_symbol = futures[future]
            try:
                symbol, data = future.result()
                results[symbol] = data
            except Exception:
                results[original_symbol] = {
                    "7D %": float("nan"),
                    "7D Valid": False,
                    "Volume Ratio": float("nan"),
                    "Volume Valid": False
                }
    return results

def fetch_breakout_klines(symbol):
    data, _ = request_binance_json(
        BINANCE_KLINE_PATH,
        params={"symbol": symbol, "interval": BREAKOUT_INTERVAL, "limit": BREAKOUT_KLINE_LIMIT},
        retries=1
    )
    return data if isinstance(data, list) else None

def analyze_pre_breakout(symbol, market_row):
    data = fetch_breakout_klines(symbol)
    if not data or len(data) < 45:
        return None

    closed = data[:-1]
    if len(closed) < 40:
        return None

    try:
        highs = pd.Series([float(x[2]) for x in closed], dtype="float64")
        lows = pd.Series([float(x[3]) for x in closed], dtype="float64")
        closes = pd.Series([float(x[4]) for x in closed], dtype="float64")
        volumes = pd.Series([float(x[5]) for x in closed], dtype="float64")
    except Exception:
        return None

    if closes.empty or closes.iloc[-1] <= 0 or not closes.notna().all():
        return None

    price = float(closes.iloc[-1])
    resistance_window = highs.iloc[-25:-1]
    if resistance_window.empty:
        return None

    resistance = float(resistance_window.quantile(0.85))
    if resistance <= 0:
        return None

    distance_pct = ((resistance - price) / price) * 100
    if distance_pct < 0 or distance_pct > BREAKOUT_MAX_DISTANCE:
        return None

    recent_lows = lows.iloc[-20:].reset_index(drop=True)
    first_half_low = float(recent_lows.iloc[:10].mean())
    second_half_low = float(recent_lows.iloc[-10:].mean())
    higher_low_pct = ((second_half_low - first_half_low) / max(first_half_low, 1e-12)) * 100

    range_now = float(highs.iloc[-10:].max()) - float(lows.iloc[-10:].min())
    range_old = float(highs.iloc[-30:-20].max()) - float(lows.iloc[-30:-20].min())
    compression_pct = ((1 - (range_now / range_old)) * 100) if range_old > 0 else 0.0

    candle_range = ((highs - lows) / closes.replace(0, pd.NA)).astype("float64")
    old_range_avg = safe_float(candle_range.iloc[-30:-20].mean(), default=0.0)
    new_range_avg = safe_float(candle_range.iloc[-10:].mean(), default=0.0)
    volatility_compression = ((1 - new_range_avg / old_range_avg) * 100) if old_range_avg > 0 else 0.0
    compression_signal = max(compression_pct, volatility_compression)

    avg_volume = safe_float(volumes.iloc[-21:-1].mean(), default=0.0)
    latest_volume = safe_float(volumes.iloc[-1], default=0.0)
    volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 0.0

    ema10 = safe_float(closes.ewm(span=10).mean().iloc[-1], default=price)
    ema20 = safe_float(closes.ewm(span=20).mean().iloc[-1], default=price)

    trend_points = 0
    if price >= ema10:
        trend_points += 1
    if ema10 >= ema20:
        trend_points += 1
    if higher_low_pct > 0.5:
        trend_points += 1

    resistance_tests = int((highs.iloc[-25:-1] >= resistance * 0.995).sum())
    change_24h = safe_float(market_row["24H %"], 0)

    if change_24h > BREAKOUT_MAX_24H:
        return None
    if higher_low_pct < BREAKOUT_MIN_HIGHER_LOW:
        return None
    if compression_signal < BREAKOUT_MIN_COMPRESSION:
        return None
    if volume_ratio < BREAKOUT_MIN_VOLUME_RATIO:
        return None
    if resistance_tests < BREAKOUT_MIN_RESISTANCE_TESTS:
        return None
    if trend_points < 2:
        return None

    score = 0.0
    if distance_pct <= 1.0:
        score += 30
    elif distance_pct <= 2.0:
        score += 25
    else:
        score += 18

    if higher_low_pct >= 3:
        score += 20
    elif higher_low_pct >= 1:
        score += 16
    else:
        score += 10

    if compression_signal >= 30:
        score += 20
    elif compression_signal >= 15:
        score += 16
    else:
        score += 10

    if volume_ratio >= 1.5:
        score += 15
    elif volume_ratio >= 1.0:
        score += 12
    elif volume_ratio >= 0.8:
        score += 9
    else:
        score += 6

    score += trend_points * 5
    score += min(resistance_tests, 10)

    if change_24h > 5:
        score -= 8
    elif change_24h > 3:
        score -= 3

    score = min(max(score, 0), 100)

    if distance_pct <= 1.5 and score >= 78:
        status = "🔥 NEAR BREAKOUT"
    elif distance_pct <= 3.0 and score >= BREAKOUT_MIN_SCORE:
        status = "🟢 COILING"
    else:
        status = "👀 WATCH"

    return {
        "Symbol": symbol,
        "Price": price,
        "Resistance": resistance,
        "Distance to Resistance %": round(distance_pct, 2),
        "Higher Low %": round(higher_low_pct, 2),
        "Compression %": round(compression_signal, 2),
        "Raw Range Compression %": round(compression_pct, 2),
        "Volatility Compression %": round(volatility_compression, 2),
        "Volume Ratio": round(volume_ratio, 2),
        "Trend Points": trend_points,
        "Price vs EMA10 %": round(((price - ema10) / max(ema10, 1e-12)) * 100, 2),
        "Price vs EMA20 %": round(((price - ema20) / max(ema20, 1e-12)) * 100, 2),
        "Resistance Tests": resistance_tests,
        "Trend": "Positive" if trend_points >= 2 else "Mixed",
        "24H %": round(change_24h, 2),
        "Score": round(score, 1),
        "Status": status
    }

@st.cache_data(ttl=60, show_spinner=False)
def get_breakout_candidates(symbols, market_rows):
    rows = []
    market_lookup = {row["Symbol"]: row for row in market_rows}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(analyze_pre_breakout, symbol, market_lookup[symbol]): symbol
            for symbol in symbols if symbol in market_lookup
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    rows.append(result)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["Score", "Distance to Resistance %", "Higher Low %", "Volume Ratio"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)

def load_signal_history():
    if not os.path.exists(SIGNAL_HISTORY_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(SIGNAL_HISTORY_FILE)
        if df.empty:
            return df
        if "Detected At" in df.columns:
            df["Detected At"] = pd.to_datetime(df["Detected At"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

def append_signal_rows(rows):
    if not rows:
        return

    new_df = pd.DataFrame(rows)
    existing = load_signal_history()
    combined = new_df if existing.empty else pd.concat([existing, new_df], ignore_index=True)

    if "Detected At" in combined.columns:
        combined["Detected At"] = combined["Detected At"].astype(str)

    if {"Detected At", "Symbol", "Signal"}.issubset(combined.columns):
        combined = combined.drop_duplicates(
            subset=["Detected At", "Symbol", "Signal"], keep="last"
        )

    combined.to_csv(SIGNAL_HISTORY_FILE, index=False)

def build_signal_rows(scanner_df, small_leaders, a_level, watchlist, breakout_df):
    detected_at = get_bd_time().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    def add_from_df(source_df, signal_name, score_col="Score"):
        if source_df is None or source_df.empty:
            return
        for _, row in source_df.iterrows():
            symbol = str(row.get("Symbol", ""))
            if not symbol:
                continue
            rows.append({
                "Detected At": detected_at,
                "Symbol": symbol,
                "Signal": signal_name,
                "Price": safe_float(row.get("Price"), default=float("nan")),
                "24H %": safe_float(row.get("24H %"), default=float("nan")),
                "7D %": safe_float(row.get("7D %"), default=float("nan")),
                "RS vs BTC": safe_float(row.get("RS vs BTC"), default=float("nan")),
                "Volume Ratio": safe_float(row.get("Volume Ratio"), default=float("nan")),
                "Score": safe_float(row.get(score_col), default=float("nan")),
                "Market Cap": safe_float(row.get("Market Cap"), default=float("nan"))
            })

    add_from_df(a_level, "A-Level Candidate")

    if small_leaders is not None and not small_leaders.empty and "Leader Status" in small_leaders.columns:
        top_strong_leaders = (
            small_leaders[small_leaders["Leader Status"] == "🔥 STRONG LEADER"]
            .sort_values(["Leader Score", "Score", "RS vs BTC", "Volume Ratio"], ascending=False)
            .head(3)
        )
        add_from_df(top_strong_leaders, "Strong Leader", "Leader Score")

    return rows

if "live_enabled" not in st.session_state:
    st.session_state.live_enabled = True
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 30

with st.sidebar:
    st.header("⚡ Live Control")

    live_enabled = st.toggle("🟢 Live Auto Refresh", key="live_enabled")
    refresh_interval = st.selectbox(
        "🔄 Refresh Interval",
        options=[30, 60, 120],
        format_func=lambda x: f"{x} seconds",
        key="refresh_interval"
    )

    st.caption("Binance data auto-refresh হবে selected interval অনুযায়ী।")

    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if live_enabled:
        st.success(f"🟢 LIVE • Every {refresh_interval}s")
    else:
        st.warning("⏸️ PAUSED")

    st.divider()
    st.caption("📁 Signal history: local CSV")
    st.caption("⚠️ Tracking runs while dashboard is running.")

run_every = f"{refresh_interval}s" if live_enabled else None

@st.fragment(run_every=run_every)
def live_dashboard():
    st.caption(
        "🟢 LIVE DATA • Last updated: "
        f"{get_bd_time():%Y-%m-%d %H:%M:%S} (BD Time)"
    )

    df, binance_diagnostics = get_binance_data()

    if df.empty:
        st.error("Binance data পাওয়া যায়নি। Failover endpoints-ও চেষ্টা করা হয়েছে।")
        with st.expander("🔧 Binance Connection Diagnostics"):
            if binance_diagnostics:
                for message in binance_diagnostics:
                    st.write("•", message)
            else:
                st.write("কোনো endpoint response পাওয়া যায়নি।")
            st.caption("Network/VPN/firewall/rate-limit সমস্যা হলে এই status দেখে বোঝা যাবে।")
        return

    cap_df = get_market_caps()
    if not cap_df.empty:
        df = df.merge(cap_df, on="Coin", how="left")
    else:
        df["Market Cap"] = float("nan")

    btc_result = get_deep_market_data(["BTCUSDT"])
    btc_data = btc_result.get("BTCUSDT", {})
    btc_7d_raw = btc_data.get("7D %", float("nan"))
    btc_7d_valid = btc_data.get("7D Valid", False) and is_finite_number(btc_7d_raw)
    btc_7d = float(btc_7d_raw) if btc_7d_valid else float("nan")

    st.divider()
    st.subheader("📊 Market Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("USDT Markets", len(df))
    c2.metric("BTC 7D", f"{btc_7d:.2f}%" if btc_7d_valid else "N/A")
    c3.metric("Momentum Markets", len(df[df["24H %"] >= MIN_MOMENTUM]))
    c4.metric(
        "Small-Cap Markets",
        len(df[(df["Market Cap"] >= SMALL_CAP_MIN) & (df["Market Cap"] <= SMALL_CAP_MAX)])
    )

    narrative_symbols = {coin.upper() + "USDT" for coins in NARRATIVE_MAP.values() for coin in coins}
    top_momentum_symbols = set(df.sort_values("24H %", ascending=False).head(DEEP_SCAN_LIMIT)["Symbol"].tolist())
    top_volume_symbols = set(df.sort_values("24H Volume", ascending=False).head(DEEP_SCAN_LIMIT)["Symbol"].tolist())

    small_candidates = df[
        (df["Market Cap"] >= SMALL_CAP_MIN) &
        (df["Market Cap"] <= SMALL_CAP_MAX) &
        (df["24H %"] >= 1.0)
    ]
    small_cap_symbols = set(small_candidates.sort_values("24H %", ascending=False).head(50)["Symbol"].tolist())

    deep_symbols = narrative_symbols | top_momentum_symbols | top_volume_symbols | small_cap_symbols
    deep_df = df[df["Symbol"].isin(deep_symbols)].copy()
    deep_data = get_deep_market_data(deep_df["Symbol"].tolist())

    st.divider()
    st.subheader("🔎 Coin Scanner")

    scanner_rows = []
    progress = st.progress(0)
    total = len(deep_df)

    for index, (_, row) in enumerate(deep_df.iterrows()):
        symbol = row["Symbol"]
        change_24h = safe_float(row["24H %"], 0.0)
        volume = safe_float(row["24H Volume"], 0.0)
        trades = safe_int(row["Trades"], 0)
        market_cap = row["Market Cap"]
        coin_data = deep_data.get(symbol, {})

        change_7d = safe_float(coin_data.get("7D %", float("nan")), float("nan"))
        volume_ratio = safe_float(coin_data.get("Volume Ratio", float("nan")), float("nan"))

        seven_day_valid = coin_data.get("7D Valid", False) and is_finite_number(change_7d)
        volume_valid = coin_data.get("Volume Valid", False) and is_finite_number(volume_ratio)

        quality_issues = []
        if not pd.notna(market_cap):
            quality_issues.append("Missing Market Cap")
        if not seven_day_valid:
            quality_issues.append("Missing 7D Data")
        if not volume_valid:
            quality_issues.append("Missing Volume Data")
        if seven_day_valid and abs(change_7d) > MAX_VALID_7D_CHANGE:
            quality_issues.append("Extreme 7D Move")

        rs = change_7d - btc_7d if btc_7d_valid and seven_day_valid else float("nan")
        if is_finite_number(rs) and abs(rs) > MAX_VALID_RS:
            quality_issues.append("Extreme RS")
        if not btc_7d_valid:
            quality_issues.append("BTC Baseline Missing")

        data_quality = "⚠️ " + " • ".join(quality_issues) if quality_issues else "OK"

        score_7d = change_7d if seven_day_valid and abs(change_7d) <= MAX_VALID_7D_CHANGE else 0.0
        score_rs = rs if is_finite_number(rs) and abs(rs) <= MAX_VALID_RS else 0.0
        score_volume_ratio = volume_ratio if volume_valid else 0.0

        liquidity_score = min(max(volume, 0) / 50_000_000, 1) * 25
        momentum_score = min(max(change_24h, 0) / 10, 1) * 20
        rs_score = min(max(score_rs, 0) / 10, 1) * 25
        volume_score = min(max(score_volume_ratio - 1, 0) / 2, 1) * 20
        activity_score = min(max(trades, 0) / 100_000, 1) * 10
        score = liquidity_score + momentum_score + rs_score + volume_score + activity_score

        if pd.notna(market_cap):
            if market_cap < 100_000_000:
                cap_class = "🟣 Micro Cap"
            elif market_cap <= 500_000_000:
                cap_class = "🔵 Small Cap"
            elif market_cap <= 2_000_000_000:
                cap_class = "🟢 Mid Cap"
            else:
                cap_class = "⚫ Large Cap"
        else:
            cap_class = "⚪ Unknown"

        clean_data = data_quality == "OK"

        if clean_data and score >= 70 and rs > 0 and volume_ratio >= MIN_VOLUME_RATIO:
            status = "🔥 A-Level Candidate"
        elif clean_data and score >= 60 and rs > 0:
            status = "🟢 Watchlist"
        elif clean_data and volume_ratio >= MIN_VOLUME_RATIO and change_24h >= MIN_MOMENTUM:
            status = "🟡 Volume Alert"
        else:
            status = "⚪ Developing"

        is_small_cap = pd.notna(market_cap) and SMALL_CAP_MIN <= market_cap <= SMALL_CAP_MAX

        scanner_rows.append({
            "Symbol": symbol,
            "Price": row["Price"],
            "Market Cap": market_cap,
            "Cap Class": cap_class,
            "24H %": round(change_24h, 2),
            "7D %": round(change_7d, 2) if is_finite_number(change_7d) else float("nan"),
            "RS vs BTC": round(rs, 2) if is_finite_number(rs) else float("nan"),
            "Volume Ratio": round(volume_ratio, 2) if is_finite_number(volume_ratio) else float("nan"),
            "24H Volume": round(volume, 0),
            "Score": round(score, 1),
            "Status": status,
            "Small Cap": is_small_cap,
            "Data Quality": data_quality
        })
        progress.progress((index + 1) / max(total, 1))

    progress.empty()
    scanner_df = pd.DataFrame(scanner_rows)

    # =====================================================
    # NARRATIVE ENGINE
    # =====================================================

    narrative_rows = []
    for narrative, coins in NARRATIVE_MAP.items():
        available = []
        for coin in coins:
            symbol = coin.upper() + "USDT"
            match = scanner_df[scanner_df["Symbol"] == symbol]
            if not match.empty:
                available.append(match.iloc[0])

        if not available:
            continue

        clean_available = [item for item in available if item["Data Quality"] == "OK"]
        clean_count = len(clean_available)
        available_count = len(available)
        clean_coverage = clean_count / available_count if available_count > 0 else 0.0
        if not clean_available:
            continue

        positive = sum(1 for item in clean_available if is_finite_number(item["24H %"]) and item["24H %"] > 0)
        breadth = (positive / len(clean_available)) * 100

        avg_momentum = sum(item["24H %"] for item in clean_available if is_finite_number(item["24H %"])) / len(clean_available)
        avg_rs = sum(item["RS vs BTC"] for item in clean_available if is_finite_number(item["RS vs BTC"])) / len(clean_available)
        avg_volume = sum(item["Volume Ratio"] for item in clean_available if is_finite_number(item["Volume Ratio"])) / len(clean_available)

        clean_sorted = sorted(clean_available, key=lambda item: safe_float(item["Score"], 0), reverse=True)
        leader = clean_sorted[0]
        confirmations = 0

        if breadth >= 60 and clean_coverage >= MIN_CLEAN_COVERAGE:
            confirmations += 1
        if is_finite_number(avg_rs) and avg_rs > 0:
            confirmations += 1
        if is_finite_number(leader["RS vs BTC"]) and leader["RS vs BTC"] > 0:
            confirmations += 1
        if is_finite_number(avg_volume) and avg_volume >= MIN_VOLUME_RATIO:
            confirmations += 1

        if (
            confirmations == 4 and breadth >= 60 and avg_rs > 0 and avg_momentum >= 0
            and clean_count >= MIN_STRONG_NARRATIVE_COINS and clean_coverage >= MIN_CLEAN_COVERAGE
        ):
            status, confidence = "🔥 STRONG", "HIGH"
        elif confirmations >= 3 and breadth >= 50 and avg_rs > 0 and avg_momentum >= 0 and clean_count >= 2:
            status, confidence = "🟢 DEVELOPING", "MEDIUM"
        elif breadth >= 60 and avg_momentum > 0:
            status, confidence = "🟡 MOMENTUM ONLY", "LOW"
        else:
            status, confidence = "⚪ WEAK", "LOW"

        momentum_component = min(max(avg_momentum, 0), 10) * 2.5
        breadth_component = breadth * 0.25
        rs_component = min(max(avg_rs, 0), 10) * 2
        volume_component = min(max(avg_volume - 1, 0), 2) * 5

        narrative_score = momentum_component + breadth_component + rs_component + volume_component
        radar_score = narrative_score + safe_float(leader["Score"], 0) * 0.25

        narrative_rows.append({
            "Narrative": narrative,
            "Status": status,
            "Confidence": confidence,
            "Coins": available_count,
            "Breadth %": round(breadth, 1),
            "Avg 24H %": round(avg_momentum, 2),
            "Avg RS": round(avg_rs, 2),
            "Avg Volume": round(avg_volume, 2),
            "Leader": leader["Symbol"],
            "Leader Score": round(safe_float(leader["Score"], 0), 1),
            "Confirmations": f"{confirmations}/4",
            "Narrative Score": round(narrative_score, 1),
            "Radar Score": round(radar_score, 1)
        })

    narrative_df = pd.DataFrame(narrative_rows)
    if not narrative_df.empty:
        narrative_df = narrative_df.sort_values("Radar Score", ascending=False).reset_index(drop=True)

    # =====================================================
    # BTC & ETH ANALYSIS ENGINE
    # =====================================================

    def analyze_btc_eth_asset(symbol):
        ticker_df = df[df["Symbol"] == symbol]
        if ticker_df.empty:
            return None

        row = ticker_df.iloc[0]
        price = safe_float(row["Price"])
        change_24h = safe_float(row["24H %"])
        volume = safe_float(row["24H Volume"])

        klines, _ = request_binance_json(
            BINANCE_KLINE_PATH,
            params={"symbol": symbol, "interval": "1d", "limit": 30}
        )
        if not klines or len(klines) < 20:
            return None

        closes = [safe_float(k[4]) for k in klines[:-1]]
        highs = [safe_float(k[2]) for k in klines[:-1]]
        lows = [safe_float(k[3]) for k in klines[:-1]]
        vols = [safe_float(k[5]) for k in klines[:-1]]

        if len(closes) < 20:
            return None

        first_close_7d = closes[-7]
        last_close = closes[-1]
        change_7d = ((last_close - first_close_7d) / first_close_7d) * 100 if first_close_7d > 0 else 0.0

        sma20 = sum(closes[-20:]) / 20
        ema7 = sum(closes[-7:]) / 7

        support = min(lows[-14:])
        resistance = max(highs[-14:])

        if price > ema7 and price > sma20:
            trend = "🟢 Strong Bullish"
            bias = "Bullish / Upward Continuation"
        elif price > sma20:
            trend = "🟢 Bullish"
            bias = "Slightly Bullish / Consolidating High"
        elif price < ema7 and price < sma20:
            trend = "🔴 Bearish"
            bias = "Bearish / Downward Pressure"
        else:
            trend = "🟡 Neutral / Sideways"
            bias = "Range Bound"

        if change_24h > 2.0 and change_7d > 3.0:
            momentum = "🔥 Strong High Momentum"
        elif change_24h > 0:
            momentum = "🟢 Positive / Moderate"
        elif change_24h < -2.0:
            momentum = "🔴 Strong Selling Pressure"
        else:
            momentum = "🟡 Weak / Neutral"

        avg_vol_7d = sum(vols[-7:]) / 7
        if volume > avg_vol_7d * 1.2:
            vol_cond = "⚡ High Volume (Above Average)"
        elif volume < avg_vol_7d * 0.8:
            vol_cond = "💤 Low Volume (Consolidation)"
        else:
            vol_cond = "📊 Normal Volume"

        summary = (
            f"{symbol[:-4]} is currently trading at ${price:,.2f}. "
            f"The asset exhibits a {trend.lower()} structure with market bias towards '{bias}'. "
            f"Key support rests around ${support:,.2f} while resistance stands near ${resistance:,.2f}."
        )

        return {
            "Price": price, "24H %": change_24h, "7D %": change_7d,
            "Trend": trend, "Bias": bias, "Momentum": momentum,
            "Volume Condition": vol_cond, "Support": support,
            "Resistance": resistance, "Summary": summary
        }

    btc_analysis = analyze_btc_eth_asset("BTCUSDT")
    eth_analysis = analyze_btc_eth_asset("ETHUSDT")

    st.divider()
    st.header("⚡ Major Assets Analysis (BTC & ETH)")
    col_btc, col_eth = st.columns(2)

    with col_btc:
        st.subheader("₿ BTC ANALYSIS")
        st.markdown("---")
        if btc_analysis:
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${btc_analysis['Price']:,.2f}")
            m2.metric("24H %", f"{btc_analysis['24H %']:.2f}%")
            m3.metric("7D %", f"{btc_analysis['7D %']:.2f}%")
            st.write(f"**4. Trend:** {btc_analysis['Trend']}")
            st.write(f"**5. Market Direction / Bias:** {btc_analysis['Bias']}")
            st.write(f"**6. Momentum:** {btc_analysis['Momentum']}")
            st.write(f"**7. Volume Condition:** {btc_analysis['Volume Condition']}")
            st.write(f"**8. Main Support:** ${btc_analysis['Support']:,.2f}")
            st.write(f"**9. Main Resistance:** ${btc_analysis['Resistance']:,.2f}")
            st.info(f"**10. Overall BTC Analysis:**\n{btc_analysis['Summary']}")
        else:
            st.warning("BTC data unavailable for detailed analysis.")

    with col_eth:
        st.subheader("Ξ ETH ANALYSIS")
        st.markdown("---")
        if eth_analysis:
            e1, e2, e3 = st.columns(3)
            e1.metric("Current Price", f"${eth_analysis['Price']:,.2f}")
            e2.metric("24H %", f"{eth_analysis['24H %']:.2f}%")
            e3.metric("7D %", f"{eth_analysis['7D %']:.2f}%")
            st.write(f"**4. Trend:** {eth_analysis['Trend']}")
            st.write(f"**5. Market Direction / Bias:** {eth_analysis['Bias']}")
            st.write(f"**6. Momentum:** {eth_analysis['Momentum']}")
            st.write(f"**7. Volume Condition:** {eth_analysis['Volume Condition']}")
            st.write(f"**8. Main Support:** ${eth_analysis['Support']:,.2f}")
            st.write(f"**9. Main Resistance:** ${eth_analysis['Resistance']:,.2f}")
            st.info(f"**10. Overall ETH Analysis:**\n{eth_analysis['Summary']}")
        else:
            st.warning("ETH data unavailable for detailed analysis.")

    # =====================================================
    # TRADE RADAR
    # =====================================================

    st.divider()
    st.header("🎯 TRADE RADAR")
    st.caption("সবচেয়ে শক্তিশালী narrative ও leader এখানে দেখাবে।")

    if not narrative_df.empty:
        radar_df = narrative_df[narrative_df["Status"].isin(["🔥 STRONG", "🟢 DEVELOPING"])].copy()
        if not radar_df.empty:
            st.dataframe(
                radar_df[[
                    "Narrative","Status","Confidence","Leader","Leader Score",
                    "Breadth %","Avg 24H %","Avg RS","Avg Volume","Confirmations","Radar Score"
                ]].head(10),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("এই মুহূর্তে STRONG/DEVELOPING Radar setup পাওয়া যায়নি।")
            st.dataframe(narrative_df.head(5), use_container_width=True, hide_index=True)
    else:
        st.warning("Narrative data এখনো তৈরি হয়নি।")

    st.divider()
    st.subheader("🔥 Top Trading Narratives")
    if not narrative_df.empty:
        st.dataframe(narrative_df.head(TOP_NARRATIVES), use_container_width=True, hide_index=True)
    else:
        st.info("কোনো qualifying narrative পাওয়া যায়নি।")

    st.divider()
    st.subheader("🏆 Confirmed Narratives")
    if not narrative_df.empty:
        confirmed = narrative_df[narrative_df["Status"] == "🔥 STRONG"].copy()
        if not confirmed.empty:
            st.dataframe(
                confirmed[[
                    "Narrative","Status","Confidence","Leader","Leader Score",
                    "Breadth %","Avg 24H %","Avg RS","Avg Volume","Confirmations","Narrative Score"
                ]],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("এই মুহূর্তে fully confirmed narrative পাওয়া যায়নি।")

    st.divider()
    st.subheader("🏆 Narrative Leaders")
    if not narrative_df.empty:
        leaders = narrative_df[[
            "Narrative","Status","Leader","Leader Score","Breadth %",
            "Avg 24H %","Avg RS","Avg Volume","Confirmations","Radar Score"
        ]]
        st.dataframe(leaders.head(TOP_NARRATIVES), use_container_width=True, hide_index=True)

    # =====================================================
    # SMALL-CAP MOMENTUM
    # =====================================================

    st.divider()
    st.header("🚀 Small-Cap Momentum")
    st.caption("শুধু $10M–$500M market-cap coin এখানে আসবে।")

    small_caps = scanner_df[
        (scanner_df["Market Cap"] >= SMALL_CAP_MIN) &
        (scanner_df["Market Cap"] <= SMALL_CAP_MAX) &
        (scanner_df["24H %"] >= MIN_MOMENTUM) &
        (scanner_df["Data Quality"] == "OK")
    ].copy()

    strong_small = small_caps[
        (small_caps["RS vs BTC"] > 0) &
        (small_caps["Volume Ratio"] >= MIN_VOLUME_RATIO) &
        (small_caps["Score"] >= 55)
    ].copy().sort_values(["Score","RS vs BTC","Volume Ratio"], ascending=False)

    st.subheader("🔥 Strong Small-Cap")
    if not strong_small.empty:
        st.dataframe(
            strong_small[[
                "Symbol","Market Cap","24H %","7D %","RS vs BTC",
                "Volume Ratio","24H Volume","Score","Status"
            ]].head(TOP_SMALL_CAP),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("এই মুহূর্তে Strong Small-Cap পাওয়া যায়নি।")

    small_watch = small_caps[~small_caps["Symbol"].isin(strong_small["Symbol"])].copy()
    small_watch = small_watch.sort_values(["Score","24H %","Volume Ratio"], ascending=False)

    st.subheader("👀 Small-Cap Watch")
    if not small_watch.empty:
        st.dataframe(
            small_watch[[
                "Symbol","Market Cap","24H %","7D %","RS vs BTC",
                "Volume Ratio","24H Volume","Score","Status"
            ]].head(TOP_SMALL_CAP),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("এই মুহূর্তে Small-Cap Watch-এ কোনো coin নেই।")

    # =====================================================
    # SMALL-CAP LEADERS
    # =====================================================

    st.divider()
    st.header("🏆 Small-Cap Leaders")
    st.caption("ছোট market-cap coin-এর মধ্যে momentum, relative strength, volume এবং score অনুযায়ী সেরা leaders।")

    small_leaders = scanner_df[
        (scanner_df["Market Cap"] >= SMALL_CAP_MIN) &
        (scanner_df["Market Cap"] <= SMALL_CAP_MAX) &
        (scanner_df["24H %"] > 0) &
        (scanner_df["RS vs BTC"] > 0) &
        (scanner_df["Data Quality"] == "OK")
    ].copy()

    if not small_leaders.empty:
        small_leaders["Leader Score"] = (
            small_leaders["Score"] * 0.50
            + small_leaders["RS vs BTC"].clip(lower=0, upper=10) * 2.0
            + small_leaders["24H %"].clip(lower=0, upper=10) * 1.5
            + small_leaders["Volume Ratio"].clip(lower=0, upper=3) * 5.0
        ).clip(lower=0, upper=100).round(1)

        def get_small_cap_status(row):
            if (
                row["Leader Score"] >= STRONG_LEADER_SCORE
                and row["Score"] >= STRONG_LEADER_COIN_SCORE
                and row["RS vs BTC"] > 0
                and row["Volume Ratio"] >= STRONG_LEADER_VOLUME
                and row["24H %"] >= STRONG_LEADER_MOMENTUM
            ):
                return "🔥 STRONG LEADER"
            elif (
                row["Leader Score"] >= DEVELOPING_LEADER_SCORE
                and row["Score"] >= DEVELOPING_LEADER_COIN_SCORE
                and row["RS vs BTC"] > 0
                and row["Volume Ratio"] >= DEVELOPING_LEADER_VOLUME
                and row["24H %"] >= DEVELOPING_LEADER_MOMENTUM
            ):
                return "🟢 DEVELOPING LEADER"
            return "🟡 MOMENTUM"

        small_leaders["Leader Status"] = small_leaders.apply(get_small_cap_status, axis=1)
        small_leaders = small_leaders.sort_values(
            ["Leader Score","Score","RS vs BTC","Volume Ratio"],
            ascending=False
        )

        st.dataframe(
            small_leaders[[
                "Symbol","Market Cap","24H %","7D %","RS vs BTC",
                "Volume Ratio","24H Volume","Score","Leader Score","Leader Status"
            ]].head(TOP_SMALL_CAP),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("এই মুহূর্তে positive-RS Small-Cap Leader পাওয়া যায়নি.")

    # =====================================================
    # A-LEVEL COINS
    # =====================================================

    st.divider()
    st.subheader("⭐ A-Level Coin Setups")

    a_level = scanner_df[
        (scanner_df["Status"] == "🔥 A-Level Candidate") &
        (scanner_df["Data Quality"] == "OK") &
        (scanner_df["Market Cap"].notna())
    ].sort_values("Score", ascending=False)

    if not a_level.empty:
        st.dataframe(
            a_level[[
                "Symbol","Market Cap","24H %","7D %","RS vs BTC",
                "Volume Ratio","24H Volume","Score","Cap Class"
            ]].head(TOP_COINS),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("এই মুহূর্তে A-Level setup নেই।")

    # =====================================================
    # WATCHLIST
    # =====================================================

    st.divider()
    st.subheader("🟢 Watchlist")

    watchlist = scanner_df[
        (scanner_df["Status"] == "🟢 Watchlist") &
        (scanner_df["Data Quality"] == "OK")
    ].sort_values("Score", ascending=False)

    if not watchlist.empty:
        st.dataframe(
            watchlist[[
                "Symbol","Market Cap","24H %","7D %","RS vs BTC",
                "Volume Ratio","Score","Cap Class"
            ]].head(TOP_COINS),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Watchlist empty.")

    # =====================================================
    # PRE-BREAKOUT
    # =====================================================

    st.divider()
    st.header("🚨 Pre-Breakout Candidates")
    st.caption(
        "Resistance-এর কাছে price compression, higher-low এবং volume/trend context দেখে সম্ভাব্য early setup খোঁজা হয়। "
        "এটি guaranteed breakout signal নয়।"
    )

    breakout_pool = (
        df[(df["24H Volume"] >= MIN_VOLUME) & (df["24H %"] >= -3.0)]
        .sort_values(["24H Volume","24H %"], ascending=False)
        .head(BREAKOUT_SCAN_LIMIT)
    )

    breakout_df = get_breakout_candidates(
        breakout_pool["Symbol"].tolist(),
        breakout_pool.to_dict("records")
    )

    if not breakout_df.empty:
        st.dataframe(
            breakout_df[[
                "Symbol","Price","Resistance","Distance to Resistance %",
                "Higher Low %","Compression %","Volume Ratio","Trend",
                "24H %","Score","Status"
            ]].head(20),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("এই মুহূর্তে পরিষ্কার pre-breakout/coiling setup পাওয়া যায়নি।")

    # =====================================================
    # SIGNAL TRACKING
    # =====================================================

    st.divider()
    st.header("📈 Signal Tracking")
    st.caption("শুধু A-Level এবং সবচেয়ে শক্তিশালী Top 3 Small-Cap Leader snapshots local CSV-তে সংরক্ষণ হবে।")

    signal_rows = build_signal_rows(scanner_df, small_leaders, a_level, watchlist, breakout_df)
    if signal_rows:
        append_signal_rows(signal_rows)

    history_df = load_signal_history()

    if history_df.empty:
        st.info("এখনো signal history তৈরি হয়নি। Dashboard কিছুক্ষণ চালু রাখলে snapshots জমবে।")
    else:
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Tracked Snapshots", len(history_df))
        hc2.metric("Tracked Coins", history_df["Symbol"].nunique() if "Symbol" in history_df.columns else 0)
        hc3.metric("History File", "signal_history.csv")

        display_history = history_df.copy()
        display_history.insert(0, "Delete", False)
        display_history.insert(1, "_History ID", range(len(display_history)))
        display_history = display_history.tail(50).iloc[::-1].copy()

        edited_history = st.data_editor(
            display_history,
            use_container_width=True,
            hide_index=True,
            disabled=[col for col in display_history.columns if col not in ["Delete"]],
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    "🗑️ Delete",
                    help="Tick করে যেসব history row delete করতে চান সেগুলো select করুন।",
                    default=False
                ),
                "_History ID": None
            },
            key="signal_history_editor"
        )

        delete_selected = st.button("🗑️ Delete Selected History", use_container_width=False, key="delete_selected_history")

        if delete_selected:
            selected_ids = edited_history.loc[
                edited_history["Delete"] == True, "_History ID"
            ].tolist()

            if not selected_ids:
                st.warning("আগে যেসব history delete করতে চান সেগুলোতে 🗑️ checkbox দিন।")
            else:
                history_df = history_df.drop(index=selected_ids).reset_index(drop=True)
                history_df.to_csv(SIGNAL_HISTORY_FILE, index=False)
                st.success(f"✅ {len(selected_ids)}টি selected history delete হয়েছে।")
                st.rerun()

        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Signal History CSV",
            data=csv_bytes,
            file_name="signal_history.csv",
            mime="text/csv",
            use_container_width=False
        )

    # =====================================================
    # RADAR LOGIC
    # =====================================================

    st.divider()
    st.subheader("🧠 Radar Logic")
    for item in [
        "Narrative breadth", "24H momentum", "7D Relative Strength vs BTC",
        "Volume expansion", "Leader strength", "4-point narrative confirmation",
        "Market-cap classification", "Small-cap momentum detection",
        "Data-quality protection", "Extreme-move protection",
        "Binance multi-endpoint failover", "Pre-breakout resistance proximity",
        "Higher-low / compression detection", "Local signal tracking"
    ]:
        st.write("✅ " + item)

    st.caption("Research and screening tool only — not a guaranteed buy/sell signal.")
    st.caption("Risal Trading Dashboard V7.2 • Failover + Signal Tracking + V7.2 Pre-Breakout")

live_dashboard()
