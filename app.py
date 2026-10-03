import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import quote
import time

# ============================================================
# BINANCE SPOT ACCUMULATION + 1H VOLUME + CATALYST SCANNER
# Single-file Streamlit app
# ============================================================

st.set_page_config(
    page_title="Binance Spot Accumulation Scanner",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Changed to Binance official public market data mirror to fix HTTP 451 Error
BINANCE_BASE = "https://data.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Binance-Spot-Scanner/1.0"})

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1 {margin-bottom: 0.2rem;}
.small-note {font-size: 0.85rem; opacity: 0.75;}
.metric-card {
    padding: 0.65rem 0.8rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
@st.cache_data(ttl=300, show_spinner=False)
def get_exchange_info():
    r = SESSION.get(f"{BINANCE_BASE}/api/v3/exchangeInfo", timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def get_24h_tickers():
    r = SESSION.get(f"{BINANCE_BASE}/api/v3/ticker/24hr", timeout=20)
    r.raise_for_status()
    return r.json()

def get_symbols():
    info = get_exchange_info()
    symbols = []
    for s in info.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", True)
        ):
            symbols.append(s["symbol"])
    return symbols

def get_klines(symbol, interval, limit):
    r = SESSION.get(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=12,
    )
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df

    numeric = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Ignore the currently forming candle.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] < now].copy()
    return df.reset_index(drop=True)

def accumulation_score(df):
    """
    Technical-only 4H accumulation model.
    No EMA, RSI, MACD or entry logic.

    Measures:
      - Tight recent range
      - Range contraction
      - Repeated lower-zone support reactions
      - Avoids a one-sided breakdown
      - Current close remains inside the accumulation area
    """
    if len(df) < 20:
        return 0, "Not enough data", {}

    recent = df.tail(18).copy()
    prev = df.tail(36).head(18).copy()

    hi = recent["high"].max()
    lo = recent["low"].min()
    mid = recent["close"].median()

    if mid <= 0:
        return 0, "Weak", {}

    width_pct = (hi - lo) / mid * 100.0

    # Compare recent range width to the preceding 18-candle range.
    prev_hi = prev["high"].max()
    prev_lo = prev["low"].min()
    prev_mid = max(prev["close"].median(), 1e-12)
    prev_width_pct = (prev_hi - prev_lo) / prev_mid * 100.0

    contraction = 1.0 - (width_pct / max(prev_width_pct, 1e-9))

    # Support zone = lower 30% of the recent range.
    rng = max(hi - lo, 1e-12)
    support_ceiling = lo + rng * 0.30
    support_hits = int((recent["low"] <= support_ceiling).sum())

    # How often candles finish in/near the upper half without breaking down.
    close_pos = (recent["close"] - lo) / rng
    upper_close_ratio = float((close_pos >= 0.45).mean())

    # Penalize a recent breakdown from the accumulation low.
    last_close = float(recent["close"].iloc[-1])
    breakdown = last_close < lo * 0.985

    score = 0

    # Tightness: <= 8% gets strongest contribution, <= 20% still acceptable.
    if width_pct <= 8:
        score += 35
    elif width_pct <= 12:
        score += 30
    elif width_pct <= 16:
        score += 24
    elif width_pct <= 20:
        score += 16
    elif width_pct <= 28:
        score += 8

    # Contraction
    if contraction >= 0.35:
        score += 25
    elif contraction >= 0.20:
        score += 20
    elif contraction >= 0.08:
        score += 12
    elif contraction >= 0:
        score += 6

    # Repeated support reactions
    if support_hits >= 5:
        score += 25
    elif support_hits >= 4:
        score += 20
    elif support_hits >= 3:
        score += 14
    elif support_hits >= 2:
        score += 7

    # Healthy closes / no persistent drift at bottom
    if upper_close_ratio >= 0.65:
        score += 15
    elif upper_close_ratio >= 0.50:
        score += 10
    elif upper_close_ratio >= 0.35:
        score += 5

    if breakdown:
        score -= 25

    score = int(np.clip(score, 0, 100))

    if score >= 80:
        label = "Very Strong"
    elif score >= 65:
        label = "Strong"
    elif score >= 50:
        label = "Moderate"
    else:
        label = "Weak"

    return score, label, {
        "range_pct": width_pct,
        "contraction": contraction,
        "support_hits": support_hits,
        "upper_close_ratio": upper_close_ratio,
    }

def volume_multiple(df):
    """
    1H volume explosion:
    latest CLOSED 1H quote volume / median of previous 20 CLOSED 1H candles.
    """
    if len(df) < 22:
        return np.nan, 0.0

    latest = float(df["quote_volume"].iloc[-1])
    baseline = float(df["quote_volume"].iloc[-21:-1].median())

    if baseline <= 0:
        return np.nan, latest

    return latest / baseline, latest

def volume_status(mult):
    if pd.isna(mult):
        return "—"
    if mult >= 5:
        return "🚀"
    if mult >= 4:
        return "🔥"
    if mult >= 3:
        return "🟢"
    if mult >= 2:
        return "🟡"
    return "—"

def fmt_usdt(x):
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:.0f}"

def scan_one(symbol):
    try:
        d4 = get_klines(symbol, "4h", 40)
        d1 = get_klines(symbol, "1h", 30)

        if len(d4) < 20 or len(d1) < 22:
            return None

        score, label, meta = accumulation_score(d4)
        mult, vol = volume_multiple(d1)

        # Scanner keeps coins with 2x+ 1H volume.
        if pd.isna(mult) or mult < MIN_VOLUME_MULTIPLE:
            return None

        return {
            "Coin": symbol,
            "4H Score": score,
            "4H Accumulation": label,
            "4H Range": meta.get("range_pct", np.nan),
            "Support Hits": meta.get("support_hits", 0),
            "1H Volume": fmt_usdt(vol),
            "Volume Multiple": mult,
            "Status": volume_status(mult),
        }
    except Exception:
        return None

# ---------- Catalyst ----------
CATALYST_TERMS = [
    "partnership",
    "strategic agreement",
    "institutional",
    "financial institution",
    "bank",
    "integration",
    "mainnet",
    "launch",
    "funding",
    "investment",
    "adoption",
]

def classify_catalyst(title):
    t = title.lower()
    if any(x in t for x in ["bank", "financial institution"]):
        return "🏦 Financial"
    if "institutional" in t:
        return "🏛️ Institutional"
    if "partnership" in t or "strategic agreement" in t:
        return "🤝 Partnership"
    if "integration" in t:
        return "🌐 Integration"
    if "mainnet" in t or "launch" in t:
        return "🚀 Launch"
    if "funding" in t or "investment" in t:
        return "💰 Funding"
    if "adoption" in t:
        return "📈 Adoption"
    return "📰 News"

def news_strength(title, published_time=None):
    t = title.lower()
    strong_terms = [
        "institutional", "bank", "financial institution",
        "major partnership", "strategic partnership",
        "mainnet", "funding", "investment"
    ]
    moderate_terms = ["integration", "launch", "adoption", "partnership", "agreement"]

    if any(x in t for x in strong_terms):
        return "🔥 Strong"
    if any(x in t for x in moderate_terms):
        return "🟢 Moderate"
    return "🟡 Normal"

@st.cache_data(ttl=600, show_spinner=False)
def get_google_news(symbol):
    base = symbol.replace("USDT", "")
    query = quote(
        f'"{base}" (partnership OR institutional OR bank OR integration OR '
        f'mainnet OR funding OR investment OR adoption)'
    )
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        response = SESSION.get(url, timeout=12)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = []

        for item in root.findall(".//item")[:8]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published = (item.findtext("pubDate") or "").strip()

            if not title or not link:
                continue

            # Keep only catalyst-type headlines.
            if not any(term in title.lower() for term in CATALYST_TERMS):
                continue

            items.append({
                "Coin": symbol,
                "Catalyst": title,
                "Type": classify_catalyst(title),
                "News Strength": news_strength(title),
                "Published": published,
                "Source": link,
            })

        return items

    except (requests.RequestException, ET.ParseError):
        return []
    except Exception:
        return []


def run_catalyst_scan(coins):
    all_items = []
    # Keep this separate from technical scoring.
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(get_google_news, c): c for c in coins}
        for fut in as_completed(futures):
            try:
                all_items.extend(fut.result())
            except Exception:
                pass
    return all_items

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Scanner Settings")

MIN_VOLUME_MULTIPLE = st.sidebar.select_slider(
    "Minimum 1H Volume Multiple",
    options=[2.0, 3.0, 4.0, 5.0],
    value=2.0,
    help="Only coins whose latest closed 1H quote volume is at least this multiple of the previous 20-candle median will appear.",
)

MIN_ACC_SCORE = st.sidebar.slider(
    "Minimum 4H Accumulation Score",
    min_value=0,
    max_value=90,
    value=45,
    step=5,
)

MAX_COINS = st.sidebar.slider(
    "Maximum Technical Rows",
    min_value=20,
    max_value=300,
    value=100,
    step=10,
)

WORKERS = st.sidebar.slider(
    "Scan Parallel Requests",
    min_value=3,
    max_value=12,
    value=8,
)

SHOW_WEAK = st.sidebar.checkbox(
    "Show weaker accumulation if volume qualifies",
    value=True,
)

st.sidebar.caption(
    "Technical scanner uses only 4H price/volume structure and 1H volume. "
    "No EMA, RSI, MACD or trade-entry logic."
)

# ---------- Main ----------
st.title("🔥 Binance Spot Accumulation + Volume Scanner")
st.caption(
    "USDT Spot only • 4H accumulation • 1H volume explosion • separate catalyst watchlist"
)

c1, c2, c3 = st.columns(3)
c1.metric("Universe", "Binance Spot USDT")
c2.metric("4H", "Accumulation")
c3.metric("1H", f"Volume ≥ {MIN_VOLUME_MULTIPLE:.0f}×")

st.markdown(
    '<div class="small-note">Volume multiple = latest CLOSED 1H quote volume ÷ '
    'median quote volume of the previous 20 CLOSED 1H candles.</div>',
    unsafe_allow_html=True
)

run = st.button("🔄 Scan Binance Now", type="primary", use_container_width=True)

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if run:
    start_time = time.time()

    try:
        symbols = get_symbols()
        tickers = get_24h_tickers()

        ticker_map = {
            x["symbol"]: float(x.get("quoteVolume", 0))
            for x in tickers
            if x.get("symbol") in symbols
        }

        # Scan higher-liquidity symbols first only to reduce waiting time.
        # 24h volume is NOT used as a technical signal/filter.
        symbols = sorted(
            symbols,
            key=lambda s: ticker_map.get(s, 0),
            reverse=True
        )

        st.info(f"Scanning {len(symbols)} Binance USDT spot pairs…")

        results = []
        progress = st.progress(0.0)

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(scan_one, s): s for s in symbols}
            total = len(futures)

            for done, future in enumerate(as_completed(futures), start=1):
                progress.progress(done / max(total, 1))
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass

        progress.empty()

        df_scan = pd.DataFrame(results)

        if df_scan.empty:
            st.session_state.last_scan = pd.DataFrame()
        else:
            df_scan = df_scan[df_scan["4H Score"] >= MIN_ACC_SCORE].copy()

            if not SHOW_WEAK:
                df_scan = df_scan[
                    df_scan["4H Accumulation"].isin(["Strong", "Very Strong"])
                ].copy()

            df_scan = df_scan.sort_values(
                ["Volume Multiple", "4H Score"],
                ascending=[False, False]
            ).head(MAX_COINS)

            st.session_state.last_scan = df_scan

        elapsed = time.time() - start_time
        st.success(f"Scan complete in {elapsed:.1f}s")

    except requests.RequestException as e:
        st.error(f"Binance API connection error: {e}")
    except Exception as e:
        st.error(f"Scanner error: {e}")

# ---------- Technical table ----------
df = st.session_state.last_scan

st.divider()
st.subheader("🔥 Technical Scanner — 4H Accumulation + 1H Volume")

if df is None:
    st.info("Press “Scan Binance Now” to start.")
elif df.empty:
    st.warning("No coins matched the current settings.")
else:
    display = df.copy()
    display["Volume Multiple"] = display["Volume Multiple"].map(lambda x: f"{x:.2f}×")
    display["4H Range"] = display["4H Range"].map(lambda x: f"{x:.1f}%")

    st.dataframe(
        display[
            [
                "Coin",
                "4H Accumulation",
                "4H Score",
                "4H Range",
                "Support Hits",
                "1H Volume",
                "Volume Multiple",
                "Status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "🟡 2×+  |  🟢 3×+  |  🔥 4×+  |  🚀 5×+. "
        "These are scanner classifications, not buy/sell signals."
    )

# ---------- Catalyst watchlist ----------
st.divider()
st.subheader("📰 Catalyst Watchlist")
st.caption(
    "News is separate from the technical scanner and is shown only as context. "
    "Headlines come from Google News RSS and should be manually verified at the original source."
)

if df is None or df.empty:
    st.info("Run the technical scanner first. Catalyst search will use the technical shortlist.")
else:
    catalyst_limit = st.slider(
        "Coins to check for catalysts",
        min_value=5,
        max_value=min(50, len(df)),
        value=min(20, len(df)),
        step=5,
    )

    if st.button("📰 Scan Recent Catalysts", use_container_width=True):
        with st.spinner("Checking recent catalyst headlines…"):
            catalyst_rows = run_catalyst_scan(df["Coin"].head(catalyst_limit).tolist())

        if not catalyst_rows:
            st.warning("No matching catalyst headlines were found in the current RSS search.")
        else:
            cdf = pd.DataFrame(catalyst_rows)
            cdf = cdf.drop_duplicates(subset=["Coin", "Catalyst"])
            cdf = cdf.head(100)

            for _, row in cdf.iterrows():
                st.markdown(
                    f"**{row['Coin']}** — {row['Type']}  \n"
                    f"{row['Catalyst']}  \n"
                    f"**News Strength:** {row['News Strength']}  •  "
                    f"**Published:** {row['Published']}  \n"
                    f"[Open source article]({row['Source']})"
                )
                st.divider()

# ---------- Method ----------
with st.expander("ℹ️ Scanner Method"):
    st.markdown("""
**4H Accumulation score**
- Recent 18 closed 4H candles are evaluated.
- Measures range tightness, contraction versus the preceding period,
  repeated lower-range support reactions, and whether closes remain healthy.
- The score is a screening heuristic, not a claim that accumulation is occurring.

**1H Volume Multiple**
- Latest closed 1H quote volume.
- Divided by the median quote volume of the previous 20 closed 1H candles.
- Thresholds: 2×, 3×, 4× and 5×+.

**Excluded from the technical model**
- EMA
- RSI
- MACD
- Buy/sell signals
- Entry/exit rules
- Stop-loss logic

**Catalyst**
- Separate RSS-based news context.
- It does not change the technical score.
- Verify important headlines with the original publisher/project source before acting on them.
""")

st.caption(
    f"Last UI refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
