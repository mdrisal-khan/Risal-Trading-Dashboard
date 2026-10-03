"""
Binance Spot Accumulation + Volume Expansion + Catalyst Radar
Designed for Streamlit Community Cloud.

IMPORTANT:
- Screening/radar only. It does NOT guarantee a trade.
- Uses Binance public market data (no API key required for market scan).
- Catalyst section uses public RSS feeds; for production use, add more reliable/news APIs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="Binance Spot Accumulation Radar",
    page_icon="📡",
    layout="wide",
)

BINANCE_BASE = "https://api.binance.com"
TIMEOUT = 10

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=30, show_spinner=False)
def get_exchange_info():
    r = requests.get(f"{BINANCE_BASE}/api/v3/exchangeInfo", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30, show_spinner=False)
def get_24h_tickers():
    r = requests.get(f"{BINANCE_BASE}/api/v3/ticker/24hr", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def spot_usdt_symbols():
    data = get_exchange_info()
    bad = {
        "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT",
        "BUSDUSDT", "DAIUSDT", "EURUSDT", "TRYUSDT"
    }
    out = []
    for s in data["symbols"]:
        if (
            s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed")
            and s.get("quoteAsset") == "USDT"
            and s.get("symbol") not in bad
        ):
            out.append(s["symbol"])
    return out

def get_klines(symbol, interval, limit=80):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    data = r.json()
    if not isinstance(data, list) or len(data) < 20:
        return None
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_base",
        "taker_quote","ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pct(a, b):
    if b == 0 or pd.isna(b):
        return 0.0
    return (a / b - 1.0) * 100.0

def accumulation_score(df4):
    """
    Heuristic accumulation model:
    1) Price is relatively range-bound over recent 4H candles.
    2) Downside volatility is not dominant.
    3) Recent closes hold near the upper half of the range.
    4) Volume is stable/quiet rather than already in blow-off mode.
    5) Optional positive OBV-style pressure.

    Score 0-100. This is deliberately conservative and transparent.
    """
    if df4 is None or len(df4) < 50:
        return None

    d = df4.copy()
    recent = d.tail(24)       # ~4 days
    base = d.tail(48)         # ~8 days

    hi = recent["high"].max()
    lo = recent["low"].min()
    last = recent["close"].iloc[-1]
    rng_pct = (hi - lo) / max(lo, 1e-12) * 100

    # Too explosive to call accumulation.
    range_score = np.clip(100 - max(rng_pct - 8, 0) * 7, 0, 100)

    returns = recent["close"].pct_change().dropna()
    down_vol = returns[returns < 0].std() if (returns < 0).any() else 0
    up_vol = returns[returns > 0].std() if (returns > 0).any() else 0.0001
    volatility_score = np.clip(100 - (down_vol / max(up_vol, 1e-6)) * 45, 0, 100)

    position = (last - lo) / max(hi - lo, 1e-12)
    hold_score = np.clip(position * 120, 0, 100)

    vol_recent = recent["quote_volume"].mean()
    vol_base = base["quote_volume"].mean()
    vol_ratio = vol_recent / max(vol_base, 1e-12)
    # Accumulation likes normal/contracting volume, but not dead volume.
    volume_score = 100 if 0.65 <= vol_ratio <= 1.15 else max(0, 100 - abs(vol_ratio - 0.9) * 100)

    # Simple money-flow pressure: close location * volume.
    clv = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / (
        (d["high"] - d["low"]).replace(0, np.nan)
    )
    mf = (clv.fillna(0) * d["quote_volume"]).tail(24).sum()
    mf_score = np.clip(50 + mf / max(recent["quote_volume"].sum(), 1e-12) * 50, 0, 100)

    score = (
        range_score * 0.28
        + volatility_score * 0.17
        + hold_score * 0.20
        + volume_score * 0.15
        + mf_score * 0.20
    )

    return {
        "acc_score": round(float(score), 1),
        "range_pct": round(float(rng_pct), 2),
        "4h_vol_ratio": round(float(vol_ratio), 2),
        "position_in_range": round(float(position * 100), 1),
    }

def volume_expansion(df1):
    """
    Current/last completed 1H candle volume divided by average of previous
    20 completed 1H candles. Excludes the currently-forming candle.
    """
    if df1 is None or len(df1) < 25:
        return None

    # Binance's final row can be the live candle. Use the last completed
    # candle conservatively by excluding the final row.
    d = df1.iloc[:-1].copy()
    if len(d) < 22:
        return None

    current = d.iloc[-1]
    baseline = d["quote_volume"].iloc[-21:-1]
    avg = baseline.mean()
    ratio = current["quote_volume"] / max(avg, 1e-12)

    return {
        "vol_x": round(float(ratio), 2),
        "1h_change": round(float(pct(current["close"], current["open"])), 2),
        "close": float(current["close"]),
        "quote_volume": float(current["quote_volume"]),
    }

def scan_symbol(symbol):
    try:
        d4 = get_klines(symbol, "4h", 70)
        d1 = get_klines(symbol, "1h", 50)
        a = accumulation_score(d4)
        v = volume_expansion(d1)
        if not a or not v:
            return None

        # Momentum should not be mandatory; it is displayed for confirmation.
        if d1 is not None and len(d1) >= 8:
            closed = d1.iloc[:-1]
            mom_6h = pct(closed["close"].iloc[-1], closed["close"].iloc[-7])
            mom_24h = pct(closed["close"].iloc[-1], closed["close"].iloc[-25]) if len(closed) >= 25 else np.nan
        else:
            mom_6h = np.nan
            mom_24h = np.nan

        # Radar score emphasizes accumulation + sudden volume.
        vol_component = np.clip((v["vol_x"] - 1) / 4 * 100, 0, 100)
        score = a["acc_score"] * 0.55 + vol_component * 0.45

        return {
            "Symbol": symbol,
            "Radar Score": round(float(score), 1),
            "Accumulation": a["acc_score"],
            "1H Volume X": v["vol_x"],
            "1H %": v["1h_change"],
            "6H %": round(float(mom_6h), 2),
            "24H %": round(float(mom_24h), 2) if pd.notna(mom_24h) else np.nan,
            "4H Range %": a["range_pct"],
            "4H Vol Ratio": a["4h_vol_ratio"],
            "Range Position %": a["position_in_range"],
            "Price": v["close"],
            "1H Quote Vol": v["quote_volume"],
        }
    except Exception:
        return None

# -----------------------------
# Catalyst radar
# -----------------------------
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The Block": "https://www.theblock.co/rss.xml",
    "Binance Blog": "https://www.binance.com/en/support/announcement/rss",
}

CATALYST_TERMS = {
    "Institutional": [
        "institution", "institutional", "asset manager", "fund", "blackrock",
        "fidelity", "jpmorgan", "jp morgan", "goldman", "morgan stanley",
        "citibank", "bank", "treasury", "etf"
    ],
    "Partnership": [
        "partnership", "partner", "collaboration", "integrat", "strategic alliance",
        "joins forces", "powered by", "adopted"
    ],
    "Adoption": [
        "adoption", "mainnet", "launch", "integration", "payments", "settlement",
        "enterprise", "real-world", "rwa"
    ],
    "Listing": [
        "listed", "listing", "launchpool", "launchpad", "spot listing"
    ],
    "Funding": [
        "funding", "raised", "investment", "invested", "series a", "series b"
    ],
    "Regulation": [
        "approved", "approval", "regulator", "regulation", "license", "licensed"
    ],
}

def catalyst_score(title, summary=""):
    text = (title + " " + summary).lower()
    matched = []
    score = 0
    for cat, terms in CATALYST_TERMS.items():
        hits = [t for t in terms if t in text]
        if hits:
            matched.append(cat)
            score += 1
    # Extra weight for institutional/bank + partnership combinations.
    if "Institutional" in matched:
        score += 2
    if "Partnership" in matched:
        score += 1
    return score, ", ".join(matched) if matched else "General"

@st.cache_data(ttl=300, show_spinner=False)
def get_catalysts():
    rows = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for item in feed.entries[:30]:
                title = item.get("title", "")
                summary = re.sub("<.*?>", " ", item.get("summary", ""))
                score, category = catalyst_score(title, summary)
                if score <= 0:
                    continue
                link = item.get("link", "")
                published = item.get("published", item.get("updated", ""))
                rows.append({
                    "Source": source,
                    "Catalyst Score": score,
                    "Category": category,
                    "Headline": title,
                    "Published": published,
                    "Link": link,
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["Source","Catalyst Score","Category","Headline","Published","Link"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["Catalyst Score", "Published"], ascending=[False, False])
    return df.drop_duplicates(subset=["Headline"]).head(50)

# -----------------------------
# UI
# -----------------------------
st.title("📡 Binance Spot Accumulation + Volume Radar")
st.caption("4H accumulation → 1H abnormal volume expansion → manual technical confirmation")

with st.sidebar:
    st.header("Scanner Settings")
    min_acc = st.slider("Minimum 4H accumulation score", 50, 90, 62)
    min_vol = st.slider("Minimum 1H volume expansion (X)", 1.5, 10.0, 2.0, 0.5)
    min_quote = st.number_input("Minimum 24H quote volume (USDT)", min_value=0.0, value=1000000.0, step=500000.0)
    workers = st.slider("Parallel workers", 2, 12, 8)
    top_n = st.slider("Show top coins", 5, 50, 20)
    auto_refresh = st.checkbox("Auto refresh every 60 sec", value=False)

if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True
    )

# Ticker liquidity filter
try:
    tickers = get_24h_tickers()
    ticker_df = pd.DataFrame(tickers)
    ticker_df["quoteVolume"] = pd.to_numeric(ticker_df["quoteVolume"], errors="coerce")
    ticker_df["priceChangePercent"] = pd.to_numeric(ticker_df["priceChangePercent"], errors="coerce")
    liquid = set(
        ticker_df[
            (ticker_df["symbol"].isin(spot_usdt_symbols())) &
            (ticker_df["quoteVolume"] >= min_quote)
        ]["symbol"].tolist()
    )
except Exception as e:
    st.error(f"Could not load Binance market list: {e}")
    st.stop()

symbols = sorted(liquid)

st.info(
    f"Scanning {len(symbols):,} liquid Binance USDT spot pairs. "
    "The volume trigger is based on the latest completed 1H candle vs the previous 20 completed 1H candles."
)

if st.button("🔄 Scan Now", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

progress = st.progress(0, text="Scanning market...")
results = []

with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(scan_symbol, s): s for s in symbols}
    total = len(futures)
    for i, future in enumerate(as_completed(futures), 1):
        item = future.result()
        if item:
            results.append(item)
        progress.progress(i / total, text=f"Scanning {i:,}/{total:,}")

progress.empty()

df = pd.DataFrame(results)

if df.empty:
    st.warning("No qualifying pairs were found in the current scan.")
else:
    df = df[
        (df["Accumulation"] >= min_acc) &
        (df["1H Volume X"] >= min_vol)
    ].sort_values(["Radar Score", "1H Volume X"], ascending=False).head(top_n)

    st.subheader("🔥 Accumulation → Volume Expansion")
    st.write(
        "These are candidates where the 4H structure looks accumulation-like and "
        "the latest completed 1H candle shows abnormal volume."
    )

    display_cols = [
        "Symbol", "Radar Score", "Accumulation", "1H Volume X",
        "1H %", "6H %", "24H %", "4H Range %",
        "4H Vol Ratio", "Range Position %", "Price"
    ]

    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Radar Score": st.column_config.NumberColumn(format="%.1f"),
            "Accumulation": st.column_config.NumberColumn(format="%.1f"),
            "1H Volume X": st.column_config.NumberColumn(format="%.2fx"),
            "1H %": st.column_config.NumberColumn(format="%.2f%%"),
            "6H %": st.column_config.NumberColumn(format="%.2f%%"),
            "24H %": st.column_config.NumberColumn(format="%.2f%%"),
            "4H Range %": st.column_config.NumberColumn(format="%.2f%%"),
            "4H Vol Ratio": st.column_config.NumberColumn(format="%.2fx"),
            "Range Position %": st.column_config.NumberColumn(format="%.1f%%"),
            "Price": st.column_config.NumberColumn(format="%.8g"),
        },
    )

    st.caption(
        "Suggested workflow: scanner → inspect 4H accumulation → inspect 1H volume candle → "
        "check support/resistance + breakout structure → wait for your confirmation candle → entry."
    )

# Catalyst section
st.divider()
st.header("🧠 Catalyst Radar")
st.caption(
    "Public-news headline scanner. Catalyst tags are keyword-based and should be manually verified "
    "at the original source before treating a headline as material."
)

cat_df = get_catalysts()

if cat_df.empty:
    st.warning("No catalyst headlines matched the current keyword set.")
else:
    for _, row in cat_df.head(20).iterrows():
        st.markdown(
            f"**{row['Catalyst Score']}★ · {row['Category']} · {row['Source']}**  \n"
            f"**{row['Headline']}**  \n"
            f"{row['Published']} · [Open source]({row['Link']})"
        )
        st.divider()

st.subheader("How to use this radar")
st.markdown("""
1. **4H Accumulation:** look for a compressed/range-bound structure rather than an already extended pump.
2. **1H Volume Expansion:** 2x/3x/5x+ means the latest completed 1H candle traded at that multiple of the prior 20-candle average quote volume.
3. **Catalyst:** verify whether the news is real, new, material, and actually related to the token.
4. **Manual TA:** confirm your own support / ascending-triangle or breakout structure, liquidity, BTC context and market regime.
5. **Entry:** this scanner does not enter trades automatically and does not replace your invalidation/position-sizing rules.
""")

st.caption(
    f"Last scan: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
