"""
Binance Spot Accumulation + 1H Volume Expansion + Catalyst Radar
Streamlit Cloud version.

Uses Binance's public market-data host:
https://data-api.binance.vision
This host is documented by Binance for public market-data endpoints,
including exchangeInfo, klines and ticker/24hr.

Radar only — no automatic trading and no guaranteed signal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
import re
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="Binance Spot Radar",
    page_icon="📡",
    layout="wide",
)

# Binance explicitly documents data-api.binance.vision for public market data.
BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
TIMEOUT = 8
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 Binance-Spot-Radar/1.0"
})

STABLES = {
    "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT",
    "BUSDUSDT", "DAIUSDT", "EURUSDT", "TRYUSDT"
}


def api_get(path, params=None, attempts=2):
    """Try Binance public market-data hosts until one responds successfully."""
    last_error = None

    for base in BASE_URLS:
        for attempt in range(attempts):
            try:
                r = SESSION.get(
                    base + path,
                    params=params or {},
                    timeout=TIMEOUT,
                )

                if r.status_code == 200:
                    return r.json(), base

                # 451/403/429 are host-specific access/rate issues.
                # Move to the next host instead of killing the app.
                last_error = f"{base} -> HTTP {r.status_code}: {r.text[:180]}"

                if r.status_code in (403, 418, 429, 451):
                    break

            except requests.RequestException as e:
                last_error = f"{base} -> {e}"

            if attempt + 1 < attempts:
                time.sleep(0.25)

    raise RuntimeError(last_error or "All Binance market-data endpoints failed.")


@st.cache_data(ttl=300, show_spinner=False)
def get_exchange_info():
    data, base = api_get("/api/v3/exchangeInfo")
    return data, base


@st.cache_data(ttl=30, show_spinner=False)
def get_24h_tickers():
    data, base = api_get("/api/v3/ticker/24hr")
    return data, base


def get_spot_usdt_symbols():
    data, _ = get_exchange_info()
    symbols = []

    for s in data.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", True)
            and s.get("symbol") not in STABLES
        ):
            symbols.append(s["symbol"])

    return symbols


def get_klines(symbol, interval, limit=80):
    try:
        data, _ = api_get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
            attempts=1,
        )

        if not isinstance(data, list) or len(data) < 25:
            return None

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base",
            "taker_quote", "ignore"
        ]

        df = pd.DataFrame(data, columns=cols)

        for c in [
            "open", "high", "low", "close",
            "volume", "quote_volume", "taker_base", "taker_quote"
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    except Exception:
        return None


def pct(new, old):
    if old == 0 or pd.isna(old):
        return 0.0
    return (new / old - 1.0) * 100.0


def accumulation_score(df):
    """
    Transparent heuristic for a quiet/compressed 4H base.

    It does NOT prove accumulation. It ranks structures for manual review.
    """
    if df is None or len(df) < 50:
        return None

    # Exclude the currently-forming candle.
    d = df.iloc[:-1].copy()

    recent = d.tail(24)  # ~4 days
    base = d.tail(48)    # ~8 days

    hi = recent["high"].max()
    lo = recent["low"].min()
    last = recent["close"].iloc[-1]

    range_pct = (hi - lo) / max(lo, 1e-12) * 100
    range_score = float(np.clip(100 - max(range_pct - 8, 0) * 7, 0, 100))

    returns = recent["close"].pct_change().dropna()
    down = returns[returns < 0]
    up = returns[returns > 0]

    down_vol = down.std() if len(down) else 0.0
    up_vol = up.std() if len(up) else 0.0001
    volatility_score = float(
        np.clip(100 - (down_vol / max(up_vol, 1e-6)) * 45, 0, 100)
    )

    position = (last - lo) / max(hi - lo, 1e-12)
    hold_score = float(np.clip(position * 120, 0, 100))

    vr = recent["quote_volume"].mean() / max(base["quote_volume"].mean(), 1e-12)
    volume_score = float(
        np.clip(100 - abs(vr - 0.9) * 100, 0, 100)
    )

    clv = (
        ((d["close"] - d["low"]) - (d["high"] - d["close"]))
        / (d["high"] - d["low"]).replace(0, np.nan)
    ).fillna(0)

    mf = (clv * d["quote_volume"]).tail(24).sum()
    mf_score = float(np.clip(
        50 + mf / max(recent["quote_volume"].sum(), 1e-12) * 50,
        0, 100
    ))

    score = (
        range_score * 0.28
        + volatility_score * 0.17
        + hold_score * 0.20
        + volume_score * 0.15
        + mf_score * 0.20
    )

    return {
        "acc_score": round(score, 1),
        "range_pct": round(float(range_pct), 2),
        "4h_vol_ratio": round(float(vr), 2),
        "range_position": round(float(position * 100), 1),
    }


def volume_expansion(df):
    """
    Latest COMPLETED 1H candle vs previous 20 completed 1H candles.
    """
    if df is None or len(df) < 25:
        return None

    d = df.iloc[:-1].copy()

    current = d.iloc[-1]
    baseline = d["quote_volume"].iloc[-21:-1]

    avg_volume = baseline.mean()
    if avg_volume <= 0:
        return None

    ratio = current["quote_volume"] / avg_volume

    return {
        "vol_x": round(float(ratio), 2),
        "one_h_change": round(
            float(pct(current["close"], current["open"])), 2
        ),
        "price": float(current["close"]),
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

        closed = d1.iloc[:-1]

        h6 = (
            pct(closed["close"].iloc[-1], closed["close"].iloc[-7])
            if len(closed) >= 7 else np.nan
        )

        h24 = (
            pct(closed["close"].iloc[-1], closed["close"].iloc[-25])
            if len(closed) >= 25 else np.nan
        )

        vol_component = np.clip(
            (v["vol_x"] - 1) / 4 * 100,
            0, 100
        )

        radar_score = a["acc_score"] * 0.55 + vol_component * 0.45

        return {
            "Symbol": symbol,
            "Radar Score": round(float(radar_score), 1),
            "4H Accumulation": a["acc_score"],
            "1H Volume X": v["vol_x"],
            "1H %": v["one_h_change"],
            "6H %": round(float(h6), 2),
            "24H %": round(float(h24), 2),
            "4H Range %": a["range_pct"],
            "4H Vol Ratio": a["4h_vol_ratio"],
            "Range Position %": a["range_position"],
            "Price": v["price"],
            "1H Quote Volume": v["quote_volume"],
        }

    except Exception:
        return None


# -----------------------------
# Catalyst Radar
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
        "institution", "institutional", "asset manager", "fund",
        "blackrock", "fidelity", "jpmorgan", "jp morgan",
        "goldman", "morgan stanley", "citibank", "treasury", "etf"
    ],
    "Bank / Finance": [
        "bank", "banking", "financial institution", "payment",
        "settlement", "custody"
    ],
    "Partnership": [
        "partnership", "partner", "collaboration",
        "strategic alliance", "integrat", "adopted"
    ],
    "Adoption": [
        "adoption", "mainnet", "launch", "enterprise",
        "real-world", "rwa", "payments"
    ],
    "Listing": [
        "listed", "listing", "launchpool", "launchpad", "spot listing"
    ],
    "Funding / Investment": [
        "funding", "raised", "investment", "invested",
        "series a", "series b"
    ],
    "Regulation": [
        "approved", "approval", "regulator", "regulation",
        "license", "licensed"
    ],
}


def catalyst_score(title, summary):
    text = (title + " " + summary).lower()

    categories = []
    score = 0

    for category, terms in CATALYST_TERMS.items():
        if any(term in text for term in terms):
            categories.append(category)
            score += 1

    if "Institutional" in categories:
        score += 2

    if "Partnership" in categories:
        score += 1

    if "Bank / Finance" in categories:
        score += 1

    return score, ", ".join(categories) if categories else "General"


@st.cache_data(ttl=300, show_spinner=False)
def get_catalysts():
    rows = []

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)

            for item in feed.entries[:30]:
                title = item.get("title", "")
                summary = re.sub(
                    r"<.*?>", " ",
                    item.get("summary", "")
                )

                score, category = catalyst_score(
                    title, summary
                )

                if score <= 0:
                    continue

                rows.append({
                    "Source": source,
                    "Catalyst Score": score,
                    "Category": category,
                    "Headline": title,
                    "Published": item.get(
                        "published",
                        item.get("updated", "")
                    ),
                    "Link": item.get("link", ""),
                })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["Catalyst Score", "Published"],
        ascending=[False, False]
    )

    return result.drop_duplicates(
        subset=["Headline"]
    ).head(50)


# -----------------------------
# UI
# -----------------------------

st.title("📡 Binance Spot Accumulation + Volume Radar")

st.caption(
    "4H accumulation → 1H abnormal volume → catalyst → manual TA confirmation"
)

with st.sidebar:
    st.header("Scanner Settings")

    min_acc = st.slider(
        "Minimum 4H accumulation score",
        50, 90, 62
    )

    min_vol = st.slider(
        "Minimum 1H volume expansion",
        1.5, 10.0, 2.0, 0.5
    )

    min_quote = st.number_input(
        "Minimum 24H quote volume (USDT)",
        min_value=0.0,
        value=1_000_000.0,
        step=500_000.0
    )

    workers = st.slider(
        "Parallel workers",
        2, 12, 8
    )

    top_n = st.slider(
        "Show top candidates",
        5, 50, 20
    )

    auto_refresh = st.checkbox(
        "Auto refresh every 60 seconds",
        False
    )

if auto_refresh:
    st.markdown(
        '<meta http-equiv="refresh" content="60">',
        unsafe_allow_html=True
    )

# -----------------------------
# Load market universe
# -----------------------------

try:
    ticker_data, ticker_base = get_24h_tickers()
    spot_symbols = set(get_spot_usdt_symbols())

    ticker_df = pd.DataFrame(ticker_data)

    ticker_df["quoteVolume"] = pd.to_numeric(
        ticker_df["quoteVolume"],
        errors="coerce"
    )

    eligible = ticker_df[
        ticker_df["symbol"].isin(spot_symbols)
        & (ticker_df["quoteVolume"] >= min_quote)
    ]

    symbols = sorted(
        eligible["symbol"].dropna().unique().tolist()
    )

except Exception as e:
    st.error(
        "Binance public market-data endpoints could not be reached "
        "from this Streamlit Cloud instance."
    )

    st.code(str(e))

    st.info(
        "The app now tries Binance's public data-api host and several "
        "documented alternate API hosts. If all are blocked, the "
        "Cloud region/network is preventing access."
    )

    st.stop()

st.success(
    f"Market data connected via: {ticker_base}"
)

st.info(
    f"Scanning {len(symbols):,} liquid Binance USDT spot pairs. "
    "1H volume X = latest completed 1H candle / previous 20 completed 1H average."
)

if st.button(
    "🔄 Scan Now",
    type="primary",
    use_container_width=True
):
    st.cache_data.clear()
    st.rerun()

# -----------------------------
# Scanner
# -----------------------------

progress = st.progress(
    0,
    text="Scanning Binance Spot market..."
)

results = []

with ThreadPoolExecutor(
    max_workers=workers
) as executor:

    futures = {
        executor.submit(scan_symbol, symbol): symbol
        for symbol in symbols
    }

    total = len(futures)

    for i, future in enumerate(
        as_completed(futures),
        1
    ):
        result = future.result()

        if result:
            results.append(result)

        progress.progress(
            i / max(total, 1),
            text=f"Scanning {i:,}/{total:,}"
        )

progress.empty()

df = pd.DataFrame(results)

if df.empty:

    st.warning(
        "No candidates passed the current filters."
    )

else:

    df = df[
        (df["4H Accumulation"] >= min_acc)
        & (df["1H Volume X"] >= min_vol)
    ].copy()

    df = df.sort_values(
        ["Radar Score", "1H Volume X"],
        ascending=False
    ).head(top_n)

    st.subheader(
        "🔥 4H Accumulation → 1H Volume Expansion"
    )

    if df.empty:
        st.warning(
            "Market scanned successfully, but no pair "
            "passed your current thresholds."
        )
    else:

        columns = [
            "Symbol",
            "Radar Score",
            "4H Accumulation",
            "1H Volume X",
            "1H %",
            "6H %",
            "24H %",
            "4H Range %",
            "4H Vol Ratio",
            "Range Position %",
            "Price",
        ]

        st.dataframe(
            df[columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Radar Score":
                    st.column_config.NumberColumn(
                        format="%.1f"
                    ),
                "4H Accumulation":
                    st.column_config.NumberColumn(
                        format="%.1f"
                    ),
                "1H Volume X":
                    st.column_config.NumberColumn(
                        format="%.2fx"
                    ),
                "1H %":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                "6H %":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                "24H %":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                "4H Range %":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                "4H Vol Ratio":
                    st.column_config.NumberColumn(
                        format="%.2fx"
                    ),
                "Range Position %":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
                "Price":
                    st.column_config.NumberColumn(
                        format="%.8g"
                    ),
            }
        )

        st.caption(
            "Scanner candidate ≠ entry. Confirm structure, support, "
            "breakout, volume quality, BTC context and invalidation manually."
        )

# -----------------------------
# Catalyst
# -----------------------------

st.divider()

st.header("🧠 Catalyst Radar")

st.caption(
    "Keyword-based public-news radar. Verify the original article "
    "before treating a headline as a material catalyst."
)

catalysts = get_catalysts()

if catalysts.empty:

    st.warning(
        "No matching catalyst headlines found."
    )

else:

    for _, row in catalysts.head(20).iterrows():

        st.markdown(
            f"**{row['Catalyst Score']}★ · "
            f"{row['Category']} · {row['Source']}**"
        )

        st.markdown(
            f"**{row['Headline']}**"
        )

        st.caption(
            f"{row['Published']}  |  "
            f"[Open original source]({row['Link']})"
        )

        st.divider()

# -----------------------------
# Workflow
# -----------------------------

st.subheader("Your workflow")

st.markdown("""
**Scanner**
→ 4H accumulation  
→ 1H 2x / 3x / 5x+ volume expansion  
→ Catalyst check  
→ 4H structure / support  
→ ascending triangle or other valid setup  
→ volume power  
→ breakout  
→ 1H confirmation  
→ manual entry + predefined invalidation

This app is a screening tool, not an automatic trading system.
""")

st.caption(
    "Last scan: "
    + datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
)
