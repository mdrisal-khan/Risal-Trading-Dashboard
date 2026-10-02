import streamlit as st
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from datetime import datetime, timezone

st.set_page_config(page_title="Risal Accumulation + Volume Scanner", page_icon="🔥", layout="wide")

# ============================================================
# Binance public Spot scanner
# 4H accumulation + completed 1H volume expansion
# ============================================================

BINANCE_BASES = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

TIMEOUT = 10

@st.cache_data(ttl=300, show_spinner=False)
def binance_get(path, params=None):
    last_error = None
    for base in BINANCE_BASES:
        try:
            r = requests.get(base + path, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last_error = f"{r.status_code}: {r.text[:180]}"
        except Exception as e:
            last_error = str(e)
    raise RuntimeError(f"Binance API unavailable. {last_error}")

@st.cache_data(ttl=300, show_spinner=False)
def get_spot_symbols(min_quote_volume):
    data = binance_get("/api/v3/exchangeInfo")
    allowed = {
        s["symbol"] for s in data.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and s.get("isSpotTradingAllowed", True)
        and not any(x in s["symbol"] for x in ["UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"])
    }

    tickers = binance_get("/api/v3/ticker/24hr")
    rows = []
    for t in tickers:
        sym = t.get("symbol")
        if sym not in allowed:
            continue
        try:
            qv = float(t.get("quoteVolume", 0))
            last = float(t.get("lastPrice", 0))
        except Exception:
            continue
        if qv >= min_quote_volume and last > 0:
            rows.append({
                "symbol": sym,
                "quoteVolume24h": qv,
                "price": last
            })

    return pd.DataFrame(rows).sort_values("quoteVolume24h", ascending=False).reset_index(drop=True)

def get_klines(symbol, interval, limit=60):
    return binance_get(
        "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit}
    )

def klines_df(raw):
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades","taker_buy_base",
        "taker_buy_quote","ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df

def accumulation_score(df4):
    # Exclude the still-forming 4H candle.
    if len(df4) < 20:
        return None

    d = df4.iloc[:-1].copy()
    recent = d.tail(18)

    hi = recent["high"].max()
    lo = recent["low"].min()
    last = recent.iloc[-1]
    rng_pct = (hi - lo) / max(lo, 1e-12) * 100

    # Simple accumulation characteristics:
    # 1) not an excessively wide range
    # 2) recent close remains in the upper half of the range
    # 3) recent lows are not continuously collapsing
    # 4) recent range is relatively compressed versus the preceding window
    mid = (hi + lo) / 2
    close_pos = (last["close"] - lo) / max(hi - lo, 1e-12)

    prior = d.tail(36).head(18) if len(d) >= 36 else d.head(18)
    prior_range = (prior["high"].max() - prior["low"].min()) / max(prior["low"].min(), 1e-12) * 100
    compression = recent["close"].std() / max(recent["close"].mean(), 1e-12) * 100

    # Higher score = tighter + stable/holding + upper-half positioning.
    score = 0

    if rng_pct <= 12: score += 35
    elif rng_pct <= 18: score += 25
    elif rng_pct <= 25: score += 12

    if close_pos >= 0.60: score += 25
    elif close_pos >= 0.45: score += 15
    elif close_pos >= 0.35: score += 8

    last6 = recent.tail(6)
    first6 = recent.head(6)
    low_first = first6["low"].min()
    low_last = last6["low"].min()
    if low_last >= low_first * 0.985:
        score += 20
    elif low_last >= low_first * 0.97:
        score += 10

    if prior_range > 0 and rng_pct < prior_range * 0.85:
        score += 20
    elif prior_range > 0 and rng_pct < prior_range:
        score += 10

    if score >= 75:
        label = "🔥 Strong"
    elif score >= 55:
        label = "🟢 Good"
    elif score >= 40:
        label = "🟡 Early"
    else:
        label = "—"

    return {
        "acc_score": score,
        "accumulation": label,
        "range_pct": rng_pct,
        "close_pos": close_pos,
        "range_compression": compression,
        "last4h_close": float(last["close"]),
    }

def scan_one(symbol):
    try:
        raw4 = get_klines(symbol, "4h", 60)
        raw1 = get_klines(symbol, "1h", 36)
        d4 = klines_df(raw4)
        d1 = klines_df(raw1)

        acc = accumulation_score(d4)
        if not acc:
            return None

        # Exclude the currently-forming 1H candle.
        c = d1.iloc[:-1].copy()
        if len(c) < 25:
            return None

        last = c.iloc[-1]
        # Average of the previous 20 completed 1H candles.
        baseline = c.iloc[-21:-1]["volume"].mean()
        if baseline <= 0:
            return None

        vol_mult = float(last["volume"] / baseline)
        vol_quote = float(last["quote_volume"])
        candle_pct = float((last["close"] / last["open"] - 1) * 100)

        if vol_mult >= 5:
            volume_status = "🚀 5×+ MASSIVE"
        elif vol_mult >= 3:
            volume_status = "🔥 3×+"
        elif vol_mult >= 2:
            volume_status = "🟢 2×+"
        else:
            volume_status = "—"

        # Keep only actual volume expansion.
        if vol_mult < volume_threshold:
            return None

        # Main signal: accumulation + volume expansion.
        if acc["acc_score"] >= 75:
            signal = "🚀 ACCUMULATION + MASSIVE VOLUME"
        elif acc["acc_score"] >= 55:
            signal = "🔥 ACCUMULATION + VOLUME"
        else:
            signal = "🟡 VOLUME — CHECK 4H"

        return {
            "Coin": symbol,
            "4H Accumulation": acc["accumulation"],
            "Acc Score": acc["acc_score"],
            "1H Volume": volume_status,
            "Volume ×": round(vol_mult, 2),
            "1H Vol (USDT)": vol_quote,
            "1H Candle %": round(candle_pct, 2),
            "4H Range %": round(acc["range_pct"], 2),
            "Signal": signal,
        }
    except Exception:
        return None

def news_strength(text):
    t = text.lower()
    strong = [
        "partnership", "strategic partnership", "agreement", "collaboration",
        "bank", "institutional", "institution", "investment", "funding",
        "integration", "acquisition", "major launch", "mainnet"
    ]
    hits = sum(1 for k in strong if k in t)
    if hits >= 2:
        return "🔥 Strong"
    if hits == 1:
        return "🟢 Relevant"
    return "🟡 Context"

@st.cache_data(ttl=900, show_spinner=False)
def google_news(symbol):
    base = symbol.replace("USDT", "")
    q = quote(f"{base} crypto partnership OR bank OR institutional OR collaboration OR integration OR investment")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item")[:5]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub = item.findtext("pubDate") or ""
            desc = item.findtext("description") or ""
            strength = news_strength(title + " " + desc)
            if strength != "🟡 Context":
                items.append({
                    "Coin": symbol,
                    "News Strength": strength,
                    "Headline": title,
                    "Published": pub,
                    "Link": link
                })
        return items
    except Exception:
        return []

# ---------------- UI ----------------
st.title("🔥 Risal 4H Accumulation + 1H Volume Scanner")
st.caption("Binance Spot • 4H accumulation • completed 1H candle volume expansion • catalyst/news backup")

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    min_quote_volume = st.number_input(
        "Minimum 24H volume (USDT)",
        min_value=100_000.0,
        value=5_000_000.0,
        step=500_000.0,
        format="%.0f"
    )
    max_coins = st.slider("Coins to scan", 30, 300, 150, 10)
    volume_threshold = st.selectbox("1H volume minimum", [2.0, 3.0, 5.0], index=0)
    workers = st.slider("Parallel requests", 2, 10, 6)
    st.info("Volume multiple = latest completed 1H volume ÷ previous 20 completed 1H average.")

if "volume_threshold" not in globals():
    volume_threshold = 2.0

if st.button("🔄 Scan Now", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    universe = get_spot_symbols(min_quote_volume).head(max_coins)
except Exception as e:
    st.error(f"Scanner could not start: {e}")
    st.stop()

st.write(f"**Scanning {len(universe)} Binance Spot USDT pairs…**")

results = []
progress = st.progress(0)
symbols = universe["symbol"].tolist()

with ThreadPoolExecutor(max_workers=workers) as ex:
    futures = {ex.submit(scan_one, s): s for s in symbols}
    done = 0
    for fut in as_completed(futures):
        done += 1
        progress.progress(done / len(futures))
        item = fut.result()
        if item:
            results.append(item)

progress.empty()

if results:
    df = pd.DataFrame(results)
    df = df.sort_values(["Volume ×", "Acc Score"], ascending=[False, False]).reset_index(drop=True)

    st.subheader("📡 Technical Scanner")
    st.caption("Only coins with the selected 1H volume expansion are shown.")
    st.dataframe(
        df[[
            "Coin","4H Accumulation","Acc Score","1H Volume","Volume ×",
            "1H Vol (USDT)","1H Candle %","4H Range %","Signal"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📰 Catalyst Watchlist — Fundamental Backup")
    st.caption("News is backup/context only. Entry timing remains your manual technical decision.")

    # News only for the strongest technical candidates, limiting requests.
    candidates = (
        df[df["Acc Score"] >= 55]
        .sort_values(["Acc Score","Volume ×"], ascending=False)
        .head(12)["Coin"]
        .tolist()
    )

    news_rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        fs = [ex.submit(google_news, s) for s in candidates]
        for f in as_completed(fs):
            try:
                news_rows.extend(f.result())
            except Exception:
                pass

    if news_rows:
        ndf = pd.DataFrame(news_rows)
        ndf = ndf.drop_duplicates(subset=["Coin","Headline"]).head(40)
        for _, r in ndf.iterrows():
            st.markdown(
                f"**{r['News Strength']} {r['Coin']}** — {r['Headline']}  \n"
                f"`{r['Published']}`  • [News source]({r['Link']})"
            )
    else:
        st.info("No relevant recent catalyst headlines were found for the strongest technical candidates.")

    st.caption(
        "Last scan: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
else:
    st.warning("No coins currently match the selected 1H volume threshold. Try 2× or lower the 24H volume filter.")

st.divider()
st.caption("Educational market scanner. It does not predict whether a coin will pump and does not place trades.")
