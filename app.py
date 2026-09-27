# Risal Narrative Rotation Scanner V1
# Streamlit + Binance Spot public market data
# Cloud-safe: no local CSV/database writes are required.
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Data source: Binance Spot public REST API.
# This app does NOT place orders.

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Risal Narrative Rotation Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BINANCE_BASE = "https://api.binance.com"
TIMEOUT = 12
USER_AGENT = "Risal-Narrative-Rotation-Scanner/1.0"

# ---------------------------------------------------------------------
# Narrative map
# Keep this list editable. Symbols are filtered against Binance Spot
# exchangeInfo at runtime, so delisted/unlisted pairs are ignored.
# ---------------------------------------------------------------------
NARRATIVES = {
    "🏦 RWA / Tokenization": [
        "QNT", "ONDO", "ENA", "POLYX", "CFG", "LINK", "XDC", "MKR", "OM",
    ],
    "💰 DeFi / DEX / Liquidity": [
        "AERO", "VELO", "KMNO", "JTO", "COW", "RUNE", "SPK", "UNI",
        "AAVE", "SUSHI", "MKR", "CAKE", "QUICK", "PENDLE", "CRV",
    ],
    "🤖 AI / Compute": [
        "PHA", "FET", "TAO", "RENDER", "AKT", "IO", "VIRTUAL", "NEAR",
        "GRASS", "ATH", "ARKM",
    ],
    "💳 Payments / Financial Infrastructure": [
        "AMP", "DASH", "XRP", "XLM", "HBAR", "KITE", "CELO", "ACH", "COTI",
        "QNT",
    ],
    "⛓️ L1 / High Performance Chains": [
        "SUI", "SOL", "AVAX", "SEI", "APT", "NEAR", "TON", "INJ", "TIA",
        "BERA", "HYPE",
    ],
    "🔮 Oracle / Data Infrastructure": [
        "PYTH", "LINK", "API3", "BAND", "TRB", "DIA", "UMA", "GRT",
    ],
    "🌐 Decentralized Infrastructure": [
        "2Z", "HNT", "FIL", "AR", "THETA", "AIOZ", "RENDER", "AKT", "IO",
        "GRASS",
    ],
    "🎨 NFT / Gaming": [
        "RARE", "MPLX", "IMX", "GALA", "BEAM", "RON", "AXS", "SAND",
        "MANA", "MAGIC",
    ],
}

# Flatten unique symbols while preserving narrative membership.
SYMBOL_TO_NARRATIVES = {}
for narrative, coins in NARRATIVES.items():
    for coin in coins:
        SYMBOL_TO_NARRATIVES.setdefault(coin, []).append(narrative)


# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_exchange_info():
    r = requests.get(
        f"{BINANCE_BASE}/api/v3/exchangeInfo",
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10, show_spinner=False)
def get_24h_tickers():
    r = requests.get(
        f"{BINANCE_BASE}/api/v3/ticker/24hr",
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=20, show_spinner=False)
def get_klines(symbol, interval="3h", limit=80):
    r = requests.get(
        f"{BINANCE_BASE}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    raw = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Drop the currently forming candle. This prevents the live candle
    # from falsely triggering "confirmation" or volume signals.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["close_time"] <= now].copy()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Technical calculations
# ---------------------------------------------------------------------
def pct(a, b):
    if b in (None, 0) or pd.isna(b):
        return 0.0
    return (a / b - 1.0) * 100.0


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def round_price(x):
    if x >= 1000:
        return round(x, 2)
    if x >= 1:
        return round(x, 4)
    if x >= 0.01:
        return round(x, 6)
    return round(x, 8)


def technicals(df):
    if len(df) < 30:
        return None

    d = df.copy()
    d["ema25"] = d["close"].ewm(span=25, adjust=False).mean()
    d["vol_avg4"] = d["volume"].shift(1).rolling(4).mean()
    d["vol_avg10"] = d["volume"].shift(1).rolling(10).mean()

    # Previous 3 completed candles, excluding the latest completed candle.
    prev = d.iloc[:-1]
    last = d.iloc[-1]

    close = safe_float(last["close"])
    ema25 = safe_float(last["ema25"])
    vol = safe_float(last["volume"])
    vol_avg4 = safe_float(last["vol_avg4"])
    vol_ratio = vol / vol_avg4 if vol_avg4 > 0 else 0.0

    # Recent 3-candle move. User prefers not to chase after a large move.
    close_3 = safe_float(d.iloc[-4]["close"]) if len(d) >= 4 else close
    move_3 = pct(close, close_3)

    # Recent support = lowest low over previous 6 completed candles.
    support = safe_float(prev["low"].tail(6).min()) if len(prev) else close
    resistance = safe_float(prev["high"].tail(6).max()) if len(prev) else close

    # Candle structure
    candle_range = max(last["high"] - last["low"], 1e-12)
    body = abs(last["close"] - last["open"])
    bullish = last["close"] > last["open"]
    body_ratio = body / candle_range

    # 25 EMA relation
    above_ema = close >= ema25
    near_ema = abs(close - ema25) / close <= 0.025 if close else False

    # Support proximity
    near_support = abs(close - support) / close <= 0.03 if close else False

    # Simple breakout/retest logic
    prior_high = safe_float(prev["high"].tail(4).max()) if len(prev) >= 4 else resistance
    breakout = close > prior_high
    retest_hold = near_ema or near_support

    # Confirmation: bullish completed candle + volume expansion.
    confirmation = bullish and body_ratio >= 0.45 and vol_ratio >= 1.5

    # User's preference: strong volume often around 2x average.
    volume_strong = vol_ratio >= 2.0

    # Do not chase a coin that already moved too much recently.
    chase_risk = move_3 >= 8.0

    # Setup score is descriptive, not a probability or guarantee.
    score = 0
    if above_ema:
        score += 20
    if near_ema:
        score += 15
    if near_support:
        score += 15
    if volume_strong:
        score += 20
    elif vol_ratio >= 1.5:
        score += 12
    if bullish:
        score += 10
    if confirmation:
        score += 10
    if breakout:
        score += 10
    if chase_risk:
        score -= 20

    if score >= 75 and confirmation and not chase_risk:
        setup = "🟢 CONFIRMED"
    elif score >= 55 and retest_hold and not chase_risk:
        setup = "🟡 WATCH"
    elif chase_risk:
        setup = "🟠 DON'T CHASE"
    elif above_ema:
        setup = "🔵 DEVELOPING"
    else:
        setup = "⚪ NO SETUP"

    return {
        "price": close,
        "ema25": ema25,
        "vol_ratio": vol_ratio,
        "move_3": move_3,
        "support": support,
        "resistance": resistance,
        "above_ema": above_ema,
        "near_ema": near_ema,
        "near_support": near_support,
        "breakout": breakout,
        "retest_hold": retest_hold,
        "confirmation": confirmation,
        "volume_strong": volume_strong,
        "chase_risk": chase_risk,
        "score": max(0, min(100, score)),
        "setup": setup,
        "last_candle": last["close_time"],
    }


def btc_regime():
    try:
        df = get_klines("BTCUSDT", "4h", 80)
        if len(df) < 30:
            return "⚪ UNKNOWN", "Not enough completed BTC candles"

        d = df.copy()
        d["ema25"] = d["close"].ewm(span=25, adjust=False).mean()
        last = d.iloc[-1]
        prev = d.iloc[-2]

        close = safe_float(last["close"])
        ema = safe_float(last["ema25"])
        prev_ema = safe_float(prev["ema25"])

        if close > ema and ema > prev_ema:
            return "🟢 BULLISH", "BTC completed 4H candle is above a rising 25 EMA."
        if close < ema and ema < prev_ema:
            return "🔴 BEARISH", "BTC completed 4H candle is below a falling 25 EMA."
        return "🟡 MIXED", "BTC 4H structure is not clearly trending."

    except Exception as e:
        return "⚪ UNKNOWN", f"BTC data unavailable: {type(e).__name__}"


# ---------------------------------------------------------------------
# Symbol universe
# ---------------------------------------------------------------------
def get_valid_usdt_symbols():
    info = get_exchange_info()
    valid = set()

    for s in info.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("isSpotTradingAllowed", True)
        ):
            valid.add(s.get("baseAsset", ""))

    return valid


def narrative_universe():
    valid = get_valid_usdt_symbols()
    output = {}

    for narrative, coins in NARRATIVES.items():
        pairs = []
        for coin in coins:
            if coin in valid:
                pairs.append(f"{coin}USDT")
        output[narrative] = sorted(set(pairs))

    return output


# ---------------------------------------------------------------------
# Scan one symbol
# ---------------------------------------------------------------------
def scan_symbol(symbol, ticker_map, interval):
    base = symbol[:-4] if symbol.endswith("USDT") else symbol

    try:
        t = ticker_map.get(symbol)
        if not t:
            return None

        change_24h = safe_float(t.get("priceChangePercent"))
        quote_volume = safe_float(t.get("quoteVolume"))
        price = safe_float(t.get("lastPrice"))

        df = get_klines(symbol, interval, 80)
        tech = technicals(df)
        if not tech:
            return None

        return {
            "Coin": base,
            "Symbol": symbol,
            "24H %": change_24h,
            "3C %": tech["move_3"],
            "Volume USDT": quote_volume,
            "Vol x": tech["vol_ratio"],
            "Price": tech["price"] or price,
            "25 EMA": tech["ema25"],
            "Support": tech["support"],
            "Resistance": tech["resistance"],
            "Above EMA": "YES" if tech["above_ema"] else "NO",
            "Retest": "YES" if tech["retest_hold"] else "NO",
            "Breakout": "YES" if tech["breakout"] else "NO",
            "Confirm": "YES" if tech["confirmation"] else "NO",
            "Score": tech["score"],
            "Setup": tech["setup"],
        }

    except Exception:
        return None


def scan_all(universe, ticker_map, interval, max_workers=8):
    pairs = sorted({p for pairs in universe.values() for p in pairs})
    rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(scan_symbol, p, ticker_map, interval): p for p in pairs
        }
        for f in as_completed(futures):
            result = f.result()
            if result:
                rows.append(result)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Add narrative labels.
    pair_to_narratives = {}
    for n, pairs in universe.items():
        for p in pairs:
            pair_to_narratives.setdefault(p, []).append(n)

    df["Narrative"] = df["Symbol"].map(
        lambda x: " | ".join(pair_to_narratives.get(x, []))
    )

    return df


# ---------------------------------------------------------------------
# Narrative strength
# ---------------------------------------------------------------------
def narrative_summary(scan_df):
    if scan_df.empty:
        return pd.DataFrame()

    rows = []

    for narrative, group in scan_df.groupby(
        scan_df["Symbol"].map(
            lambda x: next(
                (n for n, pairs in universe_cache.items() if x in pairs),
                "Other",
            )
        )
    ):
        if group.empty:
            continue

        leader = group.sort_values("24H %", ascending=False).iloc[0]
        avg_change = group["24H %"].mean()
        median_change = group["24H %"].median()
        positive_pct = (group["24H %"] > 0).mean() * 100
        avg_vol = group["Vol x"].replace([math.inf, -math.inf], pd.NA).dropna()
        avg_vol_ratio = avg_vol.mean() if not avg_vol.empty else 0

        # Participation is stronger when many members are green.
        participation = min(100, positive_pct)

        score = (
            min(40, max(0, avg_change * 2.0))
            + min(25, participation * 0.25)
            + min(25, max(0, (avg_vol_ratio - 1) * 12))
            + min(10, max(0, leader["24H %"]) * 0.2)
        )
        score = round(max(0, min(100, score)), 1)

        laggard = group.sort_values("24H %", ascending=True).iloc[0]

        if score >= 70 and positive_pct >= 60:
            status = "🔥 ROTATION"
        elif score >= 50 and positive_pct >= 50:
            status = "🟡 ACTIVE"
        elif leader["24H %"] >= 8:
            status = "👀 LEADER ONLY"
        else:
            status = "⚪ QUIET"

        rows.append({
            "Narrative": narrative,
            "Strength": score,
            "Leader": leader["Coin"],
            "Leader %": leader["24H %"],
            "Avg %": avg_change,
            "Median %": median_change,
            "Green %": positive_pct,
            "Avg Vol x": avg_vol_ratio,
            "Laggard": laggard["Coin"],
            "Laggard %": laggard["24H %"],
            "Status": status,
        })

    return pd.DataFrame(rows).sort_values(
        ["Strength", "Leader %"], ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("📡 Risal Narrative Rotation Scanner")
st.caption(
    "Live Binance Spot scanner • Narrative rotation • 25 EMA • volume • "
    "support/retest • confirmation"
)

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    interval = st.selectbox(
        "Trading timeframe",
        ["3h", "4h", "1h"],
        index=0,
        help="3H is the primary timeframe for this scanner.",
    )
    min_volume = st.number_input(
        "Minimum 24H volume (USDT)",
        min_value=0.0,
        value=2_000_000.0,
        step=500_000.0,
    )
    max_workers = st.slider("Parallel requests", 2, 10, 6)
    show_all = st.checkbox("Show all scanned coins", value=False)

    st.divider()
    st.subheader("💼 Risk Calculator")
    portfolio = st.number_input(
        "Portfolio Balance (USDT)",
        min_value=1.0,
        value=500.0,
        step=10.0,
    )
    auto_risk = 2.0 if portfolio < 1000 else 1.0
    st.info(f"Risk rule: {'2%' if portfolio < 1000 else '1%'} maximum")
    risk_pct = auto_risk
    risk_usdt = portfolio * risk_pct / 100

    entry = st.number_input("Entry Price", min_value=0.0, value=0.0, format="%.8f")
    sl = st.number_input("Stop Loss Price", min_value=0.0, value=0.0, format="%.8f")

    if entry > 0 and sl > 0 and sl < entry:
        loss_per_coin = entry - sl
        qty = risk_usdt / loss_per_coin
        position_usdt = qty * entry
        st.metric("Maximum Loss at SL", f"${risk_usdt:.2f}")
        st.metric("Position Size", f"${position_usdt:.2f}")
        st.metric("Coin Quantity", f"{qty:.8f}")
    elif entry > 0 and sl > 0:
        st.warning("For a long spot trade, SL should be below Entry.")

    st.divider()
    st.caption("No API key is required. This app uses public Binance market data only.")


# Refresh controls
top1, top2, top3 = st.columns([1, 1, 2])
with top1:
    refresh = st.button("🔄 Refresh Now", use_container_width=True)
with top2:
    auto = st.checkbox("Auto refresh", value=False)
with top3:
    st.caption("Auto refresh uses a normal Streamlit rerun and avoids persistent local files.")

if refresh:
    st.cache_data.clear()
    st.rerun()

# Auto-refresh without relying on st.fragment.
if auto:
    st.markdown(
        """
        <script>
        setTimeout(function(){ window.parent.location.reload(); }, 60000);
        </script>
        """,
        unsafe_allow_html=True,
    )

try:
    universe_cache = narrative_universe()
    ticker_data = get_24h_tickers()
    ticker_map = {
        x.get("symbol"): x
        for x in ticker_data
        if x.get("symbol")
    }

    with st.spinner("Scanning Binance Spot market..."):
        scan_df = scan_all(
            universe_cache,
            ticker_map,
            interval,
            max_workers=max_workers,
        )

    if scan_df.empty:
        st.error("No scan data returned. Try Refresh Now or check Binance connectivity.")
        st.stop()

    # Minimum volume filter.
    scan_df = scan_df[scan_df["Volume USDT"] >= min_volume].copy()

    # BTC regime from completed 4H candle.
    regime, regime_text = btc_regime()

    # -----------------------------------------------------------------
    # Market header
    # -----------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC 4H Regime", regime)
    c2.metric("Coins Scanned", len(scan_df))
    c3.metric("Strong Volume", int((scan_df["Vol x"] >= 2).sum()))
    c4.metric("Confirmed", int((scan_df["Setup"] == "🟢 CONFIRMED").sum()))

    st.caption(f"BTC: {regime_text}")

    # -----------------------------------------------------------------
    # Narrative summary
    # -----------------------------------------------------------------
    rows = []
    for narrative, pairs in universe_cache.items():
        group = scan_df[scan_df["Symbol"].isin(pairs)]
        if group.empty:
            continue

        leader = group.sort_values("24H %", ascending=False).iloc[0]
        laggard = group.sort_values("24H %", ascending=True).iloc[0]

        avg_change = group["24H %"].mean()
        green_pct = (group["24H %"] > 0).mean() * 100
        vol_series = group["Vol x"].replace([math.inf, -math.inf], pd.NA).dropna()
        avg_vol = float(vol_series.mean()) if not vol_series.empty else 0.0

        strength = (
            min(40, max(0, avg_change * 2))
            + min(25, green_pct * 0.25)
            + min(25, max(0, (avg_vol - 1) * 12))
            + min(10, max(0, leader["24H %"]) * 0.2)
        )
        strength = round(max(0, min(100, strength)), 1)

        if strength >= 70 and green_pct >= 60:
            status = "🔥 ROTATION"
        elif strength >= 50 and green_pct >= 50:
            status = "🟡 ACTIVE"
        elif leader["24H %"] >= 8:
            status = "👀 LEADER ONLY"
        else:
            status = "⚪ QUIET"

        rows.append({
            "Narrative": narrative,
            "Strength": strength,
            "Leader": leader["Coin"],
            "Leader %": leader["24H %"],
            "Avg %": avg_change,
            "Green %": green_pct,
            "Avg Vol x": avg_vol,
            "Laggard": laggard["Coin"],
            "Laggard %": laggard["24H %"],
            "Status": status,
        })

    summary = pd.DataFrame(rows).sort_values(
        ["Strength", "Leader %"], ascending=False
    ).reset_index(drop=True)

    st.subheader("🔥 Narrative Rotation Map")
    st.dataframe(
        summary.style.format({
            "Strength": "{:.1f}",
            "Leader %": "{:+.2f}%",
            "Avg %": "{:+.2f}%",
            "Green %": "{:.0f}%",
            "Avg Vol x": "{:.2f}x",
            "Laggard %": "{:+.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------
    # Laggard candidates
    # -----------------------------------------------------------------
    st.subheader("🎯 Laggard Watch — Leader Pump হলে এগুলো দেখবে")

    lag = scan_df[
        (scan_df["24H %"] < 8)
        & (scan_df["Above EMA"] == "YES")
        & (scan_df["Vol x"] >= 1.2)
        & (scan_df["3C %"] < 8)
    ].copy()

    if not lag.empty:
        lag["Rotation Signal"] = lag.apply(
            lambda r: (
                "🟢 SETUP + ROTATION"
                if r["Setup"] == "🟢 CONFIRMED"
                else "🟡 WATCH"
            ),
            axis=1,
        )
        lag = lag.sort_values(
            ["Score", "Vol x", "24H %"],
            ascending=[False, False, False],
        )

        st.dataframe(
            lag[
                [
                    "Coin", "Narrative", "24H %", "3C %",
                    "Vol x", "Above EMA", "Retest", "Breakout",
                    "Confirm", "Score", "Setup", "Rotation Signal",
                ]
            ].head(30).style.format({
                "24H %": "{:+.2f}%",
                "3C %": "{:+.2f}%",
                "Vol x": "{:.2f}x",
                "Score": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("এই মুহূর্তে পরিষ্কার laggard setup পাওয়া যায়নি।")

    # -----------------------------------------------------------------
    # Strong setups
    # -----------------------------------------------------------------
    st.subheader("🟢 Technical Setup Candidates")

    confirmed = scan_df[
        scan_df["Setup"].isin(["🟢 CONFIRMED", "🟡 WATCH"])
    ].copy()

    if not confirmed.empty:
        confirmed = confirmed.sort_values(
            ["Score", "Vol x", "24H %"],
            ascending=[False, False, False],
        )
        st.dataframe(
            confirmed[
                [
                    "Coin", "Narrative", "24H %", "3C %",
                    "Vol x", "Price", "25 EMA", "Support",
                    "Resistance", "Retest", "Breakout", "Confirm",
                    "Score", "Setup",
                ]
            ].head(50).style.format({
                "24H %": "{:+.2f}%",
                "3C %": "{:+.2f}%",
                "Vol x": "{:.2f}x",
                "Price": "{:.8f}",
                "25 EMA": "{:.8f}",
                "Support": "{:.8f}",
                "Resistance": "{:.8f}",
                "Score": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No confirmed/watch setup right now.")

    # -----------------------------------------------------------------
    # Full scan
    # -----------------------------------------------------------------
    if show_all:
        st.subheader("📋 Full Scan")
        st.dataframe(
            scan_df.sort_values(
                ["Score", "24H %"],
                ascending=[False, False],
            ),
            use_container_width=True,
            hide_index=True,
        )

    # Timestamp
    st.caption(
        "Last scan: "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        + " • Only completed candles are used for technical confirmation."
    )

except requests.RequestException as e:
    st.error(
        "Binance market-data request failed. "
        "Check internet access, Binance availability, or try Refresh Now."
    )
    st.caption(f"Technical detail: {type(e).__name__}")

except Exception as e:
    st.error("Scanner could not complete this refresh.")
    st.caption(f"Technical detail: {type(e).__name__}: {e}")
