import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE = "https://api.binance.com"

INTERVAL_4H = "4h"
INTERVAL_1H = "1h"

# 4H structure settings
LOOKBACK_4H = 30
RANGE_LOOKBACK = 12

# 1H volume settings
VOLUME_LOOKBACK_1H = 10

# User-defined thresholds
VOLUME_EXPANSION = 1.5
STRONG_EXPANSION = 3.0

# Resistance proximity
RESISTANCE_LOOKBACK = 30
RESISTANCE_DISTANCE_PCT = 3.0

# Scanner performance
MAX_WORKERS = 8

REQUEST_TIMEOUT = 15


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Binance USDT Accumulation Scanner",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Binance Spot USDT Accumulation Scanner")
st.caption(
    "4H accumulation + tight range + volume analysis + 1H volume expansion + resistance proximity"
)


# ============================================================
# HELPERS
# ============================================================

session = requests.Session()


@st.cache_data(ttl=300)
def get_usdt_symbols():
    """Get Binance Spot USDT symbols."""
    url = f"{BINANCE_BASE}/api/v3/exchangeInfo"

    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    data = r.json()

    symbols = []

    for s in data["symbols"]:
        if (
            s["status"] == "TRADING"
            and s["quoteAsset"] == "USDT"
            and s["isSpotTradingAllowed"]
        ):
            symbols.append(s["symbol"])

    return symbols


@st.cache_data(ttl=60)
def get_klines(symbol, interval, limit):
    """Get Binance OHLCV candles."""
    url = f"{BINANCE_BASE}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    r = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    data = r.json()

    if not data:
        return pd.DataFrame()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=columns)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
    )

    return df


def safe_pct(a, b):
    if b == 0 or pd.isna(b):
        return np.nan

    return ((a - b) / b) * 100


# ============================================================
# 4H ACCUMULATION
# ============================================================

def detect_accumulation(df):
    """
    Heuristic accumulation detector.

    Conditions:
    1. Recent range relatively tight.
    2. Price remains above the range low.
    3. Lower volatility than earlier period.
    4. No major breakdown.
    5. Recent closes are not aggressively trending down.
    """

    if len(df) < LOOKBACK_4H:
        return False, {}

    d = df.tail(LOOKBACK_4H).copy()

    recent = d.tail(RANGE_LOOKBACK)

    range_high = recent["high"].max()
    range_low = recent["low"].min()

    if range_low <= 0:
        return False, {}

    range_pct = ((range_high - range_low) / range_low) * 100

    # Volatility comparison
    d["returns"] = d["close"].pct_change()

    old_vol = d["returns"].head(15).std()
    recent_vol = d["returns"].tail(12).std()

    volatility_contracting = (
        not pd.isna(old_vol)
        and not pd.isna(recent_vol)
        and recent_vol <= old_vol * 1.15
    )

    # Recent price location
    current_price = d["close"].iloc[-1]

    location_in_range = (
        (current_price - range_low)
        / (range_high - range_low)
        if range_high != range_low
        else 0.5
    )

    # Avoid obvious breakdown
    recent_closes = recent["close"]

    breakdown = (
        recent_closes.iloc[-1] < range_low * 0.985
    )

    # Basic downward trend check
    first_close = recent_closes.iloc[0]
    last_close = recent_closes.iloc[-1]

    trend_change = (
        ((last_close - first_close) / first_close) * 100
        if first_close != 0
        else 0
    )

    not_strong_downtrend = trend_change > -8

    # Volume behavior
    old_volume = d["volume"].head(15).mean()
    recent_volume = d["volume"].tail(10).mean()

    volume_ratio = (
        recent_volume / old_volume
        if old_volume > 0
        else np.nan
    )

    volume_stable_or_increasing = (
        not pd.isna(volume_ratio)
        and volume_ratio >= 0.80
    )

    # Tight-range condition
    tight_range = range_pct <= 12

    accumulation = (
        tight_range
        and volatility_contracting
        and not breakdown
        and not_strong_downtrend
        and volume_stable_or_increasing
    )

    details = {
        "range_pct": range_pct,
        "location_in_range": location_in_range * 100,
        "old_volume": old_volume,
        "recent_volume": recent_volume,
        "volume_ratio": volume_ratio,
        "volatility_contracting": volatility_contracting,
        "volume_stable": volume_stable_or_increasing,
        "tight_range": tight_range,
    }

    return accumulation, details


# ============================================================
# 4H VOLUME
# ============================================================

def analyze_4h_volume(df):
    if len(df) < 20:
        return False, np.nan

    recent = df["volume"].tail(8).mean()
    previous = df["volume"].iloc[-16:-8].mean()

    if previous <= 0:
        return False, np.nan

    ratio = recent / previous

    stable_or_increasing = ratio >= 0.80

    return stable_or_increasing, ratio


# ============================================================
# 1H VOLUME EXPANSION
# ============================================================

def analyze_1h_volume(df):
    """
    Current 1H volume / previous 10-candle average volume.
    """

    if len(df) < VOLUME_LOOKBACK_1H + 1:
        return np.nan, "Insufficient Data"

    current_volume = df["volume"].iloc[-1]

    previous_10 = df["volume"].iloc[
        -(VOLUME_LOOKBACK_1H + 1):-1
    ]

    avg_volume = previous_10.mean()

    if avg_volume <= 0:
        return np.nan, "No Data"

    ratio = current_volume / avg_volume

    if ratio >= STRONG_EXPANSION:
        label = "Strong Expansion"

    elif ratio >= VOLUME_EXPANSION:
        label = "Volume Expansion"

    else:
        label = "Normal"

    return ratio, label


# ============================================================
# RESISTANCE
# ============================================================

def analyze_resistance(df):
    """
    Uses recent swing high as practical resistance reference.
    """

    if len(df) < RESISTANCE_LOOKBACK:
        return np.nan, np.nan, False

    recent = df.tail(RESISTANCE_LOOKBACK)

    resistance = recent["high"].max()
    current_price = recent["close"].iloc[-1]

    if resistance <= 0:
        return np.nan, np.nan, False

    distance_pct = (
        (resistance - current_price)
        / resistance
    ) * 100

    near_resistance = (
        distance_pct >= 0
        and distance_pct <= RESISTANCE_DISTANCE_PCT
    )

    return resistance, distance_pct, near_resistance


# ============================================================
# SCAN ONE SYMBOL
# ============================================================

def scan_symbol(symbol):

    try:
        df4 = get_klines(
            symbol,
            INTERVAL_4H,
            max(LOOKBACK_4H + 10, RESISTANCE_LOOKBACK + 10),
        )

        df1 = get_klines(
            symbol,
            INTERVAL_1H,
            VOLUME_LOOKBACK_1H + 5,
        )

        if df4.empty or df1.empty:
            return None

        accumulation, acc_details = detect_accumulation(df4)

        volume_ok, volume_ratio_4h = analyze_4h_volume(df4)

        volume_ratio_1h, volume_status = analyze_1h_volume(df1)

        resistance, resistance_distance, near_resistance = (
            analyze_resistance(df4)
        )

        current_price = df1["close"].iloc[-1]

        # Overall highlight
        accumulation_expansion = (
            accumulation
            and volume_status in [
                "Volume Expansion",
                "Strong Expansion",
            ]
        )

        strong_signal = (
            accumulation
            and volume_status == "Strong Expansion"
        )

        # Only calculate score for sorting
        score = 0

        if accumulation:
            score += 3

        if volume_ok:
            score += 1

        if volume_status == "Volume Expansion":
            score += 2

        if volume_status == "Strong Expansion":
            score += 4

        if near_resistance:
            score += 1

        return {
            "Symbol": symbol,
            "Price": current_price,

            "Accumulation": accumulation,

            "4H Range %": acc_details.get(
                "range_pct",
                np.nan,
            ),

            "4H Volume Ratio": volume_ratio_4h,

            "1H Vol Ratio": volume_ratio_1h,

            "Volume Status": volume_status,

            "Resistance": resistance,

            "Resistance Distance %": resistance_distance,

            "Near Resistance": near_resistance,

            "ACC + Expansion": accumulation_expansion,

            "STRONG Signal": strong_signal,

            "Score": score,
        }

    except Exception:
        return None


# ============================================================
# SCANNER
# ============================================================

def run_scanner(symbols):

    results = []

    progress = st.progress(0)
    status = st.empty()

    total = len(symbols)
    completed = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(scan_symbol, symbol): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):

            result = future.result()

            if result is not None:
                results.append(result)

            completed += 1

            progress.progress(
                min(completed / total, 1.0)
            )

            status.text(
                f"Scanning {completed}/{total} pairs..."
            )

    progress.empty()
    status.empty()

    return pd.DataFrame(results)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Scanner Settings")

range_limit = st.sidebar.slider(
    "Tight Range Maximum %",
    min_value=5.0,
    max_value=20.0,
    value=12.0,
    step=0.5,
)

resistance_limit = st.sidebar.slider(
    "Resistance Distance %",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
)

min_volume_ratio = st.sidebar.slider(
    "Minimum 4H Volume Ratio",
    min_value=0.50,
    max_value=1.50,
    value=0.80,
    step=0.05,
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Volume Expansion: **≥ {VOLUME_EXPANSION}×**"
)

st.sidebar.write(
    f"Strong Expansion: **≥ {STRONG_EXPANSION}×**"
)


# ============================================================
# APPLY SIDEBAR VALUES
# ============================================================

# Update globals from UI
RANGE_TIGHT_LIMIT = range_limit
RESISTANCE_DISTANCE_PCT = resistance_limit


# ============================================================
# MAIN
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Volume Expansion",
        f"≥ {VOLUME_EXPANSION}×",
    )

with col2:
    st.metric(
        "Strong Expansion",
        f"≥ {STRONG_EXPANSION}×",
    )

with col3:
    st.metric(
        "4H Range Limit",
        f"≤ {range_limit:.1f}%",
    )


st.markdown("---")


if st.button(
    "🔎 Scan Binance USDT Pairs",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Loading Binance Spot pairs..."):

        symbols = get_usdt_symbols()

    st.info(
        f"Found {len(symbols)} Binance Spot USDT pairs."
    )

    with st.spinner(
        "Scanning 4H + 1H market structure..."
    ):

        results = run_scanner(symbols)

    if results.empty:

        st.warning(
            "No valid market data was returned."
        )

    else:

        # Sort by strongest signals first
        results = results.sort_values(
            by=[
                "STRONG Signal",
                "ACC + Expansion",
                "Score",
                "1H Vol Ratio",
            ],
            ascending=False,
        )

        # ====================================================
        # SIGNAL SUMMARY
        # ====================================================

        strong = results[
            results["STRONG Signal"] == True
        ]

        acc_expansion = results[
            results["ACC + Expansion"] == True
        ]

        volume_expansion = results[
            results["Volume Status"].isin(
                [
                    "Volume Expansion",
                    "Strong Expansion",
                ]
            )
        ]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accumulation + Expansion",
            len(acc_expansion),
        )

        c2.metric(
            "Strong Expansion",
            len(strong),
        )

        c3.metric(
            "Any Volume Expansion",
            len(volume_expansion),
        )

        # ====================================================
        # HIGHLIGHT SECTION
        # ====================================================

        st.markdown("## 🚨 Accumulation + Volume Expansion")

        if acc_expansion.empty:

            st.info(
                "No Accumulation + Volume Expansion pair found."
            )

        else:

            highlight_cols = [
                "Symbol",
                "Price",
                "Accumulation",
                "4H Range %",
                "4H Volume Ratio",
                "1H Vol Ratio",
                "Volume Status",
                "Resistance Distance %",
                "Near Resistance",
            ]

            highlight_df = acc_expansion[
                highlight_cols
            ].copy()

            st.dataframe(
                highlight_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(
                        format="%.8f"
                    ),
                    "4H Range %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                    "4H Volume Ratio": st.column_config.NumberColumn(
                        format="%.2fx"
                    ),
                    "1H Vol Ratio": st.column_config.NumberColumn(
                        format="%.2fx"
                    ),
                    "Resistance Distance %": st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                },
            )

        # ====================================================
        # STRONG SIGNAL
        # ====================================================

        st.markdown("## 🔥 Accumulation + Strong Expansion")

        if strong.empty:

            st.info(
                "No Accumulation + Strong Expansion pair found."
            )

        else:

            strong_cols = [
                "Symbol",
                "Price",
                "4H Range %",
                "4H Volume Ratio",
                "1H Vol Ratio",
                "Volume Status",
                "Resistance Distance %",
                "Near Resistance",
            ]

            st.dataframe(
                strong[strong_cols],
                use_container_width=True,
                hide_index=True,
            )

        # ====================================================
        # FULL SCAN
        # ====================================================

        st.markdown("## 📋 Full Scan")

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            show_only_acc = st.checkbox(
                "Only Accumulation"
            )

        with filter_col2:

            show_only_expansion = st.checkbox(
                "Only Volume Expansion"
            )

        display_df = results.copy()

        if show_only_acc:

            display_df = display_df[
                display_df["Accumulation"] == True
            ]

        if show_only_expansion:

            display_df = display_df[
                display_df["Volume Status"].isin(
                    [
                        "Volume Expansion",
                        "Strong Expansion",
                    ]
                )
            ]

        display_cols = [
            "Symbol",
            "Price",
            "Accumulation",
            "4H Range %",
            "4H Volume Ratio",
            "1H Vol Ratio",
            "Volume Status",
            "Resistance Distance %",
            "Near Resistance",
            "ACC + Expansion",
            "STRONG Signal",
            "Score",
        ]

        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(
                    format="%.8f"
                ),
                "4H Range %": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
                "4H Volume Ratio": st.column_config.NumberColumn(
                    format="%.2fx"
                ),
                "1H Vol Ratio": st.column_config.NumberColumn(
                    format="%.2fx"
                ),
                "Resistance Distance %": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
            },
        )

        # ====================================================
        # CSV DOWNLOAD
        # ====================================================

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Full Scan CSV",
            csv,
            "binance_usdt_scan.csv",
            "text/csv",
            use_container_width=True,
        )

        st.success(
            f"Scan complete — {len(results)} pairs processed."
        )


else:

    st.info(
        "👆 Click **Scan Binance USDT Pairs** to start."
    )

    st.markdown(
        """
### Scanner Logic

**4H**
- Accumulation structure
- Tight consolidation range
- Volatility contraction
- Stable/increasing volume
- Breakdown avoidance

**1H**
- Current 1H volume ÷ previous 10-candle average
- `≥ 1.5×` → Volume Expansion
- `≥ 3×` → Strong Expansion

**Highlight**
- 🟡 Accumulation + Volume Expansion
- 🔥 Accumulation + Strong Expansion
- Resistance proximity shown separately
"""
    )
