import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Binance USDT Accumulation Scanner",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# BINANCE CONFIG
# ============================================================

# Public market-data endpoint.
# No API key is required.
BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

API_PATH_EXCHANGE_INFO = "/api/v3/exchangeInfo"
API_PATH_KLINES = "/api/v3/klines"

REQUEST_TIMEOUT = 15

# Keep concurrency moderate to reduce rate-limit problems.
MAX_WORKERS = 6

# Retry count per endpoint.
MAX_RETRIES = 3


# ============================================================
# SCANNER DEFAULTS
# ============================================================

DEFAULT_RANGE_LIMIT = 12.0

DEFAULT_RESISTANCE_DISTANCE = 3.0

DEFAULT_4H_VOLUME_MIN_RATIO = 0.80

VOLUME_EXPANSION = 1.5

STRONG_EXPANSION = 3.0

LOOKBACK_4H = 36

ACCUMULATION_RANGE_CANDLES = 12

RESISTANCE_LOOKBACK = 30

VOLUME_LOOKBACK_1H = 10


# ============================================================
# HTTP SESSION
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# UTILITY
# ============================================================

def format_price(value):
    if value is None or pd.isna(value):
        return "-"

    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:,.4f}"

    if value >= 0.01:
        return f"{value:,.6f}"

    return f"{value:.10f}"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# BINANCE REQUEST
# ============================================================

def binance_get(path, params=None):
    """
    Try Binance public endpoints with retries and fallback hosts.

    Handles:
    - timeout
    - connection error
    - HTTP 403
    - HTTP 418
    - HTTP 429
    - server errors
    - invalid JSON
    """

    last_error = None

    for base_url in BINANCE_ENDPOINTS:

        url = base_url + path

        for attempt in range(MAX_RETRIES):

            try:

                response = session.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                status = response.status_code

                # -------------------------------
                # SUCCESS
                # -------------------------------

                if status == 200:

                    try:
                        return response.json()

                    except ValueError:

                        last_error = (
                            f"Invalid JSON from {base_url}"
                        )

                        break

                # -------------------------------
                # RATE LIMIT
                # -------------------------------

                if status in (418, 429):

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    try:
                        wait_seconds = float(
                            retry_after
                        )
                    except Exception:
                        wait_seconds = (
                            2 ** attempt
                        )

                    wait_seconds = min(
                        wait_seconds,
                        15,
                    )

                    time.sleep(wait_seconds)

                    last_error = (
                        f"Rate limited: HTTP {status}"
                    )

                    continue

                # -------------------------------
                # WAF / BLOCK
                # -------------------------------

                if status == 403:

                    last_error = (
                        f"HTTP 403 from {base_url}"
                    )

                    break

                # -------------------------------
                # SERVER ERROR
                # -------------------------------

                if status >= 500:

                    last_error = (
                        f"HTTP {status} from {base_url}"
                    )

                    time.sleep(
                        min(2 ** attempt, 8)
                    )

                    continue

                # -------------------------------
                # OTHER HTTP ERROR
                # -------------------------------

                try:
                    body = response.json()
                except Exception:
                    body = response.text[:300]

                last_error = (
                    f"HTTP {status}: {body}"
                )

                break

            except requests.exceptions.Timeout:

                last_error = (
                    f"Timeout: {base_url}"
                )

                time.sleep(
                    min(2 ** attempt, 5)
                )

            except requests.exceptions.ConnectionError as e:

                last_error = (
                    f"Connection error: {e}"
                )

                time.sleep(
                    min(2 ** attempt, 5)
                )

            except requests.exceptions.RequestException as e:

                last_error = (
                    f"Request error: {e}"
                )

                break

    raise RuntimeError(
        last_error or "Unknown Binance API error"
    )


# ============================================================
# GET SYMBOLS
# ============================================================

@st.cache_data(ttl=600)
def get_usdt_symbols():

    data = binance_get(
        API_PATH_EXCHANGE_INFO
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "Binance exchangeInfo returned invalid data."
        )

    symbols = []

    for item in data.get("symbols", []):

        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get(
                "isSpotTradingAllowed",
                False,
            )
        ):

            symbols.append(
                item["symbol"]
            )

    if not symbols:
        raise RuntimeError(
            "No Binance Spot USDT pairs found."
        )

    return sorted(symbols)


# ============================================================
# GET KLINES
# ============================================================

@st.cache_data(ttl=60)
def get_klines(
    symbol,
    interval,
    limit,
):

    data = binance_get(
        API_PATH_KLINES,
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )

    if not isinstance(data, list):
        return pd.DataFrame()

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

    df = pd.DataFrame(
        data,
        columns=columns,
    )

    numeric_columns = [
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

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

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


# ============================================================
# 4H ACCUMULATION DETECTION
# ============================================================

def detect_accumulation(
    df,
    range_limit,
    volume_min_ratio,
):

    if len(df) < LOOKBACK_4H:

        return {
            "accumulation": False,
            "tight_range": False,
            "range_pct": np.nan,
            "volatility_ratio": np.nan,
            "volume_ratio": np.nan,
            "volume_ok": False,
            "breakdown": False,
            "trend_change_pct": np.nan,
        }

    d = df.tail(
        LOOKBACK_4H
    ).copy()

    recent = d.tail(
        ACCUMULATION_RANGE_CANDLES
    ).copy()

    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    range_high = recent["high"].max()

    range_low = recent["low"].min()

    if range_low <= 0:

        return {
            "accumulation": False,
            "tight_range": False,
            "range_pct": np.nan,
            "volatility_ratio": np.nan,
            "volume_ratio": np.nan,
            "volume_ok": False,
            "breakdown": False,
            "trend_change_pct": np.nan,
        }

    range_pct = (
        (range_high - range_low)
        / range_low
    ) * 100

    tight_range = (
        range_pct <= range_limit
    )

    # --------------------------------------------------------
    # VOLATILITY CONTRACTION
    # --------------------------------------------------------

    d["returns"] = (
        d["close"].pct_change()
    )

    old_volatility = (
        d["returns"]
        .head(18)
        .std()
    )

    recent_volatility = (
        d["returns"]
        .tail(12)
        .std()
    )

    if (
        pd.isna(old_volatility)
        or old_volatility <= 0
    ):

        volatility_ratio = np.nan

        volatility_contracting = False

    else:

        volatility_ratio = (
            recent_volatility
            / old_volatility
        )

        volatility_contracting = (
            volatility_ratio <= 1.15
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    old_volume = (
        d["volume"]
        .head(18)
        .mean()
    )

    recent_volume = (
        d["volume"]
        .tail(10)
        .mean()
    )

    if old_volume > 0:

        volume_ratio = (
            recent_volume
            / old_volume
        )

    else:

        volume_ratio = np.nan

    volume_ok = (
        not pd.isna(volume_ratio)
        and volume_ratio >= volume_min_ratio
    )

    # --------------------------------------------------------
    # PRICE TREND
    # --------------------------------------------------------

    first_close = (
        recent["close"].iloc[0]
    )

    last_close = (
        recent["close"].iloc[-1]
    )

    if first_close > 0:

        trend_change_pct = (
            (last_close - first_close)
            / first_close
        ) * 100

    else:

        trend_change_pct = np.nan

    # Don't classify an aggressive dump as accumulation.
    not_strong_downtrend = (
        trend_change_pct > -8
    )

    # --------------------------------------------------------
    # BREAKDOWN
    # --------------------------------------------------------

    previous_range_low = (
        recent["low"].iloc[:-1].min()
    )

    current_close = (
        recent["close"].iloc[-1]
    )

    breakdown = (
        current_close
        < previous_range_low * 0.985
    )

    # --------------------------------------------------------
    # PRICE LOCATION
    # --------------------------------------------------------

    range_size = (
        range_high - range_low
    )

    if range_size > 0:

        price_position = (
            (current_close - range_low)
            / range_size
        )

    else:

        price_position = 0.5

    # --------------------------------------------------------
    # FINAL ACCUMULATION
    # --------------------------------------------------------

    accumulation = all(
        [
            tight_range,
            volatility_contracting,
            volume_ok,
            not_strong_downtrend,
            not breakdown,
        ]
    )

    return {
        "accumulation": accumulation,
        "tight_range": tight_range,
        "range_pct": range_pct,
        "volatility_ratio": volatility_ratio,
        "volume_ratio": volume_ratio,
        "volume_ok": volume_ok,
        "breakdown": breakdown,
        "trend_change_pct": trend_change_pct,
        "price_position": price_position * 100,
        "range_high": range_high,
        "range_low": range_low,
    }


# ============================================================
# 4H VOLUME ANALYSIS
# ============================================================

def analyze_4h_volume(
    df,
    minimum_ratio,
):

    if len(df) < 20:

        return False, np.nan

    recent_volume = (
        df["volume"]
        .tail(8)
        .mean()
    )

    previous_volume = (
        df["volume"]
        .iloc[-16:-8]
        .mean()
    )

    if previous_volume <= 0:

        return False, np.nan

    ratio = (
        recent_volume
        / previous_volume
    )

    stable_or_increasing = (
        ratio >= minimum_ratio
    )

    return (
        stable_or_increasing,
        ratio,
    )


# ============================================================
# 1H VOLUME EXPANSION
# ============================================================

def analyze_1h_volume(df):

    if len(df) < (
        VOLUME_LOOKBACK_1H + 1
    ):

        return (
            np.nan,
            "Insufficient Data",
        )

    # Latest 1H candle = current 1H volume
    current_volume = (
        df["volume"].iloc[-1]
    )

    previous_10 = (
        df["volume"]
        .iloc[
            -(VOLUME_LOOKBACK_1H + 1):-1
        ]
    )

    average_volume = (
        previous_10.mean()
    )

    if (
        pd.isna(average_volume)
        or average_volume <= 0
    ):

        return (
            np.nan,
            "No Data",
        )

    ratio = (
        current_volume
        / average_volume
    )

    if ratio >= STRONG_EXPANSION:

        status = "Strong Expansion"

    elif ratio >= VOLUME_EXPANSION:

        status = "Volume Expansion"

    else:

        status = "Normal"

    return (
        ratio,
        status,
    )


# ============================================================
# RESISTANCE ANALYSIS
# ============================================================

def analyze_resistance(
    df,
    resistance_distance_limit,
):

    if len(df) < RESISTANCE_LOOKBACK:

        return (
            np.nan,
            np.nan,
            False,
        )

    recent = df.tail(
        RESISTANCE_LOOKBACK
    )

    current_price = (
        recent["close"].iloc[-1]
    )

    # Exclude the current candle's high
    # to avoid calling the current price itself resistance.
    resistance = (
        recent["high"]
        .iloc[:-1]
        .max()
    )

    if (
        pd.isna(resistance)
        or resistance <= 0
    ):

        return (
            np.nan,
            np.nan,
            False,
        )

    distance_pct = (
        (resistance - current_price)
        / resistance
    ) * 100

    # If price has already broken resistance,
    # mark it separately rather than "near resistance".
    broken = (
        current_price > resistance
    )

    near_resistance = (
        not broken
        and distance_pct >= 0
        and distance_pct <= resistance_distance_limit
    )

    return (
        resistance,
        distance_pct,
        near_resistance,
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(row):

    score = 0

    if row["Accumulation"]:
        score += 3

    if row["4H Tight Range"]:
        score += 1

    if row["4H Volume Stable"]:
        score += 1

    if row["Volume Status"] == "Volume Expansion":
        score += 2

    if row["Volume Status"] == "Strong Expansion":
        score += 4

    if row["Near Resistance"]:
        score += 1

    return score


# ============================================================
# SCAN ONE SYMBOL
# ============================================================

def scan_symbol(
    symbol,
    range_limit,
    volume_min_ratio,
    resistance_distance_limit,
):

    try:

        # ----------------------------------------------------
        # 4H DATA
        # ----------------------------------------------------

        df4 = get_klines(
            symbol,
            "4h",
            max(
                LOOKBACK_4H + 5,
                RESISTANCE_LOOKBACK + 5,
            ),
        )

        if df4.empty:
            return None

        if len(df4) < 20:
            return None

        # ----------------------------------------------------
        # 4H ACCUMULATION
        # ----------------------------------------------------

        accumulation_data = (
            detect_accumulation(
                df4,
                range_limit,
                volume_min_ratio,
            )
        )

        # ----------------------------------------------------
        # 4H VOLUME
        # ----------------------------------------------------

        volume_ok, volume_ratio_4h = (
            analyze_4h_volume(
                df4,
                volume_min_ratio,
            )
        )

        # ----------------------------------------------------
        # 1H DATA
        # ----------------------------------------------------

        df1 = get_klines(
            symbol,
            "1h",
            VOLUME_LOOKBACK_1H + 5,
        )

        if df1.empty:
            return None

        volume_ratio_1h, volume_status = (
            analyze_1h_volume(df1)
        )

        # ----------------------------------------------------
        # RESISTANCE
        # ----------------------------------------------------

        (
            resistance,
            resistance_distance,
            near_resistance,
        ) = analyze_resistance(
            df4,
            resistance_distance_limit,
        )

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        current_price = (
            df1["close"].iloc[-1]
        )

        # ----------------------------------------------------
        # HIGHLIGHTS
        # ----------------------------------------------------

        accumulation = (
            accumulation_data["accumulation"]
        )

        accumulation_expansion = (
            accumulation
            and volume_status in [
                "Volume Expansion",
                "Strong Expansion",
            ]
        )

        strong_signal = (
            accumulation
            and volume_status
            == "Strong Expansion"
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = {
            "Symbol": symbol,

            "Price": current_price,

            "Accumulation": accumulation,

            "4H Tight Range":
                accumulation_data[
                    "tight_range"
                ],

            "4H Range %":
                accumulation_data[
                    "range_pct"
                ],

            "4H Vol Ratio":
                volume_ratio_4h,

            "4H Volume Stable":
                volume_ok,

            "4H Volatility Ratio":
                accumulation_data[
                    "volatility_ratio"
                ],

            "1H Vol Ratio":
                volume_ratio_1h,

            "Volume Status":
                volume_status,

            "Resistance":
                resistance,

            "Resistance Distance %":
                resistance_distance,

            "Near Resistance":
                near_resistance,

            "ACC + Expansion":
                accumulation_expansion,

            "STRONG Signal":
                strong_signal,

            "Score": 0,
        }

        result["Score"] = (
            calculate_score(result)
        )

        return result

    except Exception:
        return None


# ============================================================
# SCAN ALL
# ============================================================

def run_scanner(
    symbols,
    range_limit,
    volume_min_ratio,
    resistance_distance_limit,
):

    results = []

    total = len(symbols)

    progress = st.progress(0)

    status_text = st.empty()

    completed = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {}

        for symbol in symbols:

            future = executor.submit(
                scan_symbol,
                symbol,
                range_limit,
                volume_min_ratio,
                resistance_distance_limit,
            )

            future_map[future] = symbol

        for future in as_completed(
            future_map
        ):

            completed += 1

            symbol = future_map[future]

            try:

                result = future.result()

                if result is not None:
                    results.append(result)

            except Exception:
                pass

            progress.progress(
                min(
                    completed / total,
                    1.0,
                )
            )

            status_text.write(
                f"Scanning: "
                f"{completed}/{total} "
                f"— {symbol}"
            )

    progress.empty()

    status_text.empty()

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================
# FORMAT RESULT TABLE
# ============================================================

def format_dataframe(df):

    if df.empty:
        return df

    out = df.copy()

    # Sort strongest signals first.
    out = out.sort_values(
        by=[
            "STRONG Signal",
            "ACC + Expansion",
            "Score",
            "1H Vol Ratio",
        ],
        ascending=False,
    )

    return out.reset_index(
        drop=True
    )


# ============================================================
# CUSTOM TABLE STYLE
# ============================================================

def highlight_rows(row):

    if row["STRONG Signal"]:

        return [
            "background-color: #4b160f; color: white"
        ] * len(row)

    if row["ACC + Expansion"]:

        return [
            "background-color: #453500; color: white"
        ] * len(row)

    return [""] * len(row)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Scanner Settings")

range_limit = st.sidebar.slider(
    "4H Tight Range Maximum %",
    min_value=5.0,
    max_value=25.0,
    value=DEFAULT_RANGE_LIMIT,
    step=0.5,
)

volume_min_ratio = st.sidebar.slider(
    "Minimum 4H Volume Ratio",
    min_value=0.50,
    max_value=1.50,
    value=DEFAULT_4H_VOLUME_MIN_RATIO,
    step=0.05,
)

resistance_distance_limit = (
    st.sidebar.slider(
        "Resistance Distance %",
        min_value=1.0,
        max_value=10.0,
        value=DEFAULT_RESISTANCE_DISTANCE,
        step=0.5,
    )
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 1H Volume Rules"
)

st.sidebar.write(
    f"🟡 Volume Expansion: "
    f"**≥ {VOLUME_EXPANSION:.1f}×**"
)

st.sidebar.write(
    f"🔥 Strong Expansion: "
    f"**≥ {STRONG_EXPANSION:.1f}×**"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Data source: Binance public Spot market data"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Binance Spot USDT Accumulation Scanner"
)

st.markdown(
    """
Scans Binance **Spot USDT pairs** for:

- 4H accumulation structure
- 4H tight range
- 4H stable/increasing volume
- 1H current volume vs previous 10 × average volume
- Resistance proximity
- Accumulation + Volume Expansion highlights
"""
)


# ============================================================
# METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "1H Expansion",
    f"≥ {VOLUME_EXPANSION:.1f}×",
)

m2.metric(
    "Strong Expansion",
    f"≥ {STRONG_EXPANSION:.1f}×",
)

m3.metric(
    "4H Range",
    f"≤ {range_limit:.1f}%",
)

m4.metric(
    "Resistance",
    f"≤ {resistance_distance_limit:.1f}%",
)


st.markdown("---")


# ============================================================
# API TEST
# ============================================================

with st.expander(
    "🔧 Binance API Connection Test"
):

    if st.button(
        "Test Binance Connection"
    ):

        try:

            test_data = binance_get(
                "/api/v3/ping"
            )

            st.success(
                "Binance API connection OK."
            )

            st.json(test_data)

        except Exception as e:

            st.error(
                "Binance API connection failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# MAIN SCAN BUTTON
# ============================================================

if st.button(
    "🔎 SCAN BINANCE USDT PAIRS",
    type="primary",
    use_container_width=True,
):

    # --------------------------------------------------------
    # GET SYMBOLS
    # --------------------------------------------------------

    with st.spinner(
        "Loading Binance Spot USDT pairs..."
    ):

        try:

            symbols = get_usdt_symbols()

        except Exception as e:

            st.error(
                "❌ Binance Spot symbol list load করা যায়নি."
            )

            st.code(
                str(e)
            )

            st.warning(
                "উপরের API Connection Test চালিয়ে "
                "endpoint connectivity check করুন."
            )

            st.stop()

    st.success(
        f"Found {len(symbols)} Spot USDT pairs."
    )

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    start_time = time.time()

    with st.spinner(
        "Scanning 4H + 1H market structure..."
    ):

        results = run_scanner(
            symbols,
            range_limit,
            volume_min_ratio,
            resistance_distance_limit,
        )

    elapsed = (
        time.time() - start_time
    )

    if results.empty:

        st.error(
            "No valid market data পাওয়া যায়নি."
        )

        st.stop()

    results = format_dataframe(
        results
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    accumulation_count = int(
        results["Accumulation"].sum()
    )

    expansion_count = int(
        results["Volume Status"]
        .isin(
            [
                "Volume Expansion",
                "Strong Expansion",
            ]
        )
        .sum()
    )

    strong_expansion_count = int(
        (
            results["Volume Status"]
            == "Strong Expansion"
        ).sum()
    )

    acc_expansion_count = int(
        results["ACC + Expansion"].sum()
    )

    strong_signal_count = int(
        results["STRONG Signal"].sum()
    )

    st.markdown(
        "## 📈 Scan Summary"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Pairs Scanned",
        len(results),
    )

    c2.metric(
        "Accumulation",
        accumulation_count,
    )

    c3.metric(
        "Volume Expansion",
        expansion_count,
    )

    c4.metric(
        "Strong ≥3×",
        strong_expansion_count,
    )

    c5.metric(
        "ACC + Expansion",
        acc_expansion_count,
    )

    st.caption(
        f"Scan completed in {elapsed:.1f} seconds."
    )


    # ========================================================
    # MAIN HIGHLIGHT
    # ========================================================

    st.markdown(
        "## 🟡 Accumulation + Volume Expansion"
    )

    highlight = results[
        results["ACC + Expansion"] == True
    ].copy()

    if highlight.empty:

        st.info(
            "এই scan-এ Accumulation + "
            "Volume Expansion পাওয়া যায়নি."
        )

    else:

        highlight_columns = [
            "Symbol",
            "Price",
            "4H Range %",
            "4H Vol Ratio",
            "1H Vol Ratio",
            "Volume Status",
            "Resistance Distance %",
            "Near Resistance",
            "STRONG Signal",
        ]

        highlight = highlight[
            highlight_columns
        ]

        st.dataframe(
            highlight.style.apply(
                highlight_rows,
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # STRONG SECTION
    # ========================================================

    st.markdown(
        "## 🔥 Accumulation + Strong Expansion ≥3×"
    )

    strong = results[
        results["STRONG Signal"] == True
    ].copy()

    if strong.empty:

        st.info(
            "Accumulation + Strong Expansion "
            "≥3× পাওয়া যায়নি."
        )

    else:

        strong_columns = [
            "Symbol",
            "Price",
            "4H Range %",
            "4H Vol Ratio",
            "1H Vol Ratio",
            "Volume Status",
            "Resistance Distance %",
            "Near Resistance",
        ]

        st.dataframe(
            strong[
                strong_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # NEAR RESISTANCE
    # ========================================================

    st.markdown(
        "## 🎯 Accumulation + Near Resistance"
    )

    near_resistance = results[
        (
            results["Accumulation"]
            == True
        )
        &
        (
            results["Near Resistance"]
            == True
        )
    ].copy()

    if near_resistance.empty:

        st.info(
            "Accumulation structure-এর মধ্যে "
            "resistance-এর কাছে কোনো pair পাওয়া যায়নি."
        )

    else:

        near_columns = [
            "Symbol",
            "Price",
            "4H Range %",
            "1H Vol Ratio",
            "Volume Status",
            "Resistance",
            "Resistance Distance %",
            "ACC + Expansion",
        ]

        st.dataframe(
            near_resistance[
                near_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # FULL RESULTS
    # ========================================================

    st.markdown(
        "## 📋 Full Scan Results"
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        only_accumulation = st.checkbox(
            "Only Accumulation"
        )

    with f2:

        only_expansion = st.checkbox(
            "Only Volume Expansion"
        )

    with f3:

        only_resistance = st.checkbox(
            "Only Near Resistance"
        )

    display = results.copy()

    if only_accumulation:

        display = display[
            display["Accumulation"]
            == True
        ]

    if only_expansion:

        display = display[
            display["Volume Status"].isin(
                [
                    "Volume Expansion",
                    "Strong Expansion",
                ]
            )
        ]

    if only_resistance:

        display = display[
            display["Near Resistance"]
            == True
        ]

    display_columns = [
        "Symbol",
        "Price",
        "Accumulation",
        "4H Tight Range",
        "4H Range %",
        "4H Vol Ratio",
        "4H Volume Stable",
        "1H Vol Ratio",
        "Volume Status",
        "Resistance",
        "Resistance Distance %",
        "Near Resistance",
        "ACC + Expansion",
        "STRONG Signal",
        "Score",
    ]

    display = display[
        display_columns
    ]

    st.dataframe(
        display.style.apply(
            highlight_rows,
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    st.markdown(
        "## 💾 Export"
    )

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Full Scan CSV",
        data=csv_data,
        file_name=(
            "binance_usdt_accumulation_scan.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# BEFORE SCAN
# ============================================================

else:

    st.info(
        "👆 **SCAN BINANCE USDT PAIRS** "
        "button চাপলে live scan শুরু হবে."
    )

    st.markdown(
        """
### Scanner Logic

#### 4H Accumulation
- Recent 12 × 4H candles-এর range tight কিনা
- Volatility contraction
- 4H volume stable/increasing
- Strong downside breakdown নেই
- Aggressive downtrend নেই

#### 1H Volume

`Current 1H Volume ÷ Previous 10 × 1H Average Volume`

- `< 1.5×` → Normal
- `≥ 1.5×` → 🟡 **Volume Expansion**
- `≥ 3.0×` → 🔥 **Strong Expansion**

#### Highlight

- 🟡 **Accumulation + Volume Expansion**
- 🔥 **Accumulation + Strong Expansion ≥3×**
- 🎯 Accumulation + Near Resistance

### Important

এটি একটি market-screening tool।  
`Accumulation` এবং `Resistance` এখানে rule-based/heuristic detection—এগুলো guaranteed trading signal নয়।
"""
    )
