import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Binance USDT Accumulation Scanner",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# BINANCE CONFIG
# ============================================================

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
KLINES_PATH = "/api/v3/klines"
PING_PATH = "/api/v3/ping"

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
MAX_WORKERS = 6


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
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 Binance-Scanner/2.0",
    "Accept": "application/json",
}

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# API REQUEST
# ============================================================

def binance_get(path, params=None):

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

                if status == 200:

                    try:
                        return response.json()
                    except ValueError:
                        last_error = (
                            f"Invalid JSON from {base_url}"
                        )
                        break

                if status in (418, 429):

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = 2 ** attempt

                    time.sleep(min(wait, 15))

                    last_error = (
                        f"Rate limited: HTTP {status}"
                    )

                    continue

                if status == 403:

                    last_error = (
                        f"HTTP 403 from {base_url}"
                    )

                    break

                if status >= 500:

                    last_error = (
                        f"HTTP {status} from {base_url}"
                    )

                    time.sleep(
                        min(2 ** attempt, 8)
                    )

                    continue

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
# SYMBOLS
# ============================================================

@st.cache_data(ttl=600)
def get_usdt_symbols():

    data = binance_get(
        EXCHANGE_INFO_PATH
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "Invalid Binance exchangeInfo response."
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
            "No Spot USDT pairs found."
        )

    return sorted(symbols)


# ============================================================
# KLINES
# ============================================================

@st.cache_data(ttl=60)
def get_klines(
    symbol,
    interval,
    limit,
):

    data = binance_get(
        KLINES_PATH,
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

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
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
# 4H ACCUMULATION
# ============================================================

def detect_accumulation(
    df,
    range_limit,
    volume_min_ratio,
):

    default = {
        "accumulation": False,
        "tight_range": False,
        "range_pct": np.nan,
        "volatility_ratio": np.nan,
        "volume_ratio": np.nan,
        "volume_ok": False,
        "breakdown": False,
        "trend_change_pct": np.nan,
        "price_position": np.nan,
        "range_high": np.nan,
        "range_low": np.nan,
    }

    if len(df) < LOOKBACK_4H:
        return default

    d = df.tail(
        LOOKBACK_4H
    ).copy()

    recent = d.tail(
        ACCUMULATION_RANGE_CANDLES
    ).copy()

    range_high = recent["high"].max()
    range_low = recent["low"].min()

    if range_low <= 0:
        return default

    range_pct = (
        (range_high - range_low)
        / range_low
    ) * 100

    tight_range = (
        range_pct <= range_limit
    )

    # --------------------------------------------------------
    # VOLATILITY
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
    # TREND
    # --------------------------------------------------------

    first_close = recent["close"].iloc[0]
    last_close = recent["close"].iloc[-1]

    if first_close > 0:

        trend_change_pct = (
            (last_close - first_close)
            / first_close
        ) * 100

    else:

        trend_change_pct = np.nan

    not_strong_downtrend = (
        trend_change_pct > -8
    )

    # --------------------------------------------------------
    # BREAKDOWN
    # --------------------------------------------------------

    if len(recent) > 1:

        previous_low = (
            recent["low"]
            .iloc[:-1]
            .min()
        )

        breakdown = (
            last_close
            < previous_low * 0.985
        )

    else:

        breakdown = False

    # --------------------------------------------------------
    # PRICE POSITION
    # --------------------------------------------------------

    range_size = (
        range_high - range_low
    )

    if range_size > 0:

        price_position = (
            (last_close - range_low)
            / range_size
        ) * 100

    else:

        price_position = 50.0

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
        "price_position": price_position,
        "range_high": range_high,
        "range_low": range_low,
    }


# ============================================================
# 4H VOLUME
# ============================================================

def analyze_4h_volume(
    df,
    minimum_ratio,
):

    if len(df) < 20:
        return False, np.nan

    recent = (
        df["volume"]
        .tail(8)
        .mean()
    )

    previous = (
        df["volume"]
        .iloc[-16:-8]
        .mean()
    )

    if previous <= 0:
        return False, np.nan

    ratio = recent / previous

    return (
        ratio >= minimum_ratio,
        ratio,
    )


# ============================================================
# 1H VOLUME
# ============================================================

def analyze_1h_volume(df):

    if len(df) < (
        VOLUME_LOOKBACK_1H + 1
    ):

        return (
            np.nan,
            "Insufficient Data",
        )

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

    return ratio, status


# ============================================================
# RESISTANCE
# ============================================================

def analyze_resistance(
    df,
    distance_limit,
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

    # Exclude current candle.
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

    near = (
        current_price <= resistance
        and distance_pct >= 0
        and distance_pct <= distance_limit
    )

    return (
        resistance,
        distance_pct,
        near,
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(row):

    score = 0

    if bool(row.get("Accumulation", False)):
        score += 3

    if bool(row.get("4H Tight Range", False)):
        score += 1

    if bool(row.get("4H Volume Stable", False)):
        score += 1

    volume_status = row.get(
        "Volume Status",
        "",
    )

    if volume_status == "Volume Expansion":
        score += 2

    elif volume_status == "Strong Expansion":
        score += 4

    if bool(row.get("Near Resistance", False)):
        score += 1

    return score


# ============================================================
# SCAN SYMBOL
# ============================================================

def scan_symbol(
    symbol,
    range_limit,
    volume_min_ratio,
    resistance_distance,
):

    try:

        # ----------------------------------------------------
        # 4H
        # ----------------------------------------------------

        df4 = get_klines(
            symbol,
            "4h",
            max(
                LOOKBACK_4H + 5,
                RESISTANCE_LOOKBACK + 5,
            ),
        )

        if df4.empty or len(df4) < 20:
            return None

        acc = detect_accumulation(
            df4,
            range_limit,
            volume_min_ratio,
        )

        volume_4h_ok, volume_4h_ratio = (
            analyze_4h_volume(
                df4,
                volume_min_ratio,
            )
        )

        # ----------------------------------------------------
        # 1H
        # ----------------------------------------------------

        df1 = get_klines(
            symbol,
            "1h",
            VOLUME_LOOKBACK_1H + 5,
        )

        if df1.empty:
            return None

        volume_1h_ratio, volume_status = (
            analyze_1h_volume(df1)
        )

        # ----------------------------------------------------
        # RESISTANCE
        # ----------------------------------------------------

        (
            resistance,
            resistance_distance_pct,
            near_resistance,
        ) = analyze_resistance(
            df4,
            resistance_distance,
        )

        current_price = (
            df1["close"].iloc[-1]
        )

        # ----------------------------------------------------
        # SIGNALS
        # ----------------------------------------------------

        accumulation = bool(
            acc["accumulation"]
        )

        acc_expansion = (
            accumulation
            and volume_status in (
                "Volume Expansion",
                "Strong Expansion",
            )
        )

        strong_signal = (
            accumulation
            and volume_status
            == "Strong Expansion"
        )

        result = {
            "Symbol": symbol,
            "Price": current_price,

            "Accumulation":
                accumulation,

            "4H Tight Range":
                bool(acc["tight_range"]),

            "4H Range %":
                acc["range_pct"],

            "4H Vol Ratio":
                volume_4h_ratio,

            "4H Volume Stable":
                bool(volume_4h_ok),

            "4H Volatility Ratio":
                acc["volatility_ratio"],

            "1H Vol Ratio":
                volume_1h_ratio,

            "Volume Status":
                volume_status,

            "Resistance":
                resistance,

            "Resistance Distance %":
                resistance_distance_pct,

            "Near Resistance":
                bool(near_resistance),

            "ACC + Expansion":
                bool(acc_expansion),

            "STRONG Signal":
                bool(strong_signal),

            "Score": 0,
        }

        result["Score"] = (
            calculate_score(result)
        )

        return result

    except Exception:

        # One bad symbol must NEVER
        # stop the complete scan.
        return None


# ============================================================
# RUN SCANNER
# ============================================================

def run_scanner(
    symbols,
    range_limit,
    volume_min_ratio,
    resistance_distance,
):

    results = []

    total = len(symbols)

    if total == 0:
        return pd.DataFrame()

    progress = st.progress(0)

    status = st.empty()

    completed = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                scan_symbol,
                symbol,
                range_limit,
                volume_min_ratio,
                resistance_distance,
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(
            futures
        ):

            completed += 1

            try:

                result = future.result()

                if result is not None:
                    results.append(result)

            except Exception:

                pass

            progress.progress(
                completed / total
            )

            status.write(
                f"Scanning "
                f"{completed}/{total}"
            )

    progress.empty()

    status.empty()

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================
# NORMALIZE RESULT COLUMNS
# ============================================================

RESULT_COLUMNS = [
    "Symbol",
    "Price",
    "Accumulation",
    "4H Tight Range",
    "4H Range %",
    "4H Vol Ratio",
    "4H Volume Stable",
    "4H Volatility Ratio",
    "1H Vol Ratio",
    "Volume Status",
    "Resistance",
    "Resistance Distance %",
    "Near Resistance",
    "ACC + Expansion",
    "STRONG Signal",
    "Score",
]


def normalize_results(df):

    if df is None or df.empty:

        return pd.DataFrame(
            columns=RESULT_COLUMNS
        )

    df = df.copy()

    # Ensure every expected column exists.
    for column in RESULT_COLUMNS:

        if column not in df.columns:

            if column in [
                "Accumulation",
                "4H Tight Range",
                "4H Volume Stable",
                "Near Resistance",
                "ACC + Expansion",
                "STRONG Signal",
            ]:

                df[column] = False

            elif column == "Volume Status":

                df[column] = "Normal"

            elif column == "Symbol":

                df[column] = ""

            else:

                df[column] = np.nan

    # Force correct column order.
    df = df[RESULT_COLUMNS]

    # Boolean normalization.
    boolean_columns = [
        "Accumulation",
        "4H Tight Range",
        "4H Volume Stable",
        "Near Resistance",
        "ACC + Expansion",
        "STRONG Signal",
    ]

    for column in boolean_columns:

        df[column] = (
            df[column]
            .fillna(False)
            .astype(bool)
        )

    # Numeric normalization.
    numeric_columns = [
        "Price",
        "4H Range %",
        "4H Vol Ratio",
        "4H Volatility Ratio",
        "1H Vol Ratio",
        "Resistance",
        "Resistance Distance %",
        "Score",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Strongest signals first.
    df = df.sort_values(
        by=[
            "STRONG Signal",
            "ACC + Expansion",
            "Score",
            "1H Vol Ratio",
        ],
        ascending=False,
        na_position="last",
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# SAFE TABLE
# ============================================================

def show_table(df):

    if df is None or df.empty:

        st.info("No results.")
        return

    # IMPORTANT:
    # No pandas Styler is used here.
    # This prevents KeyError when a filtered
    # dataframe has a different column set.

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(
                "Price",
                format="%.8f",
            ),

            "4H Range %": st.column_config.NumberColumn(
                "4H Range %",
                format="%.2f%%",
            ),

            "4H Vol Ratio": st.column_config.NumberColumn(
                "4H Vol Ratio",
                format="%.2fx",
            ),

            "4H Volatility Ratio": st.column_config.NumberColumn(
                "4H Volatility Ratio",
                format="%.2fx",
            ),

            "1H Vol Ratio": st.column_config.NumberColumn(
                "1H Vol Ratio",
                format="%.2fx",
            ),

            "Resistance": st.column_config.NumberColumn(
                "Resistance",
                format="%.8f",
            ),

            "Resistance Distance %": st.column_config.NumberColumn(
                "Resistance Distance %",
                format="%.2f%%",
            ),

            "Score": st.column_config.NumberColumn(
                "Score",
                format="%d",
            ),
        },
    )


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

resistance_distance = st.sidebar.slider(
    "Resistance Distance %",
    min_value=1.0,
    max_value=10.0,
    value=DEFAULT_RESISTANCE_DISTANCE,
    step=0.5,
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"🟡 Volume Expansion: "
    f"**≥ {VOLUME_EXPANSION:.1f}×**"
)

st.sidebar.write(
    f"🔥 Strong Expansion: "
    f"**≥ {STRONG_EXPANSION:.1f}×**"
)

st.sidebar.caption(
    "Binance Spot USDT scanner"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Binance Spot USDT Accumulation Scanner"
)

st.write(
    "4H accumulation + tight range + volume "
    "confirmation + 1H volume expansion + resistance"
)


# ============================================================
# TOP METRICS
# ============================================================

a, b, c, d = st.columns(4)

a.metric(
    "Volume Expansion",
    f"≥ {VOLUME_EXPANSION:.1f}×",
)

b.metric(
    "Strong Expansion",
    f"≥ {STRONG_EXPANSION:.1f}×",
)

c.metric(
    "4H Tight Range",
    f"≤ {range_limit:.1f}%",
)

d.metric(
    "Resistance",
    f"≤ {resistance_distance:.1f}%",
)


st.markdown("---")


# ============================================================
# API TEST
# ============================================================

with st.expander(
    "🔧 Binance API Connection Test"
):

    if st.button(
        "Test Binance API"
    ):

        try:

            ping = binance_get(
                PING_PATH
            )

            st.success(
                "Binance API connection OK."
            )

            st.json(ping)

        except Exception as e:

            st.error(
                "Binance API connection failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# SCAN
# ============================================================

if st.button(
    "🔎 SCAN BINANCE USDT PAIRS",
    type="primary",
    use_container_width=True,
):

    # --------------------------------------------------------
    # SYMBOLS
    # --------------------------------------------------------

    with st.spinner(
        "Loading Binance Spot USDT pairs..."
    ):

        try:

            symbols = get_usdt_symbols()

        except Exception as e:

            st.error(
                "❌ Binance Spot pairs load করা যায়নি."
            )

            st.code(
                str(e)
            )

            st.stop()

    st.success(
        f"{len(symbols)} Spot USDT pairs পাওয়া গেছে."
    )

    # --------------------------------------------------------
    # SCAN
    # --------------------------------------------------------

    start = time.time()

    with st.spinner(
        "Scanning 4H + 1H data..."
    ):

        results = run_scanner(
            symbols,
            range_limit,
            volume_min_ratio,
            resistance_distance,
        )

    elapsed = (
        time.time() - start
    )

    results = normalize_results(
        results
    )

    if results.empty:

        st.warning(
            "Valid market data পাওয়া যায়নি."
        )

        st.stop()

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

    strong_count = int(
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
        "Pairs",
        len(results),
    )

    c2.metric(
        "Accumulation",
        accumulation_count,
    )

    c3.metric(
        "Expansion",
        expansion_count,
    )

    c4.metric(
        "Strong ≥3×",
        strong_count,
    )

    c5.metric(
        "ACC + Expansion",
        acc_expansion_count,
    )

    st.caption(
        f"Completed in {elapsed:.1f} seconds."
    )


    # ========================================================
    # ACCUMULATION + EXPANSION
    # ========================================================

    st.markdown(
        "## 🟡 Accumulation + Volume Expansion"
    )

    highlight = results[
        results["ACC + Expansion"]
    ].copy()

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

    if highlight.empty:

        st.info(
            "কোনো Accumulation + "
            "Volume Expansion পাওয়া যায়নি."
        )

    else:

        show_table(
            highlight[
                highlight_columns
            ]
        )


    # ========================================================
    # STRONG
    # ========================================================

    st.markdown(
        "## 🔥 Accumulation + Strong Expansion ≥3×"
    )

    strong = results[
        results["STRONG Signal"]
    ].copy()

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

    if strong.empty:

        st.info(
            "কোনো Accumulation + "
            "Strong Expansion ≥3× পাওয়া যায়নি."
        )

    else:

        show_table(
            strong[
                strong_columns
            ]
        )


    # ========================================================
    # NEAR RESISTANCE
    # ========================================================

    st.markdown(
        "## 🎯 Accumulation + Near Resistance"
    )

    near = results[
        (
            results["Accumulation"]
        )
        &
        (
            results["Near Resistance"]
        )
    ].copy()

    near_columns = [
        "Symbol",
        "Price",
        "4H Range %",
        "4H Vol Ratio",
        "1H Vol Ratio",
        "Volume Status",
        "Resistance",
        "Resistance Distance %",
        "ACC + Expansion",
    ]

    if near.empty:

        st.info(
            "Accumulation + Near Resistance "
            "pair পাওয়া যায়নি."
        )

    else:

        show_table(
            near[
                near_columns
            ]
        )


    # ========================================================
    # FULL SCAN
    # ========================================================

    st.markdown(
        "## 📋 Full Scan"
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        only_acc = st.checkbox(
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

    if only_acc:

        display = display[
            display["Accumulation"]
        ]

    if only_expansion:

        display = display[
            display["Volume Status"]
            .isin(
                [
                    "Volume Expansion",
                    "Strong Expansion",
                ]
            )
        ]

    if only_resistance:

        display = display[
            display["Near Resistance"]
        ]

    show_table(
        display
    )


    # ========================================================
    # CSV
    # ========================================================

    st.markdown(
        "## 💾 Download"
    )

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Full Scan CSV",
        csv,
        "binance_usdt_accumulation_scan.csv",
        "text/csv",
        use_container_width=True,
    )


# ============================================================
# BEFORE SCAN
# ============================================================

else:

    st.info(
        "👆 উপরের **SCAN BINANCE USDT PAIRS** "
        "button চাপুন."
    )

    st.markdown(
        """
### Scanner Rules

**4H**
- Accumulation structure
- Tight range
- Volatility contraction
- Stable/increasing volume
- Breakdown avoidance

**1H Volume**

`Current 1H Volume ÷ Previous 10 × 1H Average Volume`

- `< 1.5×` → Normal
- `≥ 1.5×` → 🟡 Volume Expansion
- `≥ 3×` → 🔥 Strong Expansion

**Highlights**

- 🟡 Accumulation + Volume Expansion
- 🔥 Accumulation + Strong Expansion
- 🎯 Accumulation + Near Resistance
"""
    )
