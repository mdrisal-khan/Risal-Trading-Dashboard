import streamlit as st
import requests
import pandas as pd
import time
import math
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================================================
# RISAL TRADING DASHBOARD V6.4
# Stable Data Engine
# =========================================================

st.set_page_config(
    page_title="Risal Trading Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Risal Trading Dashboard")
st.caption(
    "Narrative Intelligence • Trade Radar • Small-Cap Momentum • Coin Scanner"
)


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

# ---------------------------------------------------------
# DATA QUALITY
# ---------------------------------------------------------

MAX_VALID_7D_CHANGE = 75.0
MAX_VALID_RS = 80.0

# Minimum clean-data coverage for strong narrative
MIN_CLEAN_COVERAGE = 0.67

# Minimum clean coins for strong narrative
MIN_STRONG_NARRATIVE_COINS = 3

# ---------------------------------------------------------
# API / PERFORMANCE
# ---------------------------------------------------------

REQUEST_TIMEOUT = 15
MAX_WORKERS = 8

API_RETRIES = 3
API_BACKOFF = 0.8

# ---------------------------------------------------------
# SMALL-CAP LEADER RULES
# ---------------------------------------------------------

STRONG_LEADER_SCORE = 60
STRONG_LEADER_COIN_SCORE = 55
STRONG_LEADER_VOLUME = 1.50
STRONG_LEADER_MOMENTUM = 5.0

DEVELOPING_LEADER_SCORE = 45
DEVELOPING_LEADER_COIN_SCORE = 40
DEVELOPING_LEADER_VOLUME = 1.20
DEVELOPING_LEADER_MOMENTUM = 3.0


# =========================================================
# URLS
# =========================================================

BINANCE_TICKER_URL = (
    "https://api.binance.com/api/v3/ticker/24hr"
)

BINANCE_KLINE_URL = (
    "https://api.binance.com/api/v3/klines"
)

COINGECKO_MARKETS_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
)


# =========================================================
# NARRATIVE MAP
# =========================================================

NARRATIVE_MAP = {

    "Artificial Intelligence (AI)": [
        "FET", "RENDER", "TAO", "NEAR", "AKT",
        "AR", "IO", "AIOZ", "GRASS", "VIRTUAL", "WLD"
    ],

    "Real World Assets (RWA)": [
        "ONDO", "PENDLE", "MKR", "LINK",
        "POLYX", "CPOOL", "TRU", "XDC"
    ],

    "Decentralized Finance (DeFi)": [
        "UNI", "AAVE", "MKR", "CRV", "LDO",
        "COMP", "SNX", "DYDX", "PENDLE",
        "JUP", "RAY", "SUSHI"
    ],

    "Decentralized Exchange (DEX)": [
        "UNI", "JUP", "DYDX", "RAY",
        "SUSHI", "CAKE", "GMX", "ORCA"
    ],

    "Layer 1": [
        "ETH", "SOL", "BNB", "ADA", "AVAX",
        "SUI", "APT", "NEAR", "ATOM",
        "TRX", "SEI", "TON", "XRP"
    ],

    "Layer 2 / Rollup": [
        "ARB", "OP", "STRK", "ZK",
        "MANTA", "METIS", "IMX", "MNT"
    ],

    "DePIN": [
        "RENDER", "FIL", "AR", "HNT",
        "THETA", "AKT", "IO", "GRASS", "AIOZ"
    ],

    "Gaming / GameFi": [
        "IMX", "GALA", "SAND", "MANA",
        "AXS", "BEAM", "RON", "PIXEL", "SUPER"
    ],

    "Meme": [
        "DOGE", "SHIB", "PEPE", "BONK",
        "FLOKI", "WIF", "BRETT", "MEME"
    ],

    "Privacy": [
        "XMR", "ZEC", "DASH", "SCRT", "ROSE"
    ],

    "Oracle": [
        "LINK", "PYTH", "API3", "BAND", "UMA"
    ],

    "Liquid Staking / Restaking": [
        "LDO", "RPL", "ETHFI", "EIGEN", "REZ"
    ],

    "Infrastructure": [
        "LINK", "FIL", "AR", "ICP",
        "TIA", "ATOM", "QNT", "GRT"
    ]
}


# =========================================================
# GENERAL HELPERS
# =========================================================

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


# =========================================================
# THREAD-LOCAL HTTP SESSION
# =========================================================

_THREAD_LOCAL = threading.local()


def get_worker_session():

    if not hasattr(
        _THREAD_LOCAL,
        "session"
    ):

        session = requests.Session()

        session.headers.update({
            "User-Agent": (
                "Risal-Trading-Dashboard/7.0"
            ),
            "Accept": "application/json"
        })

        _THREAD_LOCAL.session = session

    return _THREAD_LOCAL.session


@st.cache_resource
def get_http_session():

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Risal-Trading-Dashboard/7.0"
        ),
        "Accept": "application/json"
    })

    return session


SESSION = get_http_session()


# =========================================================
# ROBUST HTTP REQUEST
# =========================================================

def request_json(
    url,
    params=None,
    retries=API_RETRIES
):

    session = get_worker_session()

    for attempt in range(
        retries + 1
    ):

        try:

            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            status = response.status_code

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            if 200 <= status < 300:

                try:

                    return response.json()

                except Exception:

                    return None

            # -------------------------------------------------
            # RATE LIMIT / TEMPORARY SERVER ERROR
            # -------------------------------------------------

            retryable = (
                status == 429
                or status in {
                    500,
                    502,
                    503,
                    504
                }
            )

            if not retryable:

                return None

            if attempt >= retries:

                return None

            retry_after = response.headers.get(
                "Retry-After"
            )

            try:

                wait_time = float(
                    retry_after
                )

            except Exception:

                wait_time = (
                    API_BACKOFF
                    * (2 ** attempt)
                )

            wait_time = min(
                max(wait_time, 0.25),
                8.0
            )

            time.sleep(
                wait_time
            )

        except (
            requests.RequestException,
            ValueError
        ):

            if attempt >= retries:
                return None

            wait_time = min(
                API_BACKOFF
                * (2 ** attempt),
                8.0
            )

            time.sleep(
                wait_time
            )

        except Exception:

            return None

    return None


# =========================================================
# BINANCE DATA
# =========================================================

@st.cache_data(
    ttl=30,
    show_spinner=False
)
def get_binance_data():

    data = request_json(
        BINANCE_TICKER_URL
    )

    if not isinstance(
        data,
        list
    ):

        return pd.DataFrame()

    rows = []

    for item in data:

        if not isinstance(
            item,
            dict
        ):

            continue

        symbol = str(
            item.get(
                "symbol",
                ""
            )
        ).upper()

        if not symbol.endswith(
            "USDT"
        ):

            continue

        price = safe_float(
            item.get("lastPrice"),
            default=float("nan")
        )

        change = safe_float(
            item.get("priceChangePercent"),
            default=float("nan")
        )

        volume = safe_float(
            item.get("quoteVolume"),
            default=float("nan")
        )

        trades = safe_int(
            item.get("count"),
            default=0
        )

        if not is_finite_number(
            price
        ):

            continue

        if not is_finite_number(
            change
        ):

            continue

        if not is_finite_number(
            volume
        ):

            continue

        if volume < MIN_VOLUME:

            continue

        coin = symbol[:-4].upper()

        rows.append({

            "Symbol": symbol,
            "Coin": coin,
            "Price": price,
            "24H %": change,
            "24H Volume": volume,
            "Trades": trades

        })

    if not rows:

        return pd.DataFrame()

    result = pd.DataFrame(
        rows
    )

    result = result.drop_duplicates(
        subset=["Symbol"],
        keep="first"
    )

    return result


# =========================================================
# COINGECKO MARKET CAP
# =========================================================

@st.cache_data(
    ttl=900,
    show_spinner=False
)
def get_market_caps():

    rows = []

    # -----------------------------------------------------
    # Four pages gives better small/mid-cap coverage
    # -----------------------------------------------------

    for page in range(
        1,
        5
    ):

        params = {

            "vs_currency": "usd",

            "order": "market_cap_desc",

            "per_page": 250,

            "page": page,

            "sparkline": "false"

        }

        data = request_json(
            COINGECKO_MARKETS_URL,
            params=params
        )

        if not isinstance(
            data,
            list
        ):

            break

        if not data:

            break

        for item in data:

            if not isinstance(
                item,
                dict
            ):

                continue

            symbol = str(
                item.get(
                    "symbol",
                    ""
                )
            ).upper().strip()

            market_cap = safe_float(
                item.get(
                    "market_cap"
                ),
                default=float("nan")
            )

            if not symbol:

                continue

            if not is_finite_number(
                market_cap
            ):

                continue

            if market_cap <= 0:

                continue

            rows.append({

                "Coin": symbol,
                "Market Cap": market_cap

            })

        # Small pause prevents unnecessary burst requests
        if page < 4:

            time.sleep(0.20)

    if not rows:

        return pd.DataFrame()

    cap_df = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # Same-symbol protection:
    # Keep largest known market cap for a symbol.
    # -----------------------------------------------------

    cap_df = (
        cap_df
        .groupby(
            "Coin",
            as_index=False
        )[
            "Market Cap"
        ]
        .max()
    )

    return cap_df


# =========================================================
# 7D CHANGE
# =========================================================

def fetch_7d_change(
    symbol
):

    data = request_json(

        BINANCE_KLINE_URL,

        params={

            "symbol": symbol,

            "interval": "1d",

            # 9 candles:
            # 8 closed + current incomplete
            "limit": 9

        }

    )

    if not isinstance(
        data,
        list
    ):

        return symbol, float("nan"), False

    if len(data) < 8:

        return symbol, float("nan"), False

    # -----------------------------------------------------
    # Remove current incomplete candle
    # -----------------------------------------------------

    closed = data[:-1]

    if len(closed) < 8:

        return symbol, float("nan"), False

    try:

        first_close = float(
            closed[0][4]
        )

        last_close = float(
            closed[-1][4]
        )

    except Exception:

        return symbol, float("nan"), False

    if (
        not is_finite_number(first_close)
        or
        not is_finite_number(last_close)
        or
        first_close <= 0
        or
        last_close <= 0
    ):

        return symbol, float("nan"), False

    change = (
        (
            last_close
            - first_close
        )
        / first_close
    ) * 100

    if not is_finite_number(
        change
    ):

        return symbol, float("nan"), False

    return symbol, change, True


# =========================================================
# VOLUME RATIO
# =========================================================

def fetch_volume_ratio(
    symbol
):

    data = request_json(

        BINANCE_KLINE_URL,

        params={

            "symbol": symbol,

            "interval": "1h",

            # 8 closed + current
            "limit": 8

        }

    )

    if not isinstance(
        data,
        list
    ):

        return symbol, float("nan"), False

    if len(data) < 8:

        return symbol, float("nan"), False

    # -----------------------------------------------------
    # Remove current incomplete candle
    # -----------------------------------------------------

    closed = data[:-1]

    if len(closed) < 7:

        return symbol, float("nan"), False

    try:

        latest_volume = float(
            closed[-1][5]
        )

        previous_volumes = [

            float(item[5])
            for item in closed[:-1]

        ]

    except Exception:

        return symbol, float("nan"), False

    if (
        not is_finite_number(
            latest_volume
        )
        or
        latest_volume < 0
    ):

        return symbol, float("nan"), False

    if not previous_volumes:

        return symbol, float("nan"), False

    valid_previous = [

        value

        for value in previous_volumes

        if is_finite_number(value)
        and value >= 0

    ]

    if not valid_previous:

        return symbol, float("nan"), False

    average_volume = (
        sum(valid_previous)
        / len(valid_previous)
    )

    if (
        not is_finite_number(
            average_volume
        )
        or
        average_volume <= 0
    ):

        return symbol, float("nan"), False

    ratio = (
        latest_volume
        / average_volume
    )

    if not is_finite_number(
        ratio
    ):

        return symbol, float("nan"), False

    return symbol, ratio, True


# =========================================================
# SINGLE SYMBOL DEEP FETCH
# =========================================================

def fetch_symbol_market_data(
    symbol
):

    seven_day_result = fetch_7d_change(
        symbol
    )

    volume_result = fetch_volume_ratio(
        symbol
    )

    (
        _,
        change_7d,
        seven_day_valid
    ) = seven_day_result

    (
        _,
        volume_ratio,
        volume_valid
    ) = volume_result

    return symbol, {

        "7D %": change_7d,

        "7D Valid": seven_day_valid,

        "Volume Ratio": volume_ratio,

        "Volume Valid": volume_valid

    }


# =========================================================
# PARALLEL DEEP DATA
# =========================================================

@st.cache_data(
    ttl=30,
    show_spinner=False
)
def get_deep_market_data(
    symbols
):

    symbols = tuple(
        sorted(
            set(
                symbols
            )
        )
    )

    results = {}

    if not symbols:

        return results

    # -----------------------------------------------------
    # One executor for both 7D + Volume requests
    # -----------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {

            executor.submit(
                fetch_symbol_market_data,
                symbol
            ): symbol

            for symbol in symbols

        }

        for future in as_completed(
            futures
        ):

            original_symbol = futures[
                future
            ]

            try:

                symbol, data = (
                    future.result()
                )

                results[
                    symbol
                ] = data

            except Exception:

                results[
                    original_symbol
                ] = {

                    "7D %": float("nan"),

                    "7D Valid": False,

                    "Volume Ratio": float("nan"),

                    "Volume Valid": False

                }

    return results


# =========================================================


# =========================================================
# ⚡ V7 LIVE CONTROL
# =========================================================

if "live_enabled" not in st.session_state:
    st.session_state.live_enabled = True

if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 30


with st.sidebar:

    st.header("⚡ Live Control")

    live_enabled = st.toggle(
        "🟢 Live Auto Refresh",
        value=st.session_state.live_enabled,
        key="live_enabled"
    )

    refresh_interval = st.selectbox(
        "🔄 Refresh Interval",
        options=[30, 60, 120],
        index=[30, 60, 120].index(
            st.session_state.refresh_interval
        ),
        format_func=lambda x: f"{x} seconds",
        key="refresh_interval"
    )

    st.caption(
        "Binance data auto-refresh হবে "
        "selected interval অনুযায়ী।"
    )

    if st.button(
        "🔄 Refresh Now",
        use_container_width=True
    ):

        st.cache_data.clear()

        st.rerun()

    if live_enabled:

        st.success(
            f"🟢 LIVE • Every {refresh_interval}s"
        )

    else:

        st.warning(
            "⏸️ PAUSED"
        )


# =========================================================
# LAST UPDATED
# =========================================================

def get_bd_time():

    try:

        from zoneinfo import ZoneInfo

        return datetime.now(
            ZoneInfo("Asia/Dhaka")
        )

    except Exception:

        return datetime.now()


if live_enabled:

    run_every = (
        f"{refresh_interval}s"
    )

else:

    run_every = None


# =========================================================
# LIVE DASHBOARD FRAGMENT
# =========================================================

@st.fragment(
    run_every=run_every
)
def live_dashboard():

    st.caption(
        "🟢 LIVE DATA • "
        f"Last updated: "
        f"{get_bd_time():%Y-%m-%d %H:%M:%S} "
        "(BD Time)"
    )

    # LOAD BINANCE

    # =========================================================

    df = get_binance_data()

    if df.empty:

        st.error(
            "Binance data পাওয়া যায়নি। কিছুক্ষণ পর আবার Refresh করুন।"
        )

        st.stop()


    # =========================================================
    # LOAD MARKET CAP
    # =========================================================

    cap_df = get_market_caps()

    if not cap_df.empty:

        df = df.merge(
            cap_df,
            on="Coin",
            how="left"
        )

    else:

        df["Market Cap"] = float("nan")


    # =========================================================
    # BTC BASELINE
    # =========================================================

    btc_result = get_deep_market_data(
        ["BTCUSDT"]
    )

    btc_data = btc_result.get(
        "BTCUSDT",
        {}
    )

    btc_7d_raw = btc_data.get(
        "7D %",
        float("nan")
    )

    btc_7d_valid = (
        btc_data.get(
            "7D Valid",
            False
        )
        and
        is_finite_number(
            btc_7d_raw
        )
    )

    if btc_7d_valid:

        btc_7d = float(
            btc_7d_raw
        )

    else:

        btc_7d = float("nan")


    # =========================================================
    # MARKET OVERVIEW
    # =========================================================

    st.divider()

    st.subheader(
        "📊 Market Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "USDT Markets",
        len(df)
    )

    if btc_7d_valid:

        c2.metric(
            "BTC 7D",
            f"{btc_7d:.2f}%"
        )

    else:

        c2.metric(
            "BTC 7D",
            "N/A"
        )

    momentum_markets = df[
        df["24H %"]
        >= MIN_MOMENTUM
    ]

    c3.metric(
        "Momentum Markets",
        len(momentum_markets)
    )

    small_cap_markets = df[
        (
            df["Market Cap"]
            >= SMALL_CAP_MIN
        )
        &
        (
            df["Market Cap"]
            <= SMALL_CAP_MAX
        )
    ]

    c4.metric(
        "Small-Cap Markets",
        len(small_cap_markets)
    )


    # =========================================================
    # SELECTIVE DEEP SCAN
    # =========================================================

    narrative_symbols = set()

    for coins in NARRATIVE_MAP.values():

        for coin in coins:

            narrative_symbols.add(
                coin.upper()
                + "USDT"
            )


    top_momentum_symbols = set(

        df.sort_values(
            "24H %",
            ascending=False
        )
        .head(
            DEEP_SCAN_LIMIT
        )
        ["Symbol"]
        .tolist()

    )


    top_volume_symbols = set(

        df.sort_values(
            "24H Volume",
            ascending=False
        )
        .head(
            DEEP_SCAN_LIMIT
        )
        ["Symbol"]
        .tolist()

    )


    small_candidates = df[
        (
            df["Market Cap"]
            >= SMALL_CAP_MIN
        )
        &
        (
            df["Market Cap"]
            <= SMALL_CAP_MAX
        )
        &
        (
            df["24H %"]
            >= 1.0
        )
    ]

    small_cap_symbols = set(

        small_candidates
        .sort_values(
            "24H %",
            ascending=False
        )
        .head(50)
        ["Symbol"]
        .tolist()

    )


    deep_symbols = (
        narrative_symbols
        |
        top_momentum_symbols
        |
        top_volume_symbols
        |
        small_cap_symbols
    )


    deep_df = df[
        df["Symbol"].isin(
            deep_symbols
        )
    ].copy()


    # =========================================================
    # DEEP DATA
    # =========================================================

    deep_data = get_deep_market_data(
        deep_df["Symbol"].tolist()
    )


    # =========================================================
    # COIN SCANNER
    # =========================================================

    st.divider()

    st.subheader(
        "🔎 Coin Scanner"
    )

    scanner_rows = []

    progress = st.progress(
        0
    )

    total = len(
        deep_df
    )

    for index, (_, row) in enumerate(
        deep_df.iterrows()
    ):

        symbol = row["Symbol"]

        change_24h = safe_float(
            row["24H %"],
            default=0.0
        )

        volume = safe_float(
            row["24H Volume"],
            default=0.0
        )

        trades = safe_int(
            row["Trades"],
            default=0
        )

        market_cap = row[
            "Market Cap"
        ]

        coin_data = deep_data.get(
            symbol,
            {}
        )

        change_7d_raw = coin_data.get(
            "7D %",
            float("nan")
        )

        volume_ratio_raw = coin_data.get(
            "Volume Ratio",
            float("nan")
        )

        change_7d = safe_float(
            change_7d_raw,
            default=float("nan")
        )

        volume_ratio = safe_float(
            volume_ratio_raw,
            default=float("nan")
        )

        seven_day_valid = (
            coin_data.get(
                "7D Valid",
                False
            )
            and
            is_finite_number(
                change_7d
            )
        )

        volume_valid = (
            coin_data.get(
                "Volume Valid",
                False
            )
            and
            is_finite_number(
                volume_ratio
            )
        )


        # =====================================================
        # DATA QUALITY
        # =====================================================

        quality_issues = []

        if not pd.notna(
            market_cap
        ):

            quality_issues.append(
                "Missing Market Cap"
            )

        if not seven_day_valid:

            quality_issues.append(
                "Missing 7D Data"
            )

        if not volume_valid:

            quality_issues.append(
                "Missing Volume Data"
            )

        if (
            seven_day_valid
            and
            abs(change_7d)
            > MAX_VALID_7D_CHANGE
        ):

            quality_issues.append(
                "Extreme 7D Move"
            )


        # =====================================================
        # RS VS BTC
        # =====================================================

        if (
            btc_7d_valid
            and
            seven_day_valid
        ):

            rs = (
                change_7d
                - btc_7d
            )

        else:

            rs = float("nan")


        if (
            is_finite_number(rs)
            and
            abs(rs)
            > MAX_VALID_RS
        ):

            quality_issues.append(
                "Extreme RS"
            )


        # -----------------------------------------------------
        # BTC baseline protection
        # -----------------------------------------------------

        if not btc_7d_valid:

            quality_issues.append(
                "BTC Baseline Missing"
            )


        # =====================================================
        # DATA QUALITY TEXT
        # =====================================================

        if quality_issues:

            data_quality = (
                "⚠️ "
                + " • ".join(
                    quality_issues
                )
            )

        else:

            data_quality = "OK"


        # =====================================================
        # SAFE SCORE VALUES
        # =====================================================

        score_7d = (
            change_7d
            if (
                seven_day_valid
                and
                abs(change_7d)
                <= MAX_VALID_7D_CHANGE
            )
            else 0.0
        )


        score_rs = (
            rs
            if (
                is_finite_number(rs)
                and
                abs(rs)
                <= MAX_VALID_RS
            )
            else 0.0
        )


        score_volume_ratio = (
            volume_ratio
            if volume_valid
            else 0.0
        )


        # =====================================================
        # SCORE
        # =====================================================

        liquidity_score = min(
            max(
                volume,
                0
            ) / 50_000_000,
            1
        ) * 25


        momentum_score = min(
            max(
                change_24h,
                0
            ) / 10,
            1
        ) * 20


        rs_score = min(
            max(
                score_rs,
                0
            ) / 10,
            1
        ) * 25


        volume_score = min(
            max(
                score_volume_ratio - 1,
                0
            ) / 2,
            1
        ) * 20


        activity_score = min(
            max(
                trades,
                0
            ) / 100_000,
            1
        ) * 10


        score = (
            liquidity_score
            + momentum_score
            + rs_score
            + volume_score
            + activity_score
        )


        # =====================================================
        # CAP CLASS
        # =====================================================

        if pd.notna(
            market_cap
        ):

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


        # =====================================================
        # CLEAN DATA
        # =====================================================

        clean_data = (
            data_quality == "OK"
        )


        # =====================================================
        # STATUS
        # =====================================================

        if (
            clean_data
            and
            score >= 70
            and
            rs > 0
            and
            volume_ratio >= MIN_VOLUME_RATIO
        ):

            status = (
                "🔥 A-Level Candidate"
            )

        elif (
            clean_data
            and
            score >= 60
            and
            rs > 0
        ):

            status = (
                "🟢 Watchlist"
            )

        elif (
            clean_data
            and
            volume_ratio >= MIN_VOLUME_RATIO
            and
            change_24h >= MIN_MOMENTUM
        ):

            status = (
                "🟡 Volume Alert"
            )

        else:

            status = (
                "⚪ Developing"
            )


        # =====================================================
        # SMALL CAP FLAG
        # =====================================================

        is_small_cap = False

        if pd.notna(
            market_cap
        ):

            is_small_cap = (
                SMALL_CAP_MIN
                <= market_cap
                <= SMALL_CAP_MAX
            )


        scanner_rows.append({

            "Symbol": symbol,

            "Price": row["Price"],

            "Market Cap": market_cap,

            "Cap Class": cap_class,

            "24H %": round(
                change_24h,
                2
            ),

            "7D %": (
                round(change_7d, 2)
                if is_finite_number(change_7d)
                else float("nan")
            ),

            "RS vs BTC": (
                round(rs, 2)
                if is_finite_number(rs)
                else float("nan")
            ),

            "Volume Ratio": (
                round(volume_ratio, 2)
                if is_finite_number(volume_ratio)
                else float("nan")
            ),

            "24H Volume": round(
                volume,
                0
            ),

            "Score": round(
                score,
                1
            ),

            "Status": status,

            "Small Cap": is_small_cap,

            "Data Quality": data_quality

        })


        progress.progress(
            (index + 1)
            / max(
                total,
                1
            )
        )


    progress.empty()


    scanner_df = pd.DataFrame(
        scanner_rows
    )


    # =========================================================
    # NARRATIVE ENGINE
    # =========================================================

    narrative_rows = []


    for narrative, coins in NARRATIVE_MAP.items():

        available = []

        for coin in coins:

            symbol = (
                coin.upper()
                + "USDT"
            )

            match = scanner_df[
                scanner_df["Symbol"]
                == symbol
            ]

            if not match.empty:

                available.append(
                    match.iloc[0]
                )


        if not available:

            continue


        # =====================================================
        # CLEAN NARRATIVE DATA
        # =====================================================

        clean_available = [

            item

            for item in available

            if item["Data Quality"]
            == "OK"

        ]


        clean_count = len(
            clean_available
        )

        available_count = len(
            available
        )


        if available_count > 0:

            clean_coverage = (
                clean_count
                /
                available_count
            )

        else:

            clean_coverage = 0.0


        # -----------------------------------------------------
        # If no clean data, don't fabricate narrative metrics
        # -----------------------------------------------------

        if not clean_available:

            continue


        analysis_items = (
            clean_available
        )


        # =====================================================
        # BREADTH
        # =====================================================

        positive = sum(

            1

            for item
            in analysis_items

            if (
                is_finite_number(
                    item["24H %"]
                )
                and
                item["24H %"] > 0
            )

        )


        breadth = (
            positive
            /
            len(analysis_items)
        ) * 100


        # =====================================================
        # AVERAGES
        # =====================================================

        avg_momentum = (
            sum(
                item["24H %"]
                for item
                in analysis_items
                if is_finite_number(
                    item["24H %"]
                )
            )
            /
            len(analysis_items)
        )


        avg_rs = (
            sum(
                item["RS vs BTC"]
                for item
                in analysis_items
                if is_finite_number(
                    item["RS vs BTC"]
                )
            )
            /
            len(analysis_items)
        )


        avg_volume = (
            sum(
                item["Volume Ratio"]
                for item
                in analysis_items
                if is_finite_number(
                    item["Volume Ratio"]
                )
            )
            /
            len(analysis_items)
        )


        # =====================================================
        # LEADER
        # =====================================================

        clean_sorted = sorted(

            clean_available,

            key=lambda item: (
                safe_float(
                    item["Score"],
                    0
                )
            ),

            reverse=True

        )


        leader = clean_sorted[0]


        # =====================================================
        # 4 CONFIRMATIONS
        # =====================================================

        confirmations = 0


        # 1. Breadth + clean coverage
        if (
            breadth >= 60
            and
            clean_coverage
            >= MIN_CLEAN_COVERAGE
        ):

            confirmations += 1


        # 2. Relative strength
        if (
            is_finite_number(avg_rs)
            and
            avg_rs > 0
        ):

            confirmations += 1


        # 3. Leader RS
        if (
            is_finite_number(
                leader["RS vs BTC"]
            )
            and
            leader["RS vs BTC"] > 0
        ):

            confirmations += 1


        # 4. Volume expansion
        if (
            is_finite_number(avg_volume)
            and
            avg_volume >= MIN_VOLUME_RATIO
        ):

            confirmations += 1


        # =====================================================
        # NARRATIVE STATUS
        # =====================================================

        if (
            confirmations == 4
            and
            breadth >= 60
            and
            avg_rs > 0
            and
            avg_momentum >= 0
            and
            clean_count
            >= MIN_STRONG_NARRATIVE_COINS
            and
            clean_coverage
            >= MIN_CLEAN_COVERAGE
        ):

            status = "🔥 STRONG"

            confidence = "HIGH"


        elif (
            confirmations >= 3
            and
            breadth >= 50
            and
            avg_rs > 0
            and
            avg_momentum >= 0
            and
            clean_count >= 2
        ):

            status = "🟢 DEVELOPING"

            confidence = "MEDIUM"


        elif (
            breadth >= 60
            and
            avg_momentum > 0
        ):

            status = "🟡 MOMENTUM ONLY"

            confidence = "LOW"


        else:

            status = "⚪ WEAK"

            confidence = "LOW"


        # =====================================================
        # NARRATIVE SCORE
        # =====================================================

        momentum_component = min(
            max(
                avg_momentum,
                0
            ),
            10
        ) * 2.5


        breadth_component = (
            breadth * 0.25
        )


        rs_component = min(
            max(
                avg_rs,
                0
            ),
            10
        ) * 2


        volume_component = min(
            max(
                avg_volume - 1,
                0
            ),
            2
        ) * 5


        narrative_score = (
            momentum_component
            + breadth_component
            + rs_component
            + volume_component
        )


        # =====================================================
        # RADAR SCORE
        # =====================================================

        radar_score = (
            narrative_score
            +
            safe_float(
                leader["Score"],
                0
            ) * 0.25
        )


        narrative_rows.append({

            "Narrative": narrative,

            "Status": status,

            "Confidence": confidence,

            "Coins": available_count,

            "Breadth %": round(
                breadth,
                1
            ),

            "Avg 24H %": round(
                avg_momentum,
                2
            ),

            "Avg RS": round(
                avg_rs,
                2
            ),

            "Avg Volume": round(
                avg_volume,
                2
            ),

            "Leader": leader["Symbol"],

            "Leader Score": round(
                safe_float(
                    leader["Score"],
                    0
                ),
                1
            ),

            "Confirmations": (
                f"{confirmations}/4"
            ),

            "Narrative Score": round(
                narrative_score,
                1
            ),

            "Radar Score": round(
                radar_score,
                1
            )

        })


    narrative_df = pd.DataFrame(
        narrative_rows
    )


    if not narrative_df.empty:

        narrative_df = (
            narrative_df
            .sort_values(
                "Radar Score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # 🎯 TRADE RADAR
    # =========================================================

    st.divider()

    st.header(
        "🎯 TRADE RADAR"
    )

    st.caption(
        "সবচেয়ে শক্তিশালী narrative ও leader এখানে দেখাবে।"
    )


    if not narrative_df.empty:

        radar_df = narrative_df[
            narrative_df["Status"].isin(
                [
                    "🔥 STRONG",
                    "🟢 DEVELOPING"
                ]
            )
        ].copy()


        if not radar_df.empty:

            st.dataframe(

                radar_df[
                    [
                        "Narrative",
                        "Status",
                        "Confidence",
                        "Leader",
                        "Leader Score",
                        "Breadth %",
                        "Avg 24H %",
                        "Avg RS",
                        "Avg Volume",
                        "Confirmations",
                        "Radar Score"
                    ]
                ].head(10),

                use_container_width=True,

                hide_index=True

            )

        else:

            st.warning(
                "এই মুহূর্তে STRONG/DEVELOPING "
                "Radar setup পাওয়া যায়নি।"
            )

            st.dataframe(

                narrative_df.head(5),

                use_container_width=True,

                hide_index=True

            )

    else:

        st.warning(
            "Narrative data এখনো তৈরি হয়নি।"
        )


    # =========================================================
    # 🔥 TOP TRADING NARRATIVES
    # =========================================================

    st.divider()

    st.subheader(
        "🔥 Top Trading Narratives"
    )


    if not narrative_df.empty:

        st.dataframe(

            narrative_df.head(
                TOP_NARRATIVES
            ),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "কোনো qualifying narrative পাওয়া যায়নি।"
        )


    # =========================================================
    # 🏆 CONFIRMED NARRATIVES
    # =========================================================

    st.divider()

    st.subheader(
        "🏆 Confirmed Narratives"
    )


    if not narrative_df.empty:

        confirmed = narrative_df[
            narrative_df["Status"]
            == "🔥 STRONG"
        ].copy()


        if not confirmed.empty:

            st.dataframe(

                confirmed[
                    [
                        "Narrative",
                        "Status",
                        "Confidence",
                        "Leader",
                        "Leader Score",
                        "Breadth %",
                        "Avg 24H %",
                        "Avg RS",
                        "Avg Volume",
                        "Confirmations",
                        "Narrative Score"
                    ]
                ],

                use_container_width=True,

                hide_index=True

            )

        else:

            st.info(
                "এই মুহূর্তে fully confirmed "
                "narrative পাওয়া যায়নি।"
            )


    # =========================================================
    # 🏆 NARRATIVE LEADERS
    # =========================================================

    st.divider()

    st.subheader(
        "🏆 Narrative Leaders"
    )


    if not narrative_df.empty:

        leaders = narrative_df[
            [
                "Narrative",
                "Status",
                "Leader",
                "Leader Score",
                "Breadth %",
                "Avg 24H %",
                "Avg RS",
                "Avg Volume",
                "Confirmations",
                "Radar Score"
            ]
        ]


        st.dataframe(

            leaders.head(
                TOP_NARRATIVES
            ),

            use_container_width=True,

            hide_index=True

        )


    # =========================================================
    # 🚀 SMALL-CAP MOMENTUM
    # =========================================================

    st.divider()

    st.header(
        "🚀 Small-Cap Momentum"
    )

    st.caption(
        "শুধু $10M–$500M market-cap coin এখানে আসবে।"
    )


    small_caps = scanner_df[
        (
            scanner_df["Market Cap"]
            >= SMALL_CAP_MIN
        )
        &
        (
            scanner_df["Market Cap"]
            <= SMALL_CAP_MAX
        )
        &
        (
            scanner_df["24H %"]
            >= MIN_MOMENTUM
        )
        &
        (
            scanner_df["Data Quality"]
            == "OK"
        )
    ].copy()


    # =========================================================
    # 🔥 STRONG SMALL-CAP
    # =========================================================

    strong_small = small_caps[
        (
            small_caps["RS vs BTC"]
            > 0
        )
        &
        (
            small_caps["Volume Ratio"]
            >= MIN_VOLUME_RATIO
        )
        &
        (
            small_caps["Score"]
            >= 55
        )
    ].copy()


    strong_small = strong_small.sort_values(

        [
            "Score",
            "RS vs BTC",
            "Volume Ratio"
        ],

        ascending=False

    )


    st.subheader(
        "🔥 Strong Small-Cap"
    )


    if not strong_small.empty:

        st.dataframe(

            strong_small[
                [
                    "Symbol",
                    "Market Cap",
                    "24H %",
                    "7D %",
                    "RS vs BTC",
                    "Volume Ratio",
                    "24H Volume",
                    "Score",
                    "Status"
                ]
            ].head(
                TOP_SMALL_CAP
            ),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "এই মুহূর্তে Strong Small-Cap পাওয়া যায়নি।"
        )


    # =========================================================
    # 👀 SMALL-CAP WATCH
    # =========================================================

    small_watch = small_caps[
        ~small_caps["Symbol"].isin(
            strong_small["Symbol"]
        )
    ].copy()


    small_watch = small_watch.sort_values(

        [
            "Score",
            "24H %",
            "Volume Ratio"
        ],

        ascending=False

    )


    st.subheader(
        "👀 Small-Cap Watch"
    )


    if not small_watch.empty:

        st.dataframe(

            small_watch[
                [
                    "Symbol",
                    "Market Cap",
                    "24H %",
                    "7D %",
                    "RS vs BTC",
                    "Volume Ratio",
                    "24H Volume",
                    "Score",
                    "Status"
                ]
            ].head(
                TOP_SMALL_CAP
            ),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "এই মুহূর্তে Small-Cap Watch-এ কোনো coin নেই।"
        )


    # =========================================================
    # 🏆 SMALL-CAP LEADERS
    # =========================================================

    st.divider()

    st.header(
        "🏆 Small-Cap Leaders"
    )

    st.caption(
        "ছোট market-cap coin-এর মধ্যে momentum, relative strength, "
        "volume এবং score অনুযায়ী সেরা leaders।"
    )


    small_leaders = scanner_df[
        (
            scanner_df["Market Cap"]
            >= SMALL_CAP_MIN
        )
        &
        (
            scanner_df["Market Cap"]
            <= SMALL_CAP_MAX
        )
        &
        (
            scanner_df["24H %"]
            > 0
        )
        &
        (
            scanner_df["RS vs BTC"]
            > 0
        )
        &
        (
            scanner_df["Data Quality"]
            == "OK"
        )
    ].copy()


    if not small_leaders.empty:

        # =====================================================
        # LEADER SCORE
        # =====================================================

        small_leaders["Leader Score"] = (

            small_leaders["Score"]
            * 0.50

            +

            small_leaders["RS vs BTC"].clip(
                lower=0,
                upper=10
            )
            * 2.0

            +

            small_leaders["24H %"].clip(
                lower=0,
                upper=10
            )
            * 1.5

            +

            small_leaders["Volume Ratio"].clip(
                lower=0,
                upper=3
            )
            * 5.0

        )


        small_leaders["Leader Score"] = (

            small_leaders["Leader Score"]
            .clip(
                lower=0,
                upper=100
            )
            .round(1)

        )


        # =====================================================
        # LEADER STATUS
        # =====================================================

        def get_small_cap_status(
            row
        ):

            # -------------------------------------------------
            # STRONG LEADER
            # -------------------------------------------------

            if (

                row["Leader Score"]
                >= STRONG_LEADER_SCORE

                and

                row["Score"]
                >= STRONG_LEADER_COIN_SCORE

                and

                row["RS vs BTC"]
                > 0

                and

                row["Volume Ratio"]
                >= STRONG_LEADER_VOLUME

                and

                row["24H %"]
                >= STRONG_LEADER_MOMENTUM

            ):

                return (
                    "🔥 STRONG LEADER"
                )


            # -------------------------------------------------
            # DEVELOPING LEADER
            # -------------------------------------------------

            elif (

                row["Leader Score"]
                >= DEVELOPING_LEADER_SCORE

                and

                row["Score"]
                >= DEVELOPING_LEADER_COIN_SCORE

                and

                row["RS vs BTC"]
                > 0

                and

                row["Volume Ratio"]
                >= DEVELOPING_LEADER_VOLUME

                and

                row["24H %"]
                >= DEVELOPING_LEADER_MOMENTUM

            ):

                return (
                    "🟢 DEVELOPING LEADER"
                )


            # -------------------------------------------------
            # MOMENTUM
            # -------------------------------------------------

            else:

                return (
                    "🟡 MOMENTUM"
                )


        small_leaders["Leader Status"] = (

            small_leaders.apply(
                get_small_cap_status,
                axis=1
            )

        )


        # =====================================================
        # SORT
        # =====================================================

        small_leaders = small_leaders.sort_values(

            [
                "Leader Score",
                "Score",
                "RS vs BTC",
                "Volume Ratio"
            ],

            ascending=False

        )


        # =====================================================
        # DISPLAY
        # =====================================================

        st.dataframe(

            small_leaders[
                [
                    "Symbol",
                    "Market Cap",
                    "24H %",
                    "7D %",
                    "RS vs BTC",
                    "Volume Ratio",
                    "24H Volume",
                    "Score",
                    "Leader Score",
                    "Leader Status"
                ]
            ].head(
                TOP_SMALL_CAP
            ),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "এই মুহূর্তে positive-RS Small-Cap Leader পাওয়া যায়নি."
        )


    # =========================================================
    # ⭐ A-LEVEL COINS
    # =========================================================

    st.divider()

    st.subheader(
        "⭐ A-Level Coin Setups"
    )


    a_level = scanner_df[
        (
            scanner_df["Status"]
            == "🔥 A-Level Candidate"
        )
        &
        (
            scanner_df["Data Quality"]
            == "OK"
        )
        &
        (
            scanner_df["Market Cap"].notna()
        )
    ].sort_values(

        "Score",

        ascending=False

    )


    if not a_level.empty:

        st.dataframe(

            a_level[
                [
                    "Symbol",
                    "Market Cap",
                    "24H %",
                    "7D %",
                    "RS vs BTC",
                    "Volume Ratio",
                    "24H Volume",
                    "Score",
                    "Cap Class"
                ]
            ].head(
                TOP_COINS
            ),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "এই মুহূর্তে A-Level setup নেই।"
        )


    # =========================================================
    # 🟢 WATCHLIST
    # =========================================================

    st.divider()

    st.subheader(
        "🟢 Watchlist"
    )


    watchlist = scanner_df[
        (
            scanner_df["Status"]
            == "🟢 Watchlist"
        )
        &
        (
            scanner_df["Data Quality"]
            == "OK"
        )
    ].sort_values(

        "Score",

        ascending=False

    )


    if not watchlist.empty:

        st.dataframe(

            watchlist[
                [
                    "Symbol",
                    "Market Cap",
                    "24H %",
                    "7D %",
                    "RS vs BTC",
                    "Volume Ratio",
                    "Score",
                    "Cap Class"
                ]
            ].head(
                TOP_COINS
            ),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.info(
            "Watchlist empty."
        )


    # =========================================================
    # 🧠 RADAR LOGIC
    # =========================================================

    st.divider()

    st.subheader(
        "🧠 Radar Logic"
    )

    st.write(
        "✅ Narrative breadth"
    )

    st.write(
        "✅ 24H momentum"
    )

    st.write(
        "✅ 7D Relative Strength vs BTC"
    )

    st.write(
        "✅ Volume expansion"
    )

    st.write(
        "✅ Leader strength"
    )

    st.write(
        "✅ 4-point narrative confirmation"
    )

    st.write(
        "✅ Market-cap classification"
    )

    st.write(
        "✅ Small-cap momentum detection"
    )

    st.write(
        "✅ Data-quality protection"
    )

    st.write(
        "✅ Extreme-move protection"
    )

    st.caption(
        "Research and screening tool only — "
        "not a guaranteed buy/sell signal."
    )


    # =========================================================
    # VERSION
    # =========================================================

    st.caption(
        "Risal Trading Dashboard V7.0 • "
        "Stable optimized engine"
    )

# =========================================================
# START LIVE DASHBOARD
# =========================================================

live_dashboard()
