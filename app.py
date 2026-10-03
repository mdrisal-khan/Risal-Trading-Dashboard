from pathlib import Path

app_code = r'''
import streamlit as st
import pandas as pd
import requests
import feedparser
import re
import html
from datetime import datetime, timezone

st.set_page_config(
    page_title="Risal Trading Dashboard - Binance Spot Radar",
    page_icon="📡",
    layout="wide",
)

# ============================================================
# CONFIG
# ============================================================

BINANCE_BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

QUOTE = "USDT"
REQUEST_TIMEOUT = 12

# RSS sources. Catalyst classification is automated; verify important
# announcements at the original source before acting.
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "The Block": "https://www.theblock.co/rss.xml",
    "Binance Blog": "https://www.binance.com/en/blog/rss",
}

# ============================================================
# GENERAL HELPERS
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def binance_get(path, params=None):
    last_error = None
    for base in BINANCE_BASE_URLS:
        try:
            r = requests.get(
                base + path,
                params=params or {},
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "RisalTradingDashboard/1.0"},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Binance API unavailable: {last_error}")


def clean_symbol(symbol):
    if symbol.endswith(QUOTE):
        return symbol[:-len(QUOTE)]
    return symbol


def fmt_age(dt):
    if not dt:
        return "Unknown"
    now = datetime.now(timezone.utc)
    seconds = max(0, int((now - dt).total_seconds()))
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def strip_html(text):
    text = html.unescape(text or "")
    return re.sub(r"<[^>]+>", "", text).strip()


# ============================================================
# BINANCE MARKET DATA
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_usdt_symbols():
    data = binance_get("/api/v3/exchangeInfo")
    symbols = []
    for s in data.get("symbols", []):
        if (
            s.get("quoteAsset") == QUOTE
            and s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed", True)
        ):
            symbols.append(s["symbol"])
    return symbols


@st.cache_data(ttl=45, show_spinner=False)
def get_24h_tickers():
    return binance_get("/api/v3/ticker/24hr")


@st.cache_data(ttl=120, show_spinner=False)
def get_klines(symbol, interval="1h", limit=60):
    return binance_get(
        "/api/v3/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# 4H ACCUMULATION DETECTOR
# ============================================================

def detect_4h_accumulation(symbol):
    try:
        rows = get_klines(symbol, "4h", 40)

        # Need enough completed candles.
        if len(rows) < 25:
            return None

        # Ignore the currently forming candle.
        rows = rows[:-1]
        df = pd.DataFrame(
            rows,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base",
                "taker_quote", "ignore"
            ],
        )

        for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        recent = df.tail(8)
        prior = df.iloc[-20:-8]

        if len(recent) < 8 or len(prior) < 8:
            return None

        recent_high = recent["high"].max()
        recent_low = recent["low"].min()
        recent_range = (recent_high - recent_low) / max(recent["close"].mean(), 1e-12)

        prior_high = prior["high"].max()
        prior_low = prior["low"].min()
        prior_range = (prior_high - prior_low) / max(prior["close"].mean(), 1e-12)

        avg_recent_vol = recent["quote_volume"].mean()
        avg_prior_vol = prior["quote_volume"].mean()

        # Price should be relatively compressed while activity remains healthy.
        compression = recent_range < max(prior_range * 0.75, 0.025)
        volume_hold = avg_recent_vol >= avg_prior_vol * 0.75

        # Avoid calling a very large expansion candle "accumulation".
        last_close = recent["close"].iloc[-1]
        move_from_low = (last_close - recent_low) / max(recent_low, 1e-12)
        not_overextended = move_from_low < 0.15

        score = 0
        if compression:
            score += 45
        if volume_hold:
            score += 30
        if not_overextended:
            score += 25

        if score < 70:
            return None

        return {
            "Accumulation Score": score,
            "4H Range %": round(recent_range * 100, 2),
            "Vol Hold %": round((avg_recent_vol / max(avg_prior_vol, 1e-12)) * 100, 1),
        }

    except Exception:
        return None


# ============================================================
# 1H SUDDEN VOLUME EXPANSION
# ============================================================

def detect_1h_volume_expansion(symbol):
    try:
        rows = get_klines(symbol, "1h", 30)
        if len(rows) < 22:
            return None

        # Compare the latest completed 1H candle against the previous 20
        # completed candles. This avoids using a partially formed candle.
        rows = rows[:-1]

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base",
                "taker_quote", "ignore"
            ],
        )

        for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        latest = df.iloc[-1]
        baseline = df.iloc[-21:-1]["quote_volume"]

        baseline_median = baseline.median()
        if baseline_median <= 0:
            return None

        ratio = latest["quote_volume"] / baseline_median

        if ratio < 2:
            return None

        if ratio >= 5:
            level = "5x+"
        elif ratio >= 3:
            level = "3x+"
        else:
            level = "2x+"

        price_change = (
            (latest["close"] - latest["open"]) /
            max(abs(latest["open"]), 1e-12)
        ) * 100

        return {
            "Volume Ratio": round(ratio, 2),
            "Volume Level": level,
            "1H Price %": round(price_change, 2),
            "1H Quote Volume": round(float(latest["quote_volume"]), 2),
        }

    except Exception:
        return None


# ============================================================
# LIVE SPOT RADAR
# ============================================================

@st.cache_data(ttl=90, show_spinner=False)
def run_spot_radar(max_symbols=80):
    tickers = get_24h_tickers()
    ticker_map = {
        x["symbol"]: x for x in tickers
        if x.get("symbol", "").endswith(QUOTE)
    }

    # Prioritize liquid/active symbols so the live scan remains practical.
    candidates = []
    for symbol, t in ticker_map.items():
        qv = safe_float(t.get("quoteVolume"))
        if qv >= 1_000_000:
            candidates.append((symbol, qv))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = [x[0] for x in candidates[:max_symbols]]

    results = []

    progress = st.progress(0)
    for i, symbol in enumerate(candidates):
        acc = detect_4h_accumulation(symbol)
        vol = detect_1h_volume_expansion(symbol)

        if acc and vol:
            t = ticker_map.get(symbol, {})
            results.append({
                "Coin": clean_symbol(symbol),
                "Symbol": symbol,
                "24H %": round(safe_float(t.get("priceChangePercent")), 2),
                "24H Volume": round(safe_float(t.get("quoteVolume")), 0),
                "4H Accumulation": acc["Accumulation Score"],
                "4H Range %": acc["4H Range %"],
                "Vol Ratio": vol["Volume Ratio"],
                "Volume Power": vol["Volume Level"],
                "1H Price %": vol["1H Price %"],
            })

        progress.progress((i + 1) / max(len(candidates), 1))

    progress.empty()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values(
        ["Vol Ratio", "4H Accumulation"],
        ascending=[False, False]
    ).reset_index(drop=True)


# ============================================================
# CATALYST ENGINE
# ============================================================

VERY_HIGH_TERMS = [
    "institutional investment",
    "institutional",
    "bank partnership",
    "bank partners",
    "bank integration",
    "major bank",
    "investment firm",
    "asset manager",
    "blackrock",
    "fidelity",
    "jpmorgan",
    "goldman sachs",
    "visa",
    "mastercard",
    "paypal",
    "stripe",
    "government adoption",
    "central bank",
    "sovereign",
    "strategic investment",
    "strategic partnership",
    "acquires",
    "acquisition",
]

HIGH_TERMS = [
    "partnership",
    "integration",
    "adoption",
    "funding",
    "raises",
    "raised",
    "listing",
    "launches",
    "mainnet",
    "payment",
    "settlement",
    "enterprise",
    "major company",
]

MEDIUM_TERMS = [
    "upgrade",
    "testnet",
    "roadmap",
    "grant",
    "ecosystem",
    "collaboration",
    "staking",
]

CATEGORY_RULES = [
    ("Institution / Investment", [
        "institution", "investment", "asset manager", "funding",
        "raises", "raised", "capital"
    ]),
    ("Bank / Financial", [
        "bank", "jpmorgan", "goldman", "fidelity", "blackrock",
        "visa", "mastercard", "paypal", "stripe", "financial"
    ]),
    ("Strategic Partnership", [
        "partnership", "strategic", "collaboration", "integrates",
        "integration"
    ]),
    ("Adoption / Payments", [
        "adoption", "payment", "settlement", "merchant"
    ]),
    ("Exchange / Listing", [
        "listed", "listing", "binance", "coinbase", "kraken"
    ]),
    ("Technology / Upgrade", [
        "upgrade", "mainnet", "testnet", "protocol"
    ]),
    ("Regulation / Government", [
        "government", "regulation", "regulator", "central bank",
        "sovereign", "law"
    ]),
]


def classify_power(title):
    text = title.lower()

    if any(term in text for term in VERY_HIGH_TERMS):
        return "🔴 VERY HIGH", 4

    if any(term in text for term in HIGH_TERMS):
        return "🟠 HIGH", 3

    if any(term in text for term in MEDIUM_TERMS):
        return "🟡 MEDIUM", 2

    return "⚪ LOW", 1


def classify_category(title):
    text = title.lower()
    for category, terms in CATEGORY_RULES:
        if any(term in text for term in terms):
            return category
    return "General News"


def extract_coin(title, summary=""):
    text = f"{title} {summary}".upper()

    # Common $TICKER format.
    m = re.search(r"\$([A-Z][A-Z0-9]{1,9})\b", text)
    if m:
        return m.group(1)

    # A conservative list of well-known crypto assets/projects.
    known = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
        "LINK", "SUI", "TON", "DOT", "TRX", "ATOM", "NEAR", "APT",
        "ARB", "OP", "PEPE", "SHIB", "UNI", "AAVE", "MATIC",
        "POL", "INJ", "SEI", "TIA", "TAO", "RENDER", "FIL",
        "ALGO", "HBAR", "ETC", "LTC", "BCH", "MKR", "CRV",
        "JUP", "WIF", "BONK", "FET", "GRT", "IMX", "RUNE",
    ]

    for coin in known:
        if re.search(rf"\b{re.escape(coin)}\b", text):
            return coin

    return "Not identified"


def extract_partner(title, summary=""):
    text = strip_html(f"{title} {summary}")

    partner_patterns = [
        r"\b(?:with|partners? with|partnered with|in partnership with|joins forces with)\s+([A-Z][A-Za-z0-9&.\- ]{2,50})",
        r"\b([A-Z][A-Za-z0-9&.\- ]{2,40})\s+(?:partners with|partners|integrates with)\b",
    ]

    for pattern in partner_patterns:
        m = re.search(pattern, text)
        if m:
            value = m.group(1).strip(" .,:;-")
            if len(value.split()) <= 8:
                return value

    # Strong named institutions that can be safely detected from titles.
    institutions = [
        "BlackRock", "Fidelity", "JPMorgan", "Goldman Sachs",
        "Visa", "Mastercard", "PayPal", "Stripe", "Coinbase",
        "Binance", "Kraken", "Google", "Microsoft", "Amazon",
        "IBM", "SWIFT",
    ]
    for name in institutions:
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.I):
            return name

    return "Not identified"


def parse_entry_time(entry):
    try:
        if getattr(entry, "published_parsed", None):
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if getattr(entry, "updated_parsed", None):
            return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def get_catalysts():
    rows = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:40]:
                title = strip_html(entry.get("title", ""))
                summary = strip_html(entry.get("summary", entry.get("description", "")))

                if not title:
                    continue

                power, power_rank = classify_power(title)
                category = classify_category(title)
                coin = extract_coin(title, summary)
                partner = extract_partner(title, summary)
                published = parse_entry_time(entry)

                source_url = entry.get("link", "")

                rows.append({
                    "_rank": power_rank,
                    "_time": published or datetime(1970, 1, 1, tzinfo=timezone.utc),
                    "Power": power,
                    "Coin": coin,
                    "Partner / Institution": partner,
                    "News": title,
                    "Age": fmt_age(published),
                    "Category": category,
                    "Source": source_name,
                    "URL": source_url,
                })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "Power", "Coin", "Partner / Institution",
            "News", "Age", "Category", "Source", "URL"
        ])

    df = pd.DataFrame(rows)

    # Stronger catalyst first, then newest within the same power level.
    df = df.sort_values(
        ["_rank", "_time"],
        ascending=[False, False]
    )

    # Remove exact duplicate headlines.
    df = df.drop_duplicates(subset=["News"], keep="first")

    return df[
        [
            "Power", "Coin", "Partner / Institution",
            "News", "Age", "Category", "Source", "URL"
        ]
    ].reset_index(drop=True)


# ============================================================
# UI
# ============================================================

st.title("📡 Risal Trading Dashboard — Binance Spot Radar")
st.caption(
    "Scanner only — not an automatic entry signal. "
    "Manual confirmation remains: market structure → support/accumulation → "
    "volume power → breakout → 1H confirmation."
)

st.divider()

# -------------------- MARKET RADAR --------------------

st.subheader("🔥 Live Spot Radar")

col1, col2 = st.columns([1, 3])
with col1:
    max_symbols = st.selectbox(
        "Scan universe",
        [40, 60, 80, 120],
        index=2,
        help="Higher values scan more USDT spot pairs but can take longer."
    )

with col2:
    st.info(
        "4H: accumulation/compression heuristic + 1H: ≥2x volume expansion "
        "against the previous 20 completed 1H candles."
    )

if st.button("🔄 Run Live Scan", type="primary"):
    st.cache_data.clear()

try:
    radar = run_spot_radar(max_symbols=max_symbols)

    if radar.empty:
        st.warning(
            "No coin currently satisfies both filters. "
            "This is normal when the market is not producing the required setup."
        )
    else:
        st.dataframe(
            radar,
            use_container_width=True,
            hide_index=True,
        )

except Exception as e:
    st.error(f"Radar error: {e}")

st.divider()

# -------------------- CATALYST --------------------

st.subheader("🧠 Catalyst — Powerful News First")
st.caption(
    "Ranking is automated from headline language. "
    "VERY HIGH/HIGH means the headline contains stronger institutional, "
    "bank, strategic partnership, investment, adoption or infrastructure terms. "
    "Always verify important announcements at the original source."
)

try:
    catalysts = get_catalysts()

    if catalysts.empty:
        st.warning(
            "No catalyst feed could be loaded right now. "
            "The market scanner can still work independently."
        )
    else:
        display_cols = [
            "Power",
            "Coin",
            "Partner / Institution",
            "News",
            "Age",
            "Category",
            "Source",
        ]

        st.dataframe(
            catalysts[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "News": st.column_config.TextColumn(
                    "News",
                    width="large",
                ),
                "Partner / Institution": st.column_config.TextColumn(
                    "Partner / Institution",
                    width="medium",
                ),
                "Power": st.column_config.TextColumn(
                    "Power",
                    width="small",
                ),
                "Age": st.column_config.TextColumn(
                    "Age",
                    width="small",
                ),
            },
        )

        st.markdown("**Catalyst priority:** 🔴 VERY HIGH → 🟠 HIGH → 🟡 MEDIUM → ⚪ LOW")

except Exception as e:
    st.error(f"Catalyst error: {e}")

st.divider()

st.caption(
    "Important: Catalyst classification is rule-based and can misclassify headlines. "
    "A headline is not proof of a trade setup. Confirm the original announcement, "
    "liquidity, market structure and breakout conditions manually."
)
'''
