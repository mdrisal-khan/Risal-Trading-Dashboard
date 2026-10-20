import streamlit as st
import pandas as pd
import requests

# Page Configuration
st.set_page_config(page_title="Elite Spot Trader Dashboard", page_icon="🚨", layout="wide")

st.title("🚨 World's Best Spot Trading Data Dashboard")
st.markdown("---")

# ----------------- SIDEBAR: RISK CALCULATOR -----------------
st.sidebar.header("🧮 Institutional Risk Calculator")
capital = st.sidebar.number_input("Total Capital ($)", min_value=100, value=10000, step=500)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5)
entry_price = st.sidebar.number_input("Entry Price ($)", min_value=0.00001, value=1.0, format="%.5f")
stop_loss = st.sidebar.number_input("Stop Loss Price ($)", min_value=0.00001, value=0.85, format="%.5f")

# Risk Logic
allowed_risk = capital * (risk_pct / 100)
if entry_price > stop_loss:
    loss_pct = (entry_price - stop_loss) / entry_price
    position_size = allowed_risk / loss_pct
    
    st.sidebar.success(f"Allowed Risk: ${allowed_risk:.2f}")
    st.sidebar.info(f"Max Position Size: ${position_size:.2f}")
    st.sidebar.warning(f"Stop Loss Distance: {loss_pct*100:.2f}%")
else:
    st.sidebar.error("Stop Loss must be lower than Entry Price!")

# ----------------- MAIN DASHBOARD: NARRATIVES & COINS -----------------
st.header("📊 Trending Narratives & Alpha Watchlist")

# Static Alpha Data (Institutionally Formatted)
data = {
    "Narrative Sector": ["Real World Assets (RWA)", "Artificial Intel. (AI)", "Layer 2 & Infra", "DePIN Network"],
    "Key Leader": ["ONDO", "FET", "ARB", "RENDER"],
    "Laggards (Watchlist)": ["PENDLE, OM, TRU", "AKT, TAO", "OP, METIS", "HNT, IOTX"],
    "Status": ["🔥 High Inflow", "📈 Accumulating", "⏳ Sideways", "🔥 High Inflow"]
}
df = pd.DataFrame(data)
st.table(df)

# ----------------- LIVE MARKET DATA FROM COINGECKO -----------------
st.header("⚡ Live Top 10 Crypto Market Caps (Real-time Flow)")

@st.cache_data(ttl=300) # Cache for 5 minutes to prevent rate limit
def fetch_live_data():
    url = "https://coingecko.com"
    response = requests.get(url).json()
    return pd.DataFrame(response)[['name', 'symbol', 'current_price', 'market_cap', 'total_volume', 'price_change_percentage_24h']]

try:
    live_df = fetch_live_data()
    # Format Columns
    live_df.columns = ['Coin Name', 'Ticker', 'Price ($)', 'Market Cap ($)', '24h Volume ($)', '24h Change (%)']
    st.dataframe(live_df.style.format({'Price ($)': '{:,.2f}', 'Market Cap ($)': '{:,.0f}', '24h Volume ($)': '{:,.0f}', '24h Change (%)': '{:+.2f}%'}), use_container_width=True)
except:
    st.error("CoinGecko API Limit reached. Refresh in a moment.")

# ----------------- IMPORTANT LIVE LINKS -----------------
st.subheader("🌐 Quick Institutional Data Links")
col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("🔗 DefiLlama Chain TVL Flow", "https://defillama.com")
with col2:
    st.link_button("🔗 CryptoRank AI Screener", "https://cryptorank.io")
with col3:
    st.link_button("🔗 Live Token Unlocks 
