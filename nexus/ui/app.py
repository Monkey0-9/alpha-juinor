import streamlit as st
import pandas as pd
import httpx
import time
from datetime import datetime
import os

# --- Configuration ---
st.set_page_config(
    page_title="Nexus Institutional Matrix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #050505;
        color: #e0e0e0;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #050505);
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .metric-label {
        color: #888;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .status-active {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        background: rgba(0, 255, 135, 0.1);
        color: #00ff87;
        border: 1px solid #00ff87;
    }

    h1, h2, h3 { color: #fff !important; font-weight: 700 !important; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = os.getenv("NEXUS_BACKEND_URL", "http://localhost:8001")
API_KEY = os.getenv("NEXUS_API_KEY", "")

def fetch_data(endpoint: str):
    """Synchronous HTTP fetch with API Key authentication."""
    try:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        with httpx.Client(timeout=10, headers=headers) as client:
            resp = client.get(f"{BACKEND_URL}/{endpoint}")
            return resp.json()
    except Exception:
        return None

def main():
    # --- Header ---
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown("# ⚡ NEXUS <span style='color:#00ff87'>MATRIX</span>", unsafe_allow_html=True)
        st.markdown("<p style='color:#888; margin-top:-15px;'>Institutional Edition v2.0 | 24/7 Autonomous</p>", unsafe_allow_html=True)
    with col_h2:
        st.markdown("<div style='text-align:right; margin-top:20px;'><span class='status-active'>SYSTEM ONLINE</span></div>", unsafe_allow_html=True)

    # --- Account Metrics ---
    data_account = fetch_data("api/alpaca/account")

    if data_account:
        equity = float(data_account.get("portfolio_value", 0))
        buying_power = float(data_account.get("buying_power", 0))
        cash = float(data_account.get("cash", 0))
        mode = "SIMULATED" if data_account.get("simulated") else "LIVE PAPER"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Portfolio Equity</div><div class='metric-value'>${equity:,.2f}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Buying Power</div><div class='metric-value'>${buying_power:,.2f}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Cash Balance</div><div class='metric-value'>${cash:,.2f}</div></div>", unsafe_allow_html=True)
        with c4:
            mode_color = "#00ff87" if mode == "LIVE PAPER" else "#ffd60a"
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Mode</div><div class='metric-value' style='color:{mode_color}'>{mode}</div></div>", unsafe_allow_html=True)
    else:
        st.warning("Waiting for backend connection...")

    # --- Market Intelligence (LIVE BRAIN DATA) ---
    st.markdown("### 🧠 Market Intelligence Matrix")
    brain_data = fetch_data("api/monitor/brain")
    analysis = brain_data.get("analysis", {}) if brain_data else {}

    i1, i2, i3 = st.columns(3)
    with i1:
        regime = analysis.get("regime", "UNCERTAIN")
        forecast = analysis.get("market_forecast", "NEUTRAL")
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("**MARKET REGIME & FORECAST**")
        st.subheader(f"{regime} / {forecast}")
        st.caption(f"Strategy: {analysis.get('selected_strategy', 'N/A')}")
        st.markdown("</div>", unsafe_allow_html=True)
    with i2:
        agreement = analysis.get("strategy_agreement", 0.5)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("**STRATEGY ENSEMBLE AGREEMENT**")
        st.subheader(f"{agreement:.0%}")
        st.progress(agreement)
        st.caption("Consensus across 14 quant strategies")
        st.markdown("</div>", unsafe_allow_html=True)
    with i3:
        sentiment = analysis.get("market_sentiment", 0.5)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("**QUANTITATIVE SENTIMENT**")
        st.subheader(f"{sentiment:.2f}")
        sentiment_color = "#00ff87" if sentiment > 0.5 else "#ff4b4b"
        st.markdown(f"<p style='color:{sentiment_color}'>{'Bullish Bias' if sentiment > 0.5 else 'Bearish Bias'}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Holdings Table ---
    st.markdown("### 📊 Institutional Holdings")
    positions_data = fetch_data("api/alpaca/positions")
    pos_list = positions_data.get("positions", []) if positions_data else []

    if pos_list:
        pos_df = pd.DataFrame(pos_list)
        cols = ["symbol", "qty", "avg_price", "current_price", "market_value", "unrealized_plpc", "side"]
        for col in cols:
            if col not in pos_df.columns: pos_df[col] = 0.0

        display = pos_df[cols].copy()
        display.columns = ["Symbol", "Shares", "Avg Price", "Market Price", "Value", "P&L %", "Side"]
        st.dataframe(display.style.format({"Avg Price": "${:,.2f}", "Market Price": "${:,.2f}", "Value": "${:,.2f}", "P&L %": "{:,.2f}%"}), use_container_width=True, height=350)
    else:
        st.info("No active holdings.")

    # --- Recent Orders ---
    st.markdown("### 📋 Recent Orders")
    orders_data = fetch_data("api/alpaca/orders?status=all&limit=10")
    if orders_data:
        order_list = orders_data.get("orders", [])
        if order_list:
            st.dataframe(pd.DataFrame(order_list)[["symbol", "side", "qty", "type", "status"]], use_container_width=True)
        else:
            st.info("No recent orders.")

    # --- Footer ---
    st.markdown("---")
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    with fc2:
        if st.button("Force Refresh"): st.rerun()

    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()
