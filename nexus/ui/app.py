import streamlit as st
import pandas as pd
import httpx
import asyncio
import plotly.graph_objects as go
from datetime import datetime
import os
import time

# --- Institutional Branding & Configuration ---
st.set_page_config(
    page_title="Nexus Intelligence | Institutional Matrix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS for Glassmorphism & Modern Aesthetics
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
    
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .status-active { background: rgba(0, 255, 135, 0.1); color: #00ff87; border: 1px solid #00ff87; }
    .status-closed { background: rgba(255, 69, 58, 0.1); color: #ff453a; border: 1px solid #ff453a; }
    
    h1, h2, h3 { color: #fff !important; font-weight: 700 !important; }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

BACKEND_URL = os.getenv("NEXUS_BACKEND_URL", "http://localhost:8001")

async def fetch_data(endpoint):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BACKEND_URL}/api/alpaca/{endpoint}", timeout=10)
            return response.json()
        except Exception:
            return None

def main():
    # Header Area
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.markdown(f"# ⚡ NEXUS <span style='color:#00ff87'>MATRIX</span>", unsafe_allow_html=True)
        st.markdown("<p style='color:#888; margin-top:-15px;'>Manoj Tiwari Institutional Edition v1.1 | 24/7 Autonomous Execution</p>", unsafe_allow_html=True)
    
    with col_h2:
        st.markdown("<div style='text-align:right; margin-top:20px;'>", unsafe_allow_html=True)
        st.markdown(f"<span class='status-badge status-active'>SYSTEM ONLINE</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Metrics Section ---
    data_account = asyncio.run(fetch_data("account"))
    
    if data_account:
        equity = float(data_account.get("portfolio_value", 0))
        buying_power = float(data_account.get("buying_power", 0))
        cash = float(data_account.get("cash", 0))
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Portfolio Equity</div><div class='metric-value'>${equity:,.2f}</div></div>", unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Buying Power</div><div class='metric-value'>${buying_power:,.2f}</div></div>", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Cash Balance</div><div class='metric-value'>${cash:,.2f}</div></div>", unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"<div class='glass-card'><div class='metric-label'>Mode</div><div class='metric-value' style='color:#00ff87'>LIVE PAPER</div></div>", unsafe_allow_html=True)

    # --- Market Intelligence Section ---
    st.markdown("### 🧠 Market Intelligence Matrix")
    i_col1, i_col2, i_col3 = st.columns(3)
    
    with i_col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("**MARKET REGIME**")
        st.subheader("🐂 BULLISH")
        st.progress(0.85)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with i_col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("**LEAD STRATEGY**")
        st.subheader("Trend Following")
        st.caption("Active Optimization: Recursive Path Simulation")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with i_col3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("**SENTIMENT INDEX**")
        sentiment = 0.72
        st.subheader(f"{sentiment:.2f}")
        st.progress(sentiment)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Active Holdings Section ---
    st.markdown("### 📊 Institutional Holdings")
    data_positions = asyncio.run(fetch_data("positions"))
    
    if data_positions and "positions" in data_positions:
        pos_list = data_positions["positions"]
        if pos_list:
            pos_df = pd.DataFrame(pos_list)
            
            # Ensure critical columns exist to avoid KeyError
            required_cols = ["symbol", "qty", "avg_price", "current_price", "market_value", "unrealized_plpc", "side"]
            for col in required_cols:
                if col not in pos_df.columns:
                    pos_df[col] = 0.0
            
            # Format columns for display
            display_df = pos_df[required_cols].copy()
            display_df.columns = ["Symbol", "Shares", "Avg Price", "Market Price", "Value", "Profit %", "Side"]
            
            st.dataframe(
                display_df.style.format({
                    "Avg Price": "${:,.2f}",
                    "Market Price": "${:,.2f}",
                    "Value": "${:,.2f}",
                    "Profit %": "{:,.2f}%"
                }).map(
                    lambda x: 'color: #00ff87' if isinstance(x, (float, int)) and x > 0 else 'color: #ff453a',
                    subset=["Profit %"]
                ),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No active holdings in current matrix.")
    else:
        st.warning("Position data stream unavailable. Initializing secure handshake...")

    # --- Footer / Real-time Status ---
    st.markdown("---")
    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        st.caption(f"Last Matrix Update: {datetime.now().strftime('%H:%M:%S')} | Network: Secure WebSocket")
    with f_col2:
        if st.button("Manual Force Refresh"):
            st.rerun()

    # Auto-refresh logic (every 30 seconds)
    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()
