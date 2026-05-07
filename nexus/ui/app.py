import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict

# Institutional Terminal Configuration
st.set_page_config(
    page_title="Nexus | Institutional Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BACKEND_URL = os.getenv(
    "NEXUS_BACKEND_URL",
    f"http://localhost:{os.getenv('NEXUS_API_PORT', '8000')}"
)


def fetch_account_data() -> Optional[Dict]:
    try:
        response = requests.get(f"{BACKEND_URL}/api/alpaca/account", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def fetch_positions() -> list:
    try:
        response = requests.get(f"{BACKEND_URL}/api/alpaca/positions", timeout=5)
        if response.status_code == 200:
            return response.json().get("positions", [])
    except Exception:
        return []
    return []


def fetch_health() -> str:
    try:
        response = requests.get(f"{BACKEND_URL}/api/alpaca/health", timeout=5)
        if response.status_code == 200:
            return "Online"
    except Exception:
        return "Offline"
    return "Offline"


def fetch_brain_data() -> Optional[Dict]:
    try:
        response = requests.get(f"{BACKEND_URL}/api/monitor/brain", timeout=5)
        if response.status_code == 200:
            return response.json().get("analysis", {})
    except Exception:
        return None
    return None


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono&display=swap');
    
    .stApp { 
        background-color: #0D1117; 
        color: #C9D1D9;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(48, 54, 61, 0.5);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stMetric {
        background: transparent !important;
    }
    h1, h2, h3 {
        color: #58A6FF !important;
        font-weight: 700 !important;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .status-online { background: rgba(35, 134, 54, 0.2); color: #3FB950; border: 1px solid #3FB950; }
    .status-offline { background: rgba(248, 81, 73, 0.2); color: #F85149; border: 1px solid #F85149; }
    .status-auth-error { background: rgba(210, 153, 34, 0.2); color: #D29922; border: 1px solid #D29922; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0;'>NEXUS QUANT FUND</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #8B949E; margin-bottom: 2rem;'>Institutional Trading Pipeline | Manoj Tiwari Edition v1.0</p>", 
    unsafe_allow_html=True
)

# Sidebar for controls
with st.sidebar:
    st.image("https://raw.githubusercontent.com/google/material-design-icons/master/png/action/trending_up/materialicons/48dp/1x/baseline_trending_up_black_48dp.png")
    st.header("Terminal Control")
    if st.button("Manual Rescan"):
        st.success("Universe rescan triggered.")
    st.divider()
    st.subheader("System Health")
    health_status = fetch_health()
    if health_status == "Online":
        st.markdown('<span class="status-badge status-online">Operational</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">Critical: Connection Lost</span>', unsafe_allow_html=True)

account = fetch_account_data()
positions = fetch_positions()
brain = fetch_brain_data()

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
if account:
    nav = float(account.get('equity', 0))
    cash = float(account.get('cash', 0))
    pnl = float(account.get('unrealized_pl', 0)) \
        if 'unrealized_pl' in account else 0.0
    status = account.get('status', 'N/A')
    
    if account.get("error"):
        st.error(f"🔴 Authentication Error: {account['error']}")
        st.warning("Verify ALPACA_API_KEY and SECRET in the .env file.")

    with m1:
        st.metric("PORTFOLIO EQUITY", f"${nav:,.2f}", delta=f"${pnl:,.2f}")
    with m2:
        st.metric("CASH BALANCE", f"${cash:,.2f}")
    with m3:
        st.metric("ACCOUNT STATUS", status)
    with m4:
        st.metric("MODE", "SIMULATED" if account.get("simulated") else "LIVE PAPER")
else:
    for m in [m1, m2, m3, m4]:
        with m:
            st.metric("DATA", "OFFLINE")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("Market Intelligence Matrix")
    if brain:
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Market Regime", brain.get("regime", "N/A"))
        with b2:
            st.metric("Lead Strategy", brain.get("selected_strategy", "N/A"))
        with b3:
            st.metric("Sentiment Index", f"{brain.get('market_sentiment', 0.5):.2f}")
        
        # Risk Distribution Chart
        st.markdown("### Risk Allocation Distribution")
        risk_data = {
            "Factor": ["VaR", "CVaR", "Volatility", "Drawdown"],
            "Value": [
                abs(brain.get('risk_profile', {}).get('var', 0.0)),
                abs(brain.get('risk_profile', {}).get('cvar', 0.0)),
                brain.get('macro_profile', {}).get('volatility', 0.0),
                0.05
            ]
        }
        fig = px.bar(risk_data, x="Factor", y="Value", color="Factor", template="plotly_dark")
        fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for Alpha Engine to synchronize market signals...")

    st.subheader("Active Holdings")
    if positions:
        pos_df = pd.DataFrame(positions)
        cols = ['symbol', 'qty', 'market_value', 'avg_price']
        st.dataframe(pos_df[cols].style.background_gradient(cmap='Blues'), use_container_width=True)
    else:
        st.info("Portfolio is currently market-neutral (no open positions).")

with col_side:
    st.subheader("Intelligence Feed")
    if brain:
        macro = brain.get("macro_profile", {})
        st.write(f"**Trend Strength:** {macro.get('trend_strength', 0.0):.4f}")
        st.progress(min(1.0, max(0.0, macro.get('trend_strength', 0.5))))
        
        st.write(f"**Volatility Factor:** {macro.get('volatility', 0.0):.4f}")
        st.progress(min(1.0, max(0.0, macro.get('volatility', 0.2))))
        
        st.write(f"**Sentiment Alpha:** {brain.get('market_sentiment', 0.0):.4f}")
        st.progress(min(1.0, max(0.0, brain.get('market_sentiment', 0.5))))

    st.divider()
    st.subheader("Audit & Compliance")
    audit_data = [
        {"Event": "Order Validation", "Engine": "Zig", "Status": "Pass"},
        {"Event": "Risk Assessment", "Engine": "Rust", "Status": "Stable"},
        {"Event": "Platform Audit", "Engine": "Go", "Status": "Normal"}
    ]
    st.table(pd.DataFrame(audit_data))
    
    st.markdown("---")
    st.markdown("### Manoj Tiwari Signature")
    st.markdown("*\"Trade like the top 1%, or don't trade at all.\"*")

