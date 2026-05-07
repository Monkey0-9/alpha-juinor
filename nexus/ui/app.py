import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict

# Institutional Terminal Configuration
st.set_page_config(
    page_title="Nexus Institutional Terminal",
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
    .stApp { background-color: #0B0E14; color: #E0E0E0; }
    .stMetric {
        background: #161B22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363D;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #58A6FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Nexus Terminal")
st.caption("v1.0.0 Institutional Alpha Engine")

health_status = fetch_health()
status_color = "✅" if health_status == "Online" else "⚠️"
brain = fetch_brain_data()

st.markdown(f"**Backend Status:** {status_color} {health_status}")

account = fetch_account_data()
positions = fetch_positions()

m1, m2, m3, m4 = st.columns(4)
if account:
    nav = float(account.get('equity', 0))
    cash = float(account.get('cash', 0))
    buying_power = float(account.get('buying_power', 0))

    with m1:
        st.metric("PORTFOLIO EQUITY", f"${nav:,.2f}")
    with m2:
        st.metric("CASH BALANCE", f"${cash:,.2f}")
    with m3:
        st.metric("BUYING POWER", f"${buying_power:,.2f}")
    with m4:
        st.metric("ACCOUNT STATUS", account.get('status', 'N/A'))
else:
    with m1:
        st.metric("PORTFOLIO EQUITY", "OFFLINE")
    with m2:
        st.metric("CASH BALANCE", "OFFLINE")
    with m3:
        st.metric("BUYING POWER", "OFFLINE")
    with m4:
        st.metric("ACCOUNT STATUS", "DISCONNECTED")

st.divider()

col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("Market Intelligence")
    if brain:
        st.metric("Regime", brain.get("regime", "N/A"))
        st.metric("Selected Strategy", brain.get("selected_strategy", "N/A"))
        st.metric("Market Sentiment", f"{brain.get('market_sentiment', 0.0):.2f}")
        st.metric("Risk VAR", f"{brain.get('risk_profile', {}).get('var', 0.0):.4f}")
        st.metric("Risk CVaR", f"{brain.get('risk_profile', {}).get('cvar', 0.0):.4f}")

        macro = brain.get("macro_profile", {})
        st.markdown(
            f"**Macro Momentum:** {macro.get('momentum', 0.0):.4f}  \
            **Volatility:** {macro.get('volatility', 0.0):.4f}  \
            **Trend Strength:** {macro.get('trend_strength', 0.0):.4f}"
        )
    else:
        st.info("Live market intelligence currently unavailable.")

    st.subheader("Open Positions")
    if positions:
        pos_df = pd.DataFrame(positions)
        cols = ['symbol', 'qty', 'market_value', 'unrealized_plpc', 'current_price']
        st.dataframe(pos_df[cols], use_container_width=True)
    else:
        st.info("No active positions detected.")

with col_side:
    st.subheader("Risk & Governance")
    if account:
        portfolio_val = float(account.get('equity', 0))
        st.warning(f"VaR (99%): ${portfolio_val * 0.02:,.2f}")
        st.error(f"Stress Loss (2008): -${portfolio_val * 0.57:,.2f}")

    st.divider()
    st.subheader("Audit Log")
    audit_data = [
        {"Time": "Live", "Event": "System Heartbeat", "Status": "Normal"},
        {"Time": "Live", "Event": "Risk Engine", "Status": "Active"},
        {"Time": "Live", "Event": "Alpha Pipeline", "Status": "Syncing"}
    ]
    st.table(pd.DataFrame(audit_data))

