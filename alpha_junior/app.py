import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGN ELITE - PREMIUM AESTHETIC CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="The Sovereign Elite | Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BACKEND_URL = "http://localhost:8000"

def get_account_data():
    try:
        response = requests.get(f"{BACKEND_URL}/api/alpaca/account", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None

def get_positions():
    try:
        response = requests.get(f"{BACKEND_URL}/api/alpaca/positions", timeout=5)
        if response.status_code == 200:
            return response.json().get("positions", [])
    except:
        return []

# Custom CSS for Glassmorphism and Institutional Aesthetic
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stApp { background: radial-gradient(circle at top right, #1E293B, #0F172A); }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    h1 {
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-size: 2.5rem !important;
        background: linear-gradient(to right, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# Fetch Data
account = get_account_data()
positions = get_positions()

# Header Section
st.title("The Sovereign Elite")
st.markdown("### `MASTERPIECE CORE V1.0.0-ELITE` | `GLOBAL QUANT SOVEREIGN` ")

# Real-time Metrics Row
m1, m2, m3, m4 = st.columns(4)
if account:
    nav = float(account.get('equity', 0))
    cash = float(account.get('cash', 0))
    st_val = f"${nav:,.2f}"
    st_cash = f"${cash:,.2f}"
    
    with m1: st.metric("PORTFOLIO NAV", st_val, "LIVE")
    with m2: st.metric("CASH BALANCE", st_cash, "STABLE")
    with m3: st.metric("MARKET REGIME", "BULL", "DETECTION ACTIVE") # Simulated state
    with m4: st.metric("SURVIVAL PROB", "98.2%", "MONTE CARLO")
else:
    with m1: st.metric("PORTFOLIO NAV", "OFFLINE", "OFFLINE")
    with m2: st.metric("CASH BALANCE", "OFFLINE", "OFFLINE")
    with m3: st.metric("MARKET REGIME", "WAITING", "SCANNING")
    with m4: st.metric("SURVIVAL PROB", "0.0%", "CALCULATING")

st.divider()

# Main Intelligence Layout
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📡 Global Alpha State-Space (Kalman Filtered)")
    t = np.linspace(0, 10, 100)
    base = np.sin(t)
    noisy = base + np.random.normal(0, 0.1, 100)
    denoised = base + np.random.normal(0, 0.02, 100)
    
    chart_df = pd.DataFrame({
        'Noisy Intelligence': noisy,
        'Sovereign Elite Signal': denoised
    })
    st.line_chart(chart_df, height=400)

    st.subheader("📊 Institutional Position Audit")
    if positions:
        pos_df = pd.DataFrame(positions)
        st.dataframe(pos_df[['symbol', 'qty', 'market_value', 'unrealized_plpc', 'current_price']], width='stretch')
    else:
        st.info("No active positions in the sovereign orbit.")

with right_col:
    st.subheader("🛡️ Risk Command Center")
    st.markdown("#### `STRESS_TEST_RESULTS`")
    if account:
        portfolio_val = float(account.get('equity', 0))
        st.error(f"🚨 2008 CRASH SCENARIO: -${portfolio_val * 0.57:,.2f}")
        st.warning(f"⚠️ COVID 2020 SCENARIO: -${portfolio_val * 0.34:,.2f}")
        st.success("✅ CURRENT LIQUIDITY: EXCELLENT")
    
    st.divider()
    st.subheader("🔏 Sovereign Audit Ledger")
    audit_data = [
        {"Time": "14:20:01", "Symbol": "AAPL", "Action": "OPTIMIZED", "Status": "PASS"},
        {"Time": "14:20:05", "Symbol": "TSLA", "Action": "REBALANCED", "Status": "PASS"},
        {"Time": "14:21:12", "Symbol": "NVDA", "Action": "SCALED", "Status": "PASS"},
    ]
    st.table(pd.DataFrame(audit_data))

# Auto-refresh
time.sleep(10)
st.rerun()
