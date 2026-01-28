# governance_dashboard.py
import os
import sys
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.manager import DatabaseManager

st.set_page_config(layout="wide", page_title="Institutional Data Governance Dashboard")

st.title("🛡️ Institutional Data Governance Dashboard")
st.markdown("Monitoring data quality, provider reliability, and symbol eligibility.")

db = DatabaseManager()

# Sidebar: Filters
st.sidebar.header("Governance Filters")
lookback_days = st.sidebar.slider("Lookback Days", 1, 30, 7)

# 1. Summary Metrics
st.header("System Health Summary")
col1, col2, col3, col4 = st.columns(4)

# Get all quality records
quality_query = """
SELECT symbol, quality_score, validation_flags, provider, recorded_at
FROM data_quality
ORDER BY recorded_at DESC
"""
conn = db.get_connection()
df_quality = pd.read_sql_query(quality_query, conn)

if not df_quality.empty:
    avg_score = df_quality['quality_score'].mean()
    col1.metric("Avg Quality Score", f"{avg_score:.2f}")

    # Active Symbols from universe
    try:
        with open("configs/universe.json", "r") as f:
            univ = json.load(f)
            tickers = univ.get("active_tickers", [])
            col2.metric("Active Universe", f"{len(tickers)}")
    except:
        col2.metric("Active Universe", "Error")

    # Quarantine Count
    try:
        elig_df = pd.read_sql_query("SELECT count(*) as count FROM trading_eligibility WHERE tradable = 0", conn)
        q_count = elig_df['count'][0]
        col3.metric("Quarantined", f"{q_count}", delta=None, delta_color="inverse")
    except:
        col3.metric("Quarantined", "N/A")

    # Provider Success Rate (Estimating from Audit)
    audit_df = pd.read_sql_query("SELECT status, count(*) as count FROM ingestion_audit GROUP BY status", conn)
    if not audit_df.empty:
        success = audit_df[audit_df['status'] == 'SUCCESS']['count'].sum()
        total = audit_df['count'].sum()
        rate = (success / total) if total > 0 else 0
        col4.metric("Ingestion Success", f"{rate:.1%}")

st.markdown("---")

# 2. Data Quality Distribution
st.header("Data Quality & Quarantine Status")
c1, c2 = st.columns(2)

if not df_quality.empty:
    # Latest score per symbol
    latest_quality = df_quality.groupby('symbol').first().reset_index()
    fig_hist = px.histogram(latest_quality, x="quality_score", nbins=20,
                            title="Latest Data Quality Score Distribution",
                            color_discrete_sequence=['#00CC96'])
    c1.plotly_chart(fig_hist, use_container_width=True)

# Eligibility Table
try:
    elig_list = pd.read_sql_query("SELECT symbol, tradable, trade_restrictions, data_quality_score FROM trading_eligibility", conn)
    if not elig_list.empty:
        def format_state(row):
            try:
                # restrictions is json string in DB
                res = json.loads(row['trade_restrictions'])
                return res.get('state', 'UNKNOWN')
            except:
                return 'UNKNOWN'

        elig_list['status'] = elig_list.apply(format_state, axis=1)
        c2.subheader("Trading Eligibility Status")
        c2.dataframe(elig_list[['symbol', 'status', 'tradable', 'data_quality_score']], height=400)
except:
    c2.warning("Trading eligibility data not yet available. Run `python data_intelligence/quarantine_manager.py`.")

st.markdown("---")

# 3. Provider Reliability
st.header("Provider Metrics")
audit_query = """
SELECT provider, status, count(*) as count
FROM ingestion_audit
GROUP BY provider, status
"""
df_audit = pd.read_sql_query(audit_query, conn)

if not df_audit.empty:
    fig_prov = px.bar(df_audit, x="provider", y="count", color="status", barmode="group",
                      title="Ingestion Status by Provider",
                      color_discrete_map={"SUCCESS": "#00CC96", "FAILED": "#EF553B", "REJECTED": "#FECB52", "INVALID_DATA": "#AB63FA"})
    st.plotly_chart(fig_prov, use_container_width=True)

# 4. Recent Audit Logs
st.header("Recent Ingestion Audit")
recent_audit = pd.read_sql_query("SELECT * FROM ingestion_audit ORDER BY finished_at DESC LIMIT 50", conn)
st.dataframe(recent_audit, use_container_width=True)

st.markdown("---")
st.caption(f"Last Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
