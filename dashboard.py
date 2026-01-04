# dashboard.py
import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from backtest.registry import BacktestRegistry

st.set_page_config(layout="wide", page_title="Mini Quant Fund — Backtest Registry")

registry = BacktestRegistry()
runs = registry.list_runs()

st.title("Mini Quant Fund — Backtest Registry")
st.markdown("Browse past backtests, inspect equity curves, drawdown and metrics. Use this as a PM / IC tool.")

if not runs:
    st.warning("No backtest runs found in `output/backtests/`. Run `python main.py` to generate runs.")
    st.stop()

# Sidebar controls
st.sidebar.header("Filters")
names = sorted(set([r.get("name", "run") for r in runs]))
sel_name = st.sidebar.multiselect("Strategy name", names, default=names)
date_min = st.sidebar.date_input("Start date (min)", value=pd.to_datetime(runs[-1]["timestamp"]).date())
date_max = st.sidebar.date_input("End date (max)", value=pd.to_datetime(runs[0]["timestamp"]).date())

# Filter runs
def run_in_window(run):
    t = pd.to_datetime(run["timestamp"]).date()
    return (run.get("name", "") in sel_name) and (date_min <= t <= date_max)

filtered = [r for r in runs if run_in_window(r)]

st.sidebar.markdown(f"**Runs matching:** {len(filtered)}")

# Show list of runs
st.header("Backtest Runs")
cols = st.columns([3, 2, 2, 2, 2])
cols[0].markdown("**Run ID / Name**")
cols[1].markdown("**Timestamp (UTC)**")
cols[2].markdown("**Final Equity**")
cols[3].markdown("**Sharpe**")
cols[4].markdown("**Max Drawdown**")

for r in filtered:
    c0, c1, c2, c3, c4 = st.columns([3, 2, 2, 2, 2])
    c0.write(f"{r['run_id'][:8]} — {r.get('name','')}")
    c1.write(r["timestamp"])
    metrics = r.get("metrics", {})
    c2.write(f"{metrics.get('final_equity', ''):,}" if metrics.get("final_equity") else "")
    c3.write(f"{metrics.get('sharpe',''):.2f}" if metrics.get("sharpe") is not None else "")
    c4.write(f"{metrics.get('max_drawdown',''):.2%}" if metrics.get("max_drawdown") is not None else "")

st.markdown("---")

# Select runs to inspect (support multi-select)
run_choices = {f"{r['run_id'][:8]} — {r.get('name','')}" : r["run_id"] for r in filtered}
sel_runs = st.multiselect("Select run(s) to inspect", list(run_choices.keys()), default=list(run_choices.keys())[:1])

if not sel_runs:
    st.info("Select a run to inspect metrics and charts.")
    st.stop()

selected_ids = [run_choices[k] for k in sel_runs]

# Load results
run_dfs = {}
for rid in selected_ids:
    try:
        data = registry.load_run(rid)
        meta = data["meta"]
        df = data["results"]
        run_dfs[rid] = {"meta": meta, "df": df}
    except Exception as e:
        st.error(f"Failed to load run {rid}: {e}")

# Plot equity curves
st.header("Equity Curves")
fig = go.Figure()
for rid, payload in run_dfs.items():
    df = payload["df"]
    # heuristic: try to pick equity/nav column
    if "equity" in df.columns:
        series = df["equity"]
    else:
        # try NAV-like columns
        cand = [c for c in df.columns if "nav" in c.lower() or "equity" in c.lower()]
        series = df[cand[0]] if cand else df.iloc[:,0]
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name=f"{payload['meta']['name']} ({rid[:8]})"))
fig.update_layout(height=400, template="plotly_white", xaxis_title="Date", yaxis_title="Equity")
st.plotly_chart(fig, use_container_width=True)

# Drawdown plot for the first selected run
st.header("Drawdown & Returns (selected run)")
first_id = selected_ids[0]
df0 = run_dfs[first_id]["df"]
if "equity" in df0.columns:
    series0 = df0["equity"]
else:
    cand = [c for c in df0.columns if "nav" in c.lower() or "equity" in c.lower()]
    series0 = df0[cand[0]] if cand else df0.iloc[:,0]

# compute drawdown
peak = series0.cummax()
drawdown = series0 / peak - 1
col1, col2 = st.columns(2)
col1.subheader("Drawdown")
fig_dd = px.line(x=drawdown.index, y=drawdown.values, labels={"x":"Date","y":"Drawdown"})
col1.plotly_chart(fig_dd, use_container_width=True)
col2.subheader("Daily Returns Distribution")
daily_ret = series0.pct_change(fill_method=None).dropna()
fig_hist = px.histogram(daily_ret, nbins=100, title="Daily returns distribution")
col2.plotly_chart(fig_hist, use_container_width=True)

# Show run metadata & metrics
st.header("Run Metadata / Metrics")
for rid, payload in run_dfs.items():
    st.subheader(f"{payload['meta']['name']} — {rid[:8]}")
    st.json(payload['meta'])

st.markdown("### Notes")
st.write("Use this dashboard to compare runs, inspect equity, and export run artifacts from `output/backtests/`.")
