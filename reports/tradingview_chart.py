import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_tradingview_style(df, ema50, ema200, vwap, donchian_upper, donchian_lower, atr):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ),
        row=1, col=1
    )

    # EMA
    fig.add_trace(go.Scatter(x=df.index, y=ema50, line=dict(color="blue"), name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema200, line=dict(color="red"), name="EMA 200"), row=1, col=1)

    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=vwap, line=dict(color="orange"), name="VWAP"), row=1, col=1)

    # Donchian Channels
    fig.add_trace(go.Scatter(x=df.index, y=donchian_upper, line=dict(color="green", width=1), name="Donchian Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=donchian_lower, line=dict(color="green", width=1), name="Donchian Lower"), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)

    # ATR
    fig.add_trace(go.Scatter(x=df.index, y=atr, line=dict(color="purple"), name="ATR"), row=3, col=1)

    fig.update_layout(
        title="Institutional TradingView-Style Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=900
    )

    fig.show()
