"""
Binance Trading Bot — Real-Time Dashboard
===========================================
A Streamlit-powered monitoring dashboard with live charts,
indicator panels, trade history, and performance metrics.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import config
import importlib
importlib.reload(config) # Force reload to pick up changes in config.py
from core.data_engine import BinanceDataEngine
from core.indicators import calculate_all_indicators
from core.strategy import ConfluenceStrategy, SignalType
from core.llm_analyst import get_analyst
from core.utils import load_trade_journal, calculate_performance, format_price, format_pct

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 Binance Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;700&display=swap');

    .stApp {
        background-color: #050510;
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism containers */
    div[data-testid="stMetric"], .metric-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        border: 1px solid rgba(0, 230, 118, 0.3) !important;
        transform: translateY(-2px);
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(to right, #fff, #aaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 11px;
        color: #707090;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* Signal Animations */
    .signal-buy {
        color: #00ffa3;
        text-shadow: 0 0 15px rgba(0, 255, 163, 0.4);
        animation: pulse 2s infinite;
    }

    .signal-sell {
        color: #ff3d71;
        text-shadow: 0 0 15px rgba(255, 61, 113, 0.4);
    }

    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #050510;
    }
    ::-webkit-scrollbar-thumb {
        background: #2a2a4a;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #707090;
        background-color: transparent !important;
        border: none !important;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        color: #00ffa3 !important;
        border-bottom: 2px solid #00ffa3 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CACHED RESOURCES
# ──────────────────────────────────────────────
@st.cache_resource
def get_data_engine():
    """Create a cached Binance data engine."""
    return BinanceDataEngine()


@st.cache_resource
def get_strategy():
    """Create a cached strategy instance."""
    return ConfluenceStrategy()


# ──────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────
@st.cache_data(ttl=config.DASHBOARD_REFRESH_SECONDS)
def fetch_market_data(symbol: str, interval: str, limit: int):
    """Fetch and cache market data."""
    engine = get_data_engine()
    df = engine.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not df.empty:
        df = calculate_all_indicators(df)
    return df


@st.cache_data(ttl=config.DASHBOARD_REFRESH_SECONDS)
def fetch_price(symbol: str):
    """Fetch current price."""
    engine = get_data_engine()
    return engine.get_current_price(symbol)


# ──────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────
def create_main_chart(df: pd.DataFrame, signal=None):
    """Create an interactive candlestick chart with indicators."""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("", "RSI", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff5252",
            increasing_fillcolor="#00e676",
            decreasing_fillcolor="#ff5252",
        ),
        row=1, col=1,
    )

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(120, 120, 200, 0.4)", width=1, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(120, 120, 200, 0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(120, 120, 200, 0.05)",
            showlegend=False,
        ), row=1, col=1)

    # EMAs
    if "EMA_short" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_short"], name=f"EMA {config.EMA_SHORT}",
            line=dict(color="#ffd740", width=1.5),
        ), row=1, col=1)
    if "EMA_long" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_long"], name=f"EMA {config.EMA_LONG}",
            line=dict(color="#40c4ff", width=1.5),
        ), row=1, col=1)

    # VWAP
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#e040fb", width=1, dash="dash"),
        ), row=1, col=1)

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#7c4dff", width=2),
        ), row=2, col=1)
        # Overbought/Oversold lines
        fig.add_hline(y=config.RSI_OVERBOUGHT, line_dash="dash",
                     line_color="rgba(255, 82, 82, 0.5)", row=2, col=1)
        fig.add_hline(y=config.RSI_OVERSOLD, line_dash="dash",
                     line_color="rgba(0, 230, 118, 0.5)", row=2, col=1)
        fig.add_hrect(y0=config.RSI_OVERSOLD, y1=config.RSI_OVERBOUGHT,
                     fillcolor="rgba(124, 77, 255, 0.05)", line_width=0, row=2, col=1)

    # Volume
    colors = ["#00e676" if c >= o else "#ff5252"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.6,
    ), row=3, col=1)

    if "vol_sma" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vol_sma"], name="Vol SMA",
            line=dict(color="#ffd740", width=1),
        ), row=3, col=1)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        font=dict(family="Inter", color="#e0e0e0"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis3=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")

    return fig


def create_equity_chart(trades: list[dict]):
    """Create an equity curve chart from trade history."""
    if not trades:
        return None

    equity = 10000.0
    points = [{"trade": 0, "equity": equity}]

    for i, t in enumerate(trades):
        try:
            pnl = float(t.get("pnl_usdt", 0))
            equity += pnl
            points.append({"trade": i + 1, "equity": equity})
        except (ValueError, TypeError):
            pass

    df = pd.DataFrame(points)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["trade"], y=df["equity"],
        fill="tozeroy",
        fillcolor="rgba(0, 230, 118, 0.1)",
        line=dict(color="#00e676", width=2),
        name="Portfolio Value",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        font=dict(family="Inter", color="#e0e0e0"),
        xaxis_title="Trade #",
        yaxis_title="Portfolio Value ($)",
    )

    return fig


def update_config_settings(new_pair: str, new_interval: str):
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        import re
        # Update TRADING_PAIR
        pattern_pair = r'(TRADING_PAIR\s*=\s*["\'])[A-Z0-9]+(["\'])'
        content = re.sub(pattern_pair, r'\g<1>' + new_pair + r'\g<2>', content)
        
        # Update TIMEFRAME
        pattern_interval = r'(TIMEFRAME\s*=\s*["\'])\w+(["\'])'
        content = re.sub(pattern_interval, r'\g<1>' + new_interval + r'\g<2>', content)
        
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        st.error(f"Failed to update config.py: {e}")
        return False


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Interactive Coin Pair Selector
    st.markdown("### 💱 Active Pair")
    popular_pairs = ["BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "Custom"]
    
    default_index = 0
    if config.TRADING_PAIR in popular_pairs:
        default_index = popular_pairs.index(config.TRADING_PAIR)
    else:
        default_index = popular_pairs.index("Custom")
        
    selected_pair = st.selectbox("Select Coin Pair", popular_pairs, index=default_index)
    
    if selected_pair == "Custom":
        symbol = st.text_input("Enter Custom Pair", value=config.TRADING_PAIR).upper().strip()
    else:
        symbol = selected_pair

    # Interactive Timeframe Selector
    st.markdown("### ⏱️ Timeframe")
    popular_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
    
    default_interval_index = 0
    if config.TIMEFRAME in popular_intervals:
        default_interval_index = popular_intervals.index(config.TIMEFRAME)
        
    interval = st.selectbox("Select Timeframe", popular_intervals, index=default_interval_index)

    # Save to config.py if desired
    settings_changed = (symbol != config.TRADING_PAIR) or (interval != config.TIMEFRAME)
    if settings_changed:
        st.warning(
            f"Viewing: **{symbol}** ({interval})\n"
            f"Bot default: **{config.TRADING_PAIR}** ({config.TIMEFRAME})"
        )
        if st.button("💾 Set as Bot Default (updates config.py)"):
            if update_config_settings(symbol, interval):
                st.success(f"Updated config.py to {symbol} ({interval})!")
                st.rerun()
    else:
        st.success(f"Active: **{symbol}** | **{interval}** (Bot Default)")

    candles = st.slider("Candles to display", 50, 500, config.KLINE_LIMIT)

    st.markdown("---")

    mode = "🟡 TESTNET" if config.USE_TESTNET else "🔴 LIVE"
    st.markdown(f"**Mode:** {mode}")
    st.markdown(f"**Risk/Trade:** {config.RISK_PER_TRADE * 100:.1f}%")
    st.markdown(f"**Max Drawdown:** {config.MAX_DAILY_DRAWDOWN * 100:.1f}%")
    st.markdown(f"**Buy Threshold:** {config.CONFLUENCE_BUY_THRESHOLD}")

    st.markdown("---")
    auto_refresh = st.toggle("Auto Refresh", value=True)
    if auto_refresh:
        st.markdown(f"_Refreshing every {config.DASHBOARD_REFRESH_SECONDS}s_")


# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
st.markdown("# 🤖 Binance Trading Bot Dashboard")

# Fetch data
df = fetch_market_data(symbol, interval, candles)

if df.empty:
    st.error("❌ Could not fetch market data. Check your connection and API configuration.")
    st.stop()

# Get current signal
strategy = get_strategy()
signal = strategy.evaluate(df)
latest = df.iloc[-1]
current_price = float(latest["close"])

# ── Top Metrics Row ──────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("💲 Price", format_price(current_price))

with col2:
    price_change = (float(df["close"].iloc[-1]) - float(df["close"].iloc[-2])) / float(df["close"].iloc[-2]) * 100
    st.metric("📊 Change", format_pct(price_change), delta=f"{price_change:.2f}%")

with col3:
    signal_map = {
        SignalType.BUY: ("🟢 BUY", "signal-buy"),
        SignalType.SELL: ("🔴 SELL", "signal-sell"),
        SignalType.HOLD: ("🟡 HOLD", "signal-hold"),
    }
    sig_text, sig_class = signal_map.get(signal.type, ("⚪ N/A", "signal-hold"))
    st.metric("🎯 Signal", sig_text)

with col4:
    score_pct = signal.score * 100
    st.metric("📈 Confidence", f"{score_pct:.1f}%")

with col5:
    rsi_val = latest.get("rsi", 0)
    rsi_display = f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "N/A"
    st.metric("⚡ RSI", rsi_display)

st.markdown("---")

# ── Main Chart ───────────────────────────────
st.markdown("### 📊 Price Chart with Indicators")
fig = create_main_chart(df, signal)
st.plotly_chart(fig, use_container_width=True)

# ── AI Analyst Reasoning ──────────────────────
if config.LLM_ANALYST_ENABLED and config.GEMINI_API_KEY:
    st.markdown("### 🧠 AI Analyst Reasoning")
    with st.status("AI Analyst is reviewing technicals...", expanded=True) as status:
        analyst = get_analyst()
        ai_verdict = analyst.analyze_trade(
            symbol=symbol,
            side=signal.type.value,
            current_price=current_price,
            df=df,
            technical_reasons=signal.reasons,
            confluence_score=signal.score
        )
        status.update(label="Analysis Complete", state="complete", expanded=True)
        
        c1, c2 = st.columns([1, 4])
        with c1:
            conf = ai_verdict.get("confidence", 0) * 100
            st.metric("AI Confidence", f"{conf:.0f}%")
            
            action = ai_verdict.get("action", "HOLD")
            if action == "BUY":
                st.markdown('<div class="signal-buy">🟢 BUY</div>', unsafe_allow_html=True)
            elif action == "SELL":
                st.markdown('<div class="signal-sell">🔴 SELL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-hold">🟡 HOLD</div>', unsafe_allow_html=True)
        
        with c2:
            st.info(ai_verdict.get("reasoning", "No reasoning provided."))

# ── Indicator Details ────────────────────────
st.markdown("### 🔍 Signal Breakdown")

ind_cols = st.columns(3)

for i, reason in enumerate(signal.reasons):
    col = ind_cols[i % 3]
    with col:
        if "bullish" in reason.lower() or "oversold" in reason.lower() or "below" in reason.lower() or "↗" in reason:
            css_class = "indicator-bullish"
        elif "bearish" in reason.lower() or "overbought" in reason.lower() or "above" in reason.lower() or "↘" in reason:
            css_class = "indicator-bearish"
        else:
            css_class = "indicator-neutral"
        st.markdown(f'<div class="{css_class}">{reason}</div>', unsafe_allow_html=True)

# ── Indicator Weights & Scores ───────────────
st.markdown("### ⚖️ Indicator Scores")

score_data = []
for key, weight in config.INDICATOR_WEIGHTS.items():
    score = signal.indicator_scores.get(key, 0.5)
    weighted = score * weight
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    direction = "🟢" if score > 0.55 else "🔴" if score < 0.45 else "🟡"
    score_data.append({
        "Indicator": key.upper(),
        "Signal": direction,
        "Score": f"{score:.2f}",
        "Weight": f"{weight:.0%}",
        "Weighted": f"{weighted:.4f}",
        "Strength": bar,
    })

score_df = pd.DataFrame(score_data)
st.dataframe(score_df, use_container_width=True, hide_index=True)

st.markdown(f"**Composite Score: {signal.score:.4f}** | Threshold: ≥ {config.CONFLUENCE_BUY_THRESHOLD} (BUY) / ≤ {config.CONFLUENCE_SELL_THRESHOLD} (SELL)")

# ── Trade History & Performance ──────────────
st.markdown("---")

tab1, tab2 = st.tabs(["📋 Trade History", "📊 Performance"])

with tab1:
    trades = load_trade_journal()
    if trades:
        trade_df = pd.DataFrame(trades)
        # Display most recent first
        trade_df = trade_df.iloc[::-1].head(50)
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet. Start the bot with `python main.py` to begin trading.")

with tab2:
    trades = load_trade_journal()
    if trades:
        perf = calculate_performance(trades)

        perf_cols = st.columns(4)
        with perf_cols[0]:
            st.metric("Total Trades", perf["total_trades"])
        with perf_cols[1]:
            st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
        with perf_cols[2]:
            st.metric("Total P&L", format_pct(perf["total_pnl_pct"]))
        with perf_cols[3]:
            st.metric("Max Drawdown", format_pct(perf["max_drawdown_pct"]))

        # Equity curve
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
        if closed_trades:
            eq_chart = create_equity_chart(closed_trades)
            if eq_chart:
                st.plotly_chart(eq_chart, use_container_width=True)
    else:
        st.info("No performance data available yet.")

# ── Auto-refresh ─────────────────────────────
if auto_refresh:
    import time
    time.sleep(0.1)  # Small delay for rendering
    st.cache_data.clear()

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #555; font-size: 12px;">'
    '🤖 Binance Trading Bot Dashboard • Confluence Strategy Engine • '
    f'Mode: {"TESTNET" if config.USE_TESTNET else "LIVE"}'
    '</div>',
    unsafe_allow_html=True,
)
