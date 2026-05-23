"""
AlgoBot v3.0 — Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

st.set_page_config(
    page_title="AlgoBot v3.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mobile-friendly CSS
st.markdown("""
<style>
    .block-container { padding: 1rem 1rem 2rem; max-width: 100%; }
    .metric-container { background: #1a1d24; border-radius: 10px; padding: 12px; margin: 4px 0; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
    [data-testid="stDataFrame"] { font-size: 0.8rem; }
    .stButton button { width: 100%; font-size: 1rem; padding: 0.6rem; }
    @media (max-width: 640px) {
        [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        .stSelectbox, .stRadio { font-size: 0.9rem; }
    }
</style>
""", unsafe_allow_html=True)

TICKERS = [
    "SPY","QQQ","TQQQ","IWM","XLK","XLF","XLE","GLD","TLT","SQQQ",
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","AVGO","CRM",
    "ORCL","NFLX","ADBE","QCOM","INTC",
    "JPM","BAC","GS","MS","V",
    "JNJ","UNH","PFE","ABBV","MRK",
    "XOM","CVX","COP","SLB","OXY",
    "WMT","COST","HD","MCD","NKE",
    "CAT","BA","GE","HON","UPS"
]

# ── INDICATORS ────────────────────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(closes, fast=12, slow=26, signal=9):
    e12 = closes.ewm(span=fast, adjust=False).mean()
    e26 = closes.ewm(span=slow, adjust=False).mean()
    line = e12 - e26
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

def calc_bb(closes, period=20, std=2.0):
    mid = closes.rolling(period).mean()
    sigma = closes.rolling(period).std()
    return mid + std * sigma, mid, mid - std * sigma

# ── CLIENTS ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_clients(api_key, api_secret):
    trading = TradingClient(api_key, api_secret, paper=True)
    data = StockHistoricalDataClient(api_key, api_secret)
    return trading, data

@st.cache_data(ttl=60, show_spinner=False)
def fetch_bars(api_key, api_secret, ticker, tf_label, days):
    _, data_client = get_clients(api_key, api_secret)
    tf_map = {
        "1 min": TimeFrame.Minute,
        "15 min": TimeFrame.Minute * 15,
        "1 hod": TimeFrame.Hour,
        "1 deň": TimeFrame.Day
    }
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf_map[tf_label],
        limit=200,
        start=datetime.now() - timedelta(days=days)
    )
    df = data_client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")
    return df.tail(150)

# ── MAIN ─────────────────────────────────────────────────
def main():
    # Header
    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.title("🤖 AlgoBot v3.0")
        st.caption("RSI · MACD · BB · LLM | Paper Trading")
    with col_refresh:
        st.write("")
        if st.button("🔄", help="Obnoviť dáta", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # ── API KĽÚČE ─────────────────────────────────────────
    api_key = st.secrets.get("ALPACA_API_KEY", "") or os.getenv("ALPACA_API_KEY", "")
    api_secret = st.secrets.get("ALPACA_SECRET_KEY", "") or os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not api_secret:
        with st.expander("🔑 Zadaj API kľúče", expanded=True):
            api_key = st.text_input("ALPACA_API_KEY", type="password")
            api_secret = st.text_input("ALPACA_SECRET_KEY", type="password")
        if not api_key or not api_secret:
            st.info("Vlož Alpaca API kľúče vyššie alebo nastav ich v Streamlit Cloud Secrets.")
            return

    try:
        trading_client, data_client = get_clients(api_key, api_secret)
    except Exception as e:
        st.error(f"Chyba pripojenia: {e}")
        return

    # ── PORTFOLIO ─────────────────────────────────────────
    try:
        acc = trading_client.get_account()
        clock = trading_client.get_clock()
        portfolio = float(acc.portfolio_value)
        cash = float(acc.buying_power)
        equity = float(acc.equity)
        last_equity = float(acc.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💼 Portfólio", f"${portfolio:,.0f}")
        c2.metric("💵 Cash", f"${cash:,.0f}")
        c3.metric("📈 Denný P&L", f"${daily_pnl:+,.0f}", f"{daily_pnl_pct:+.2f}%")
        c4.metric("🏦 Trh", "🟢 Open" if clock.is_open else "🔴 Closed")

    except Exception as e:
        st.error(f"Chyba účtu: {e}")
        return

    st.divider()

    # ── TABS (mobile-friendly navigácia) ─────────────────
    tab_pos, tab_ord, tab_chart, tab_scan = st.tabs([
        "📂 Pozície", "📋 Obchody", "📊 Graf", "🔍 Scanner"
    ])

    # ── TAB: POZÍCIE ──────────────────────────────────────
    with tab_pos:
        try:
            positions = trading_client.get_all_positions()
            if positions:
                rows = []
                total_pnl = 0.0
                for p in positions:
                    pnl = float(p.unrealized_pl)
                    total_pnl += pnl
                    rows.append({
                        "Ticker": p.symbol,
                        "Qty": round(float(p.qty), 4),
                        "Vstup $": round(float(p.avg_entry_price), 2),
                        "Cena $": round(float(p.current_price), 2),
                        "P&L $": round(pnl, 2),
                        "P&L %": round(float(p.unrealized_plpc) * 100, 2),
                    })

                df_pos = pd.DataFrame(rows).sort_values("P&L %", ascending=False)

                def _color(val):
                    return "color: #00c851" if val > 0 else "color: #ff4444" if val < 0 else ""

                st.dataframe(
                    df_pos.style
                        .format({"Vstup $": "{:.2f}", "Cena $": "{:.2f}",
                                 "P&L $": "{:+.2f}", "P&L %": "{:+.2f}%"})
                        .map(_color, subset=["P&L $", "P&L %"]),
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, 40 + len(rows) * 38)
                )
                col_a, col_b = st.columns(2)
                col_a.metric("Celkový P&L", f"${total_pnl:+,.2f}")
                col_b.metric("Počet pozícií", len(rows))
            else:
                st.info("Žiadne otvorené pozície")
        except Exception as e:
            st.error(f"Chyba: {e}")

    # ── TAB: OBCHODY ──────────────────────────────────────
    with tab_ord:
        try:
            orders = trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=30)
            )
            if orders:
                rows = []
                for o in orders:
                    rows.append({
                        "Čas": o.filled_at.strftime("%d.%m %H:%M") if o.filled_at else "-",
                        "Ticker": o.symbol,
                        "Strana": "🟢 BUY" if o.side.value == "buy" else "🔴 SELL",
                        "Qty": round(float(o.filled_qty or 0), 4),
                        "Cena $": round(float(o.filled_avg_price or 0), 2),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True, height=500)
            else:
                st.info("Žiadne uzatvorené obchody")
        except Exception as e:
            st.error(f"Chyba: {e}")

    # ── TAB: GRAF ─────────────────────────────────────────
    with tab_chart:
        col_sel, col_tf, col_days = st.columns([2, 2, 1])
        with col_sel:
            selected_ticker = st.selectbox("Ticker", TICKERS, label_visibility="collapsed")
        with col_tf:
            tf_opt = st.selectbox("Timeframe", ["1 deň", "1 hod", "15 min", "1 min"],
                                  label_visibility="collapsed")
        with col_days:
            days_back = st.number_input("Dni", min_value=1, max_value=60, value=10,
                                        label_visibility="collapsed")

        try:
            with st.spinner(f"Načítavam {selected_ticker}..."):
                df = fetch_bars(api_key, api_secret, selected_ticker, tf_opt, days_back)

            closes = df["close"]
            rsi_s = calc_rsi(closes)
            macd_line, macd_sig, macd_hist = calc_macd(closes)
            bb_upper, bb_mid, bb_lower = calc_bb(closes)

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.55, 0.23, 0.22],
                subplot_titles=[f"{selected_ticker} + BB", "RSI (14)", "MACD (12/26/9)"]
            )

            fig.add_trace(go.Candlestick(
                x=df.index, open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="Cena",
                increasing_line_color="#00c851", decreasing_line_color="#ff4444",
                increasing_fillcolor="#00c851", decreasing_fillcolor="#ff4444"
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="BB Upper",
                line=dict(color="rgba(99,132,255,0.7)", dash="dash", width=1),
                showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_mid, name="BB Mid",
                line=dict(color="rgba(99,132,255,0.4)", dash="dot", width=1),
                showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name="BB Lower",
                line=dict(color="rgba(99,132,255,0.7)", dash="dash", width=1),
                fill="tonexty", fillcolor="rgba(99,132,255,0.06)",
                showlegend=False), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=rsi_s, name="RSI",
                line=dict(color="#f0a500", width=1.5)), row=2, col=1)
            for y, clr in [(65, "rgba(255,68,68,0.6)"), (35, "rgba(0,200,81,0.6)")]:
                fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1],
                    y0=y, y1=y, line=dict(color=clr, dash="dash", width=1),
                    row=2, col=1)

            hist_colors = ["#00c851" if v >= 0 else "#ff4444" for v in macd_hist]
            fig.add_trace(go.Bar(x=df.index, y=macd_hist, marker_color=hist_colors,
                showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
                line=dict(color="#00c851", width=1.3)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=macd_sig, name="Signal",
                line=dict(color="#ff4444", width=1.3)), row=3, col=1)

            fig.update_layout(
                height=580,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117"
            )
            for row_n in [1, 2, 3]:
                fig.update_yaxes(gridcolor="#1f2937", row=row_n, col=1)
            fig.update_xaxes(gridcolor="#1f2937")
            fig.update_yaxes(range=[0, 100], row=2, col=1)

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            cur_rsi = rsi_s.iloc[-1]
            cur_hist = macd_hist.iloc[-1]
            cur_price = closes.iloc[-1]
            bb_pos = (cur_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100

            i1, i2, i3, i4 = st.columns(4)
            i1.metric("RSI", f"{cur_rsi:.1f}",
                      "Oversold" if cur_rsi < 35 else ("Overbought" if cur_rsi > 65 else "Neutrál"))
            i2.metric("MACD", f"{cur_hist:.4f}", "Bullish" if cur_hist > 0 else "Bearish")
            i3.metric("BB %", f"{bb_pos:.0f}%",
                      "Dolný" if bb_pos < 20 else ("Horný" if bb_pos > 80 else "Stred"))
            i4.metric("Cena", f"${cur_price:.2f}")

        except Exception as e:
            st.error(f"Chyba grafu: {e}")

    # ── TAB: SCANNER ──────────────────────────────────────
    with tab_scan:
        st.caption("Denné BUY/SELL/HOLD signály pre všetkých 45 tickerov")

        if st.button("▶️ Spustiť scanner", type="primary", use_container_width=True):
            results = []
            bar = st.progress(0)
            status_txt = st.empty()

            for i, tkr in enumerate(TICKERS):
                bar.progress((i + 1) / len(TICKERS))
                status_txt.text(f"Analyzujem {tkr}... ({i+1}/{len(TICKERS)})")
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=tkr,
                        timeframe=TimeFrame.Day,
                        limit=60,
                        start=datetime.now() - timedelta(days=90)
                    )
                    df_s = data_client.get_stock_bars(req).df
                    if isinstance(df_s.index, pd.MultiIndex):
                        df_s = df_s.xs(tkr, level="symbol")
                    if len(df_s) < 30:
                        continue

                    cl = df_s["close"]
                    rsi_v = calc_rsi(cl).iloc[-1]
                    _, _, hist_s = calc_macd(cl)
                    h, h_prev = hist_s.iloc[-1], hist_s.iloc[-2]
                    bu, _, bl = calc_bb(cl)
                    price_v = cl.iloc[-1]

                    score = 0
                    if rsi_v < 35: score += 1
                    elif rsi_v > 65: score -= 1
                    if h > 0 and h_prev < 0: score += 1
                    elif h < 0 and h_prev > 0: score -= 1
                    if price_v < bl.iloc[-1] * 1.01: score += 1
                    elif price_v > bu.iloc[-1] * 0.99: score -= 1

                    sig = "🟢 BUY" if score >= 2 else ("🔴 SELL" if score <= -2 else "⚪ HOLD")
                    results.append({
                        "Ticker": tkr,
                        "Cena $": round(price_v, 2),
                        "RSI": round(rsi_v, 1),
                        "MACD": round(h, 4),
                        "Skóre": score,
                        "Signál": sig
                    })
                except Exception:
                    continue

            bar.empty()
            status_txt.empty()

            if results:
                df_scan = pd.DataFrame(results).sort_values("Skóre", ascending=False)
                buys = sum(1 for r in results if r["Skóre"] >= 2)
                sells = sum(1 for r in results if r["Skóre"] <= -2)
                holds = len(results) - buys - sells

                s1, s2, s3 = st.columns(3)
                s1.metric("🟢 BUY", buys)
                s2.metric("🔴 SELL", sells)
                s3.metric("⚪ HOLD", holds)

                st.dataframe(
                    df_scan.style.format({"Cena $": "{:.2f}", "MACD": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )

if __name__ == "__main__":
    main()
