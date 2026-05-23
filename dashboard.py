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
    initial_sidebar_state="expanded"
)

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
    tf_map = {"1 min": TimeFrame.Minute, "15 min": TimeFrame.Minute * 15,
              "1 hod": TimeFrame.Hour, "1 deň": TimeFrame.Day}
    limit_map = {"1 min": 200, "15 min": 200, "1 hod": 200, "1 deň": 200}
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf_map[tf_label],
        limit=limit_map[tf_label],
        start=datetime.now() - timedelta(days=days)
    )
    df = data_client.get_stock_bars(req).df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")
    return df.tail(150)

# ── MAIN ─────────────────────────────────────────────────
def main():
    st.title("🤖 AlgoBot v3.0")
    st.caption("RSI + MACD + Bollinger Bands + LLM Sentiment | Paper Trading")

    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.header("🔑 API Kľúče")
        api_key = st.text_input("ALPACA_API_KEY", type="password",
                                value=os.getenv("ALPACA_API_KEY", ""))
        api_secret = st.text_input("ALPACA_SECRET_KEY", type="password",
                                   value=os.getenv("ALPACA_SECRET_KEY", ""))
        if st.button("🔄 Obnoviť", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        st.header("📊 Graf")
        selected_ticker = st.selectbox("Ticker", TICKERS)
        tf_opt = st.radio("Timeframe", ["1 min", "15 min", "1 hod", "1 deň"], index=3)
        days_back = st.slider("Dní dozadu", 1, 60, 10)

    if not api_key or not api_secret:
        st.info("👈 Vlož Alpaca API kľúče v bočnom paneli")
        return

    try:
        trading_client, data_client = get_clients(api_key, api_secret)
    except Exception as e:
        st.error(f"Chyba pripojenia: {e}")
        return

    # ── PORTFOLIO METRIKY ─────────────────────────────────
    try:
        acc = trading_client.get_account()
        clock = trading_client.get_clock()
        portfolio = float(acc.portfolio_value)
        cash = float(acc.buying_power)
        equity = float(acc.equity)
        last_equity = float(acc.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0
        market_open = clock.is_open

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💼 Portfólio", f"${portfolio:,.2f}")
        c2.metric("💵 Cash", f"${cash:,.2f}")
        c3.metric("📈 Denný P&L", f"${daily_pnl:+,.2f}", f"{daily_pnl_pct:+.2f}%")
        c4.metric("🏦 Trh", "🟢 Otvorený" if market_open else "🔴 Zatvorený")
    except Exception as e:
        st.error(f"Chyba účtu: {e}")
        return

    st.divider()

    # ── POZÍCIE + OBCHODY ─────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("📂 Otvorené pozície")
        try:
            positions = trading_client.get_all_positions()
            if positions:
                rows = []
                for p in positions:
                    rows.append({
                        "Ticker": p.symbol,
                        "Qty": round(float(p.qty), 4),
                        "Vstup": float(p.avg_entry_price),
                        "Cena": float(p.current_price),
                        "P&L $": float(p.unrealized_pl),
                        "P&L %": float(p.unrealized_plpc) * 100,
                    })
                df_pos = pd.DataFrame(rows)

                def _color(val):
                    return "color: #00c851" if val > 0 else "color: #ff4444" if val < 0 else ""

                st.dataframe(
                    df_pos.style
                        .format({"Vstup": "{:.2f}", "Cena": "{:.2f}",
                                 "P&L $": "{:+.2f}", "P&L %": "{:+.2f}%"})
                        .map(_color, subset=["P&L $", "P&L %"]),
                    use_container_width=True,
                    hide_index=True
                )
                total = sum(float(p.unrealized_pl) for p in positions)
                st.caption(f"Celkový nerealizovaný P&L: **${total:+,.2f}**")
            else:
                st.info("Žiadne otvorené pozície")
        except Exception as e:
            st.error(f"Chyba pozícií: {e}")

    with right:
        st.subheader("📋 Posledné obchody")
        try:
            orders = trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=25)
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
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Žiadne uzatvorené obchody")
        except Exception as e:
            st.error(f"Chyba obchodov: {e}")

    st.divider()

    # ── GRAF ─────────────────────────────────────────────
    st.subheader(f"📊 {selected_ticker} — {tf_opt}")

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
            subplot_titles=[f"{selected_ticker} + Bollinger Bands", "RSI (14)", "MACD (12/26/9)"]
        )

        # Sviečkový graf + BB
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Cena",
            increasing_line_color="#00c851", decreasing_line_color="#ff4444",
            increasing_fillcolor="#00c851", decreasing_fillcolor="#ff4444"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="BB Upper",
            line=dict(color="rgba(99,132,255,0.7)", dash="dash", width=1),
            showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_mid, name="BB Mid",
            line=dict(color="rgba(99,132,255,0.5)", dash="dot", width=1),
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name="BB Lower",
            line=dict(color="rgba(99,132,255,0.7)", dash="dash", width=1),
            fill="tonexty", fillcolor="rgba(99,132,255,0.06)",
            showlegend=True), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=rsi_s, name="RSI",
            line=dict(color="#f0a500", width=1.5)), row=2, col=1)
        for y, color in [(65, "rgba(255,68,68,0.5)"), (35, "rgba(0,200,81,0.5)")]:
            fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1],
                y0=y, y1=y, line=dict(color=color, dash="dash", width=1),
                row=2, col=1)

        # MACD
        hist_colors = ["#00c851" if v >= 0 else "#ff4444" for v in macd_hist]
        fig.add_trace(go.Bar(x=df.index, y=macd_hist, name="Histogram",
            marker_color=hist_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD",
            line=dict(color="#00c851", width=1.3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_sig, name="Signal",
            line=dict(color="#ff4444", width=1.3)), row=3, col=1)

        fig.update_layout(
            height=720,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117"
        )
        fig.update_yaxes(title_text="Cena $", row=1, col=1, gridcolor="#1f2937")
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], gridcolor="#1f2937")
        fig.update_yaxes(title_text="MACD", row=3, col=1, gridcolor="#1f2937")
        fig.update_xaxes(gridcolor="#1f2937")

        st.plotly_chart(fig, use_container_width=True)

        # Aktuálne hodnoty indikátorov
        cur_rsi = rsi_s.iloc[-1]
        cur_hist = macd_hist.iloc[-1]
        cur_price = closes.iloc[-1]
        bb_pos = (cur_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100

        i1, i2, i3, i4 = st.columns(4)
        rsi_label = "Oversold 🟢" if cur_rsi < 35 else ("Overbought 🔴" if cur_rsi > 65 else "Neutrál")
        macd_label = "Bullish ↑" if cur_hist > 0 else "Bearish ↓"
        bb_label = "Blízko dolného 🟢" if bb_pos < 20 else ("Blízko horného 🔴" if bb_pos > 80 else "Stred")
        i1.metric("RSI", f"{cur_rsi:.1f}", rsi_label)
        i2.metric("MACD Histogram", f"{cur_hist:.4f}", macd_label)
        i3.metric("BB pozícia", f"{bb_pos:.0f}%", bb_label)
        i4.metric("Posledná cena", f"${cur_price:.2f}")

    except Exception as e:
        st.error(f"Chyba grafu: {e}")

    st.divider()

    # ── TICKER SCANNER ────────────────────────────────────
    st.subheader("🔍 Ticker Scanner")
    st.caption("Denné signály pre všetkých 45 tickerov")

    if st.button("▶️ Spustiť scanner", type="primary"):
        results = []
        bar = st.progress(0)
        status_txt = st.empty()

        for i, tkr in enumerate(TICKERS):
            bar.progress((i + 1) / len(TICKERS))
            status_txt.text(f"Analyzujem {tkr} ({i+1}/{len(TICKERS)})...")
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
                    "MACD Hist": round(h, 4),
                    "Skóre": score,
                    "Signál": sig
                })
            except Exception:
                continue

        bar.empty()
        status_txt.empty()

        if results:
            df_scan = pd.DataFrame(results).sort_values("Skóre", ascending=False)
            buy_count = sum(1 for r in results if r["Skóre"] >= 2)
            sell_count = sum(1 for r in results if r["Skóre"] <= -2)
            hold_count = len(results) - buy_count - sell_count
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("🟢 BUY signály", buy_count)
            sc2.metric("🔴 SELL signály", sell_count)
            sc3.metric("⚪ HOLD", hold_count)
            st.dataframe(
                df_scan.style.format({"Cena $": "{:.2f}", "MACD Hist": "{:.4f}"}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Scanner nenašiel žiadne výsledky")

if __name__ == "__main__":
    main()
