"""
AlgoBot v3.0 — RSI + MACD + BB + LLM Sentiment
github.com/holicpeter/algobot-v3
"""

import time
import logging
import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── CONFIG ───────────────────────────────────────────────
API_KEY       = os.getenv("ALPACA_API_KEY")
API_SECRET    = os.getenv("ALPACA_SECRET_KEY")
CLAUDE_KEY    = os.getenv("ANTHROPIC_API_KEY")
PAPER         = True

TICKERS = [
    # ETF
    "SPY","QQQ","TQQQ","IWM","XLK","XLF","XLE","GLD","TLT","SQQQ",
    # Tech
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","AVGO","CRM",
    "ORCL","NFLX","ADBE","QCOM","INTC",
    # Financie
    "JPM","BAC","GS","MS","V",
    # Zdravotníctvo
    "JNJ","UNH","PFE","ABBV","MRK",
    # Energia
    "XOM","CVX","COP","SLB","OXY",
    # Spotrebný tovar
    "WMT","COST","HD","MCD","NKE",
    # Priemysel
    "CAT","BA","GE","HON","UPS"
]

# Indikátory
RSI_PERIOD      = 14
RSI_OVERSOLD    = 35
RSI_OVERBOUGHT  = 65
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
BB_STD          = 2.0

# Risk
MAX_POSITION_PCT = 0.05
STOP_LOSS_PCT    = 0.02
TAKE_PROFIT_PCT  = 0.04

# ── LOGGING ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── CLIENTS ─────────────────────────────────────────────
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
data_client    = StockHistoricalDataClient(API_KEY, API_SECRET)

# ── INDICATORS ──────────────────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    return round((100 - (100 / (1 + rs))).iloc[-1], 2)

def calc_macd(closes):
    e12  = closes.ewm(span=MACD_FAST, adjust=False).mean()
    e26  = closes.ewm(span=MACD_SLOW, adjust=False).mean()
    line = e12 - e26
    sig  = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = line - sig
    return round(hist.iloc[-1], 4), round(hist.iloc[-2], 4)

def calc_bb(closes):
    mid   = closes.rolling(BB_PERIOD).mean()
    sigma = closes.rolling(BB_PERIOD).std()
    upper = (mid + BB_STD * sigma).iloc[-1]
    lower = (mid - BB_STD * sigma).iloc[-1]
    return round(upper, 4), round(lower, 4)

def calc_multi_timeframe(ticker):
    """Analýza na 3 timeframoch — 1min, 15min, 1hod"""
    scores = []
    timeframes = [
        (TimeFrame.Minute, 100),
        (TimeFrame.Minute * 15, 50),
        (TimeFrame.Hour, 30),
    ]
    for tf, limit in timeframes:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                limit=limit,
                start=datetime.now() - timedelta(days=10)
            )
            df = data_client.get_stock_bars(req).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker, level="symbol")
            if len(df) < 30:
                continue
            closes = df["close"]
            rsi    = calc_rsi(closes)
            hist, hist_prev = calc_macd(closes)
            bb_upper, bb_lower = calc_bb(closes)
            price = closes.iloc[-1]

            # Score: +1 bullish, -1 bearish, 0 neutral
            score = 0
            if rsi < RSI_OVERSOLD: score += 1
            elif rsi > RSI_OVERBOUGHT: score -= 1
            if hist > 0 and hist_prev < 0: score += 1
            elif hist < 0 and hist_prev > 0: score -= 1
            if price < bb_lower * 1.01: score += 1
            elif price > bb_upper * 0.99: score -= 1

            scores.append({"score": score, "rsi": rsi, "hist": hist, "price": price})
        except Exception as e:
            log.debug(f"MTF error {ticker}: {e}")
            continue

    if not scores:
        return None, 0, 0

    # Všetky 3 timeframy musia súhlasiť
    total = sum(s["score"] for s in scores)
    avg_rsi = sum(s["rsi"] for s in scores) / len(scores)
    price = scores[0]["price"]
    return price, total, avg_rsi

# ── LLM SENTIMENT ────────────────────────────────────────
def get_llm_sentiment(ticker, price, rsi, macd_hist):
    """Spýta sa Claude API na sentiment pred obchodom"""
    if not CLAUDE_KEY:
        return "NEUTRAL", 0.5

    try:
        prompt = f"""Analyzuj trading signál pre akciu {ticker}.

Technické indikátory:
- Cena: ${price:.2f}
- RSI: {rsi:.1f} {'(oversold)' if rsi < 35 else '(overbought)' if rsi > 65 else '(neutral)'}
- MACD histogram: {macd_hist:.4f} {'(bullish crossover)' if macd_hist > 0 else '(bearish crossover)'}

Na základe týchto indikátorov a tvojich znalostí o tejto akcii odpovedz PRESNE v JSON formáte:
{{"sentiment": "BUY" alebo "SELL" alebo "HOLD", "confidence": 0.0-1.0, "reason": "1 veta"}}

Odpovedz IBA JSON, nič iné."""

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        data = response.json()
        text = data["content"][0]["text"].strip()
        result = json.loads(text)
        sentiment   = result.get("sentiment", "HOLD")
        confidence  = float(result.get("confidence", 0.5))
        reason      = result.get("reason", "")
        log.info(f"  🧠 LLM {ticker}: {sentiment} (conf={confidence:.2f}) — {reason}")
        return sentiment, confidence

    except Exception as e:
        log.debug(f"LLM error {ticker}: {e}")
        return "NEUTRAL", 0.5

# ── DYNAMIC POSITION SIZING ──────────────────────────────
def calc_position_size(portfolio_value, cash, tech_score, llm_confidence):
    """Veľkosť pozície podľa sily signálu — 2% až 8%"""
    base = MAX_POSITION_PCT  # 5%
    # Tech score: max +/- 3 → upraví o +/-2%
    tech_adjust = (abs(tech_score) / 3) * 0.02
    # LLM confidence: 0.5-1.0 → 0%-1.5%
    llm_adjust  = (llm_confidence - 0.5) * 0.03
    pct = min(max(base + tech_adjust + llm_adjust, 0.02), 0.08)
    invest = min(portfolio_value * pct, cash * 0.95)
    log.info(f"  💰 Position size: {pct*100:.1f}% = ${invest:.2f}")
    return invest

# ── ORDERS ───────────────────────────────────────────────
def place_buy(ticker, price, invest):
    try:
        if invest < 1:
            log.warning(f"  ⚠️ Nedostatok kapitálu pre {ticker}")
            return
        qty = round(invest / price, 6)
        order = MarketOrderRequest(
            symbol=ticker, qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        log.info(f"  ✅ BUY {ticker} | qty={qty} @ ${price:.2f} | ${invest:.2f}")
    except Exception as e:
        log.error(f"  ❌ BUY error {ticker}: {e}")

def get_position(ticker):
    try: return trading_client.get_open_position(ticker)
    except: return None

def check_exit(pos):
    if not pos: return False
    pnl = (float(pos.current_price) - float(pos.avg_entry_price)) / float(pos.avg_entry_price)
    if pnl <= -STOP_LOSS_PCT:
        log.warning(f"  🛑 STOP LOSS {pos.symbol} | {pnl*100:.2f}%")
        return True
    if pnl >= TAKE_PROFIT_PCT:
        log.info(f"  💰 TAKE PROFIT {pos.symbol} | {pnl*100:.2f}%")
        return True
    return False

# ── MAIN ────────────────────────────────────────────────
def run():
    log.info("=" * 55)
    log.info("🤖 AlgoBot v3.0 — RSI+MACD+BB+LLM — PAPER TRADING")
    log.info(f"   Tickery: {len(TICKERS)} | MTF: 1min+15min+1hod")
    log.info(f"   LLM sentiment: {'✅ aktívny' if CLAUDE_KEY else '⚠️ bez API key'}")
    log.info(f"   SL: {STOP_LOSS_PCT*100}% | TP: {TAKE_PROFIT_PCT*100}%")
    log.info("=" * 55)

    try:
        if not trading_client.get_clock().is_open:
            log.info("💤 Trh zatvorený"); return

        acc = trading_client.get_account()
        portfolio = float(acc.portfolio_value)
        cash      = float(acc.buying_power)
        log.info(f"💼 Portfólio: ${portfolio:.2f} | Cash: ${cash:.2f}")

        buys = sells = holds = 0

        for ticker in TICKERS:
            log.info(f"📊 Analyzujem {ticker}...")

            # Exit check
            pos = get_position(ticker)
            if pos and check_exit(pos):
                trading_client.close_position(ticker)
                sells += 1
                time.sleep(1)
                continue

            # Multi-timeframe analýza
            price, tech_score, avg_rsi = calc_multi_timeframe(ticker)
            if price is None:
                holds += 1
                continue

            # Rozhodnutie
            if tech_score >= 2 and not pos:
                # LLM sentiment potvrdenie
                sentiment, confidence = get_llm_sentiment(ticker, price, avg_rsi, tech_score)
                if sentiment == "BUY" and confidence >= 0.6:
                    invest = calc_position_size(portfolio, cash, tech_score, confidence)
                    place_buy(ticker, price, invest)
                    cash -= invest
                    buys += 1
                else:
                    log.info(f"  ⏸️ LLM zamietol signál: {sentiment} ({confidence:.2f})")
                    holds += 1

            elif tech_score <= -2 and pos:
                trading_client.close_position(ticker)
                sells += 1

            else:
                log.info(f"  {ticker} score={tech_score} RSI={avg_rsi:.1f} HOLD")
                holds += 1

            time.sleep(1)

        log.info(f"✅ Run hotový: {buys} BUY | {sells} SELL | {holds} HOLD")

    except Exception as e:
        log.error(f"❌ Chyba: {e}")

if __name__ == "__main__":
    run()
