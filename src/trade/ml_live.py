"""ML Live Paper Trader — Real-time paper trading with XGBoost predictions.

No broker needed. Uses yfinance for real-time prices.
Trains XGBoost on recent 1H data, makes predictions every hour,
logs all trades to SQLite database for tracking.

Usage: python -m trade.main --ml-live
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import talib
import yfinance as yf
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from trade.ml_backtest import _build_features, _build_target, SPREADS, SLIPPAGE

warnings.filterwarnings("ignore")
console = Console()

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

EXTRA_SPREADS = {
    "GBPNZD=X": 0.00030, "AUDNZD=X": 0.00020, "GBPCHF=X": 0.00020,
    "GC=F": 0.40, "USDCHF=X": 0.00010,
}
SPREADS.update(EXTRA_SPREADS)

DB_PATH = Path("data/ml_paper_trades.db")
SYMBOLS = ["GBPNZD=X", "GC=F", "AUDNZD=X", "GBPCHF=X", "USDCAD=X", "USDCHF=X"]
ACCOUNT_SIZE = 10000.0

# === FUNDERPRO PROP FIRM RULES ===
RISK_PER_TRADE = 0.0075  # 0.75% max (firm rule)
CONFIDENCE_THRESHOLD = 0.53
HORIZON = 8  # 8-hour prediction horizon
CHECK_INTERVAL = 300  # Check every 5 minutes
MAX_OPEN_TRADES = 2  # FunderPro: max 2 concurrent
MAX_TRADES_PER_DAY = 4  # FunderPro: max 4/day
MAX_DAILY_LOSS_PCT = 3.0  # Hard stop 3% (firm: 5%, safety: 4%)
MAX_TOTAL_DD_PCT = 7.0  # Hard stop 7% (firm: 10%, safety: 8%)
CONSEC_LOSS_COOLDOWN = 3600  # 60 min cooldown after 2 consecutive losses
SL_ATR_MULT = 1.5  # Stop loss at 1.5x ATR
TP_ATR_MULT = 2.5  # Take profit at 2.5x ATR → R:R = 1.67 (min 1.5)


def _init_db():
    """Initialize SQLite database for trade tracking."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL,
            take_profit REAL,
            predicted_exit_price REAL,
            actual_exit_price REAL,
            quantity REAL NOT NULL,
            confidence REAL NOT NULL,
            pnl REAL,
            status TEXT DEFAULT 'open',
            exit_reason TEXT,
            horizon_end TEXT,
            spread_cost REAL,
            notes TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            balance REAL NOT NULL,
            equity REAL NOT NULL,
            open_trades INTEGER,
            total_trades INTEGER,
            total_pnl REAL,
            win_rate REAL,
            max_dd REAL
        )
    """)
    conn.commit()
    return conn


def _get_account_stats(conn) -> dict:
    """Get current account statistics from database."""
    cur = conn.execute("SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
                       "SUM(pnl), COUNT(CASE WHEN status='open' THEN 1 END) "
                       "FROM trades")
    row = cur.fetchone()
    total = row[0] or 0
    wins = row[1] or 0
    total_pnl = row[2] or 0.0
    open_count = row[3] or 0
    wr = wins / total * 100 if total > 0 else 0
    # Daily P&L
    today = datetime.now().strftime("%Y-%m-%d")
    cur2 = conn.execute("SELECT COALESCE(SUM(pnl),0), COUNT(*) FROM trades WHERE timestamp LIKE ? AND status='closed'", (today + "%",))
    row2 = cur2.fetchone()
    daily_pnl = row2[0] or 0.0
    daily_trades = row2[1] or 0
    # Today's total trades (open + closed)
    cur3 = conn.execute("SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (today + "%",))
    trades_today = cur3.fetchone()[0] or 0
    # Consecutive losses
    cur4 = conn.execute("SELECT pnl FROM trades WHERE status='closed' ORDER BY id DESC LIMIT 2")
    recent = [r[0] for r in cur4.fetchall()]
    consec_losses = len(recent) == 2 and all(p < 0 for p in recent)

    return {
        "total_trades": total,
        "wins": wins,
        "total_pnl": total_pnl,
        "open_trades": open_count,
        "win_rate": wr,
        "balance": ACCOUNT_SIZE + total_pnl,
        "daily_pnl": daily_pnl,
        "daily_trades": daily_trades,
        "trades_today": trades_today,
        "consec_losses": consec_losses,
    }


def _train_model(sym: str) -> tuple:
    """Download latest data and train XGBoost model."""
    df = yf.download(sym, period="2y", interval="1h", progress=False)
    if df.empty or len(df) < 4500:
        return None, None, None

    if hasattr(df.columns, 'levels'):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    features = _build_features(df)
    target = _build_target(df, horizon=HORIZON, min_move_pct=0.001)

    data = features.copy()
    data["target"] = target
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 4100:
        return None, None, None

    # Train on last 4000 bars (rolling window)
    train_data = data.iloc[-4000:]
    X_train = train_data.drop(columns=["target"]).fillna(0)
    y_train = train_data["target"]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    y_map = {-1: 0, 0: 1, 1: 2}
    y_train_m = y_train.map(y_map)

    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="mlogloss", verbosity=0,
            random_state=42, seed=42,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.04,
            subsample=0.8, min_samples_leaf=20, random_state=42,
        )

    model.fit(X_train_s, y_train_m)
    return model, scaler, df


def _predict_now(model, scaler, df, sym) -> dict | None:
    """Make prediction on latest candle."""
    features = _build_features(df)
    data = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(data) < 2:
        return None

    X_latest = data.iloc[[-1]]
    X_scaled = scaler.transform(X_latest)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    y_inv = {0: -1, 1: 0, 2: 1}
    pred_class = y_inv.get(int(pred), 0)
    max_prob = float(proba.max())

    if pred_class == 0 or max_prob < CONFIDENCE_THRESHOLD:
        return None

    price = float(df["close"].iloc[-1])
    atr_val = float(features["atr_14"].iloc[-1])
    if math.isnan(atr_val) or atr_val <= 0:
        atr_val = price * 0.001

    return {
        "symbol": sym,
        "action": "BUY" if pred_class == 1 else "SELL",
        "price": price,
        "confidence": max_prob,
        "atr": atr_val,
        "pred_class": pred_class,
        "timestamp": str(df.index[-1]),
    }


def run_live_paper():
    """Main loop: train models, predict, track trades."""
    console.print(Panel.fit(
        "[bold green]ML LIVE PAPER TRADER[/bold green]\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"Risk: {RISK_PER_TRADE*100:.2f}% per trade | SL: {SL_ATR_MULT}x ATR | TP: {TP_ATR_MULT}x ATR\n"
        f"Account: ${ACCOUNT_SIZE:,.0f} (paper) | Max {MAX_OPEN_TRADES} open | Max {MAX_TRADES_PER_DAY}/day\n"
        f"DD limits: {MAX_DAILY_LOSS_PCT}% daily / {MAX_TOTAL_DD_PCT}% total\n\n"
        "[dim]No broker needed — uses yfinance real-time prices[/dim]\n"
        "[dim]Ctrl+C to stop. All trades saved to data/ml_paper_trades.db[/dim]",
        title="Live Paper Trading",
        border_style="green",
    ))

    conn = _init_db()
    models = {}
    last_train = {}
    last_candle = {}

    console.print("\n[bold]Training models on 2 years of 1H data...[/bold]")
    for sym in SYMBOLS:
        console.print(f"  Training {sym}...", end=" ")
        model, scaler, df = _train_model(sym)
        if model is not None:
            models[sym] = (model, scaler, df)
            last_train[sym] = datetime.now()
            console.print("[green]OK[/green]")
        else:
            console.print("[red]FAILED[/red]")

    if not models:
        console.print("[red]No models trained. Check internet connection.[/red]")
        return

    console.print(f"\n[bold green]Bot running. Checking for signals every {CHECK_INTERVAL//60} minutes.[/bold green]")
    console.print("[dim]Press Ctrl+C to stop.\n[/dim]")

    iteration = 0
    while True:
        try:
            iteration += 1
            now = datetime.now()

            # Check and close expired trades (horizon reached)
            _check_exits(conn, models)

            # Get account stats
            stats = _get_account_stats(conn)

            # Display dashboard
            _display_dashboard(stats, models, iteration)

            # Retrain every 24 hours
            for sym in SYMBOLS:
                if sym in last_train and (now - last_train[sym]).total_seconds() > 86400:
                    console.print(f"\n  [dim]Retraining {sym}...[/dim]", end=" ")
                    model, scaler, df = _train_model(sym)
                    if model is not None:
                        models[sym] = (model, scaler, df)
                        last_train[sym] = now
                        console.print("[green]OK[/green]")

            # === PROP FIRM RISK CHECKS ===
            # Total drawdown kill switch
            if stats["total_pnl"] < 0 and abs(stats["total_pnl"]) / ACCOUNT_SIZE * 100 >= MAX_TOTAL_DD_PCT:
                console.print(f"  [red bold]KILL SWITCH: Total DD {abs(stats['total_pnl'])/ACCOUNT_SIZE*100:.1f}% >= {MAX_TOTAL_DD_PCT}%. NO TRADING.[/red bold]")
                time.sleep(CHECK_INTERVAL)
                continue

            # Daily loss limit
            if stats["daily_pnl"] < 0 and abs(stats["daily_pnl"]) / ACCOUNT_SIZE * 100 >= MAX_DAILY_LOSS_PCT:
                console.print(f"  [red]Daily loss limit hit ({abs(stats['daily_pnl'])/ACCOUNT_SIZE*100:.1f}%). Waiting for tomorrow.[/red]")
                time.sleep(CHECK_INTERVAL)
                continue

            # Weekend check (no holding over weekend — close Friday 20:00 UTC)
            if now.weekday() == 4 and now.hour >= 20:
                console.print("  [yellow]Friday close — no new trades. Weekend rule.[/yellow]")
                time.sleep(CHECK_INTERVAL)
                continue

            # Consecutive losses cooldown
            if stats["consec_losses"]:
                cur_cool = conn.execute("SELECT timestamp FROM trades WHERE status='closed' ORDER BY id DESC LIMIT 1")
                last_loss_row = cur_cool.fetchone()
                if last_loss_row:
                    last_loss_time = datetime.fromisoformat(last_loss_row[0])
                    cooldown_left = CONSEC_LOSS_COOLDOWN - (now - last_loss_time).total_seconds()
                    if cooldown_left > 0:
                        console.print(f"  [yellow]Cooldown: {int(cooldown_left/60)}min left after 2 consecutive losses[/yellow]")
                        time.sleep(CHECK_INTERVAL)
                        continue

            # Check for new signals
            for sym in SYMBOLS:
                if sym not in models:
                    continue

                # Max open trades check
                if stats["open_trades"] >= MAX_OPEN_TRADES:
                    break

                # Max daily trades check
                if stats["trades_today"] >= MAX_TRADES_PER_DAY:
                    break

                try:
                    df_latest = yf.download(sym, period="1mo", interval="1h", progress=False)
                    if df_latest.empty:
                        continue
                    if hasattr(df_latest.columns, 'levels'):
                        df_latest.columns = [c[0].lower() for c in df_latest.columns]
                    else:
                        df_latest.columns = [c.lower() for c in df_latest.columns]
                except Exception:
                    continue

                # Check if we have a new candle
                latest_ts = str(df_latest.index[-1])
                if latest_ts == last_candle.get(sym):
                    continue
                last_candle[sym] = latest_ts

                # No duplicate positions
                cur = conn.execute("SELECT COUNT(*) FROM trades WHERE symbol=? AND status='open'", (sym,))
                if cur.fetchone()[0] > 0:
                    continue

                model, scaler, df_hist = models[sym]
                df_combined = df_latest
                if len(df_combined) < 50:
                    continue

                signal = _predict_now(model, scaler, df_combined, sym)
                if signal is None:
                    try:
                        feat = _build_features(df_combined)
                        d = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
                        X = scaler.transform(d.iloc[[-1]])
                        proba = model.predict_proba(X)[0]
                        y_inv = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
                        best = int(proba.argmax())
                        console.print(
                            f"  [dim]{sym}: {y_inv[best]} ({proba[best]:.1%}) | "
                            f"S:{proba[0]:.1%} N:{proba[1]:.1%} L:{proba[2]:.1%}[/dim]"
                        )
                    except Exception:
                        pass
                    continue

                # === POSITION SIZING WITH SL/TP (PROP FIRM COMPLIANT) ===
                price = signal["price"]
                atr = signal["atr"]
                spread = SPREADS.get(sym, 0.0002)
                slip = spread * SLIPPAGE

                # SL/TP based on ATR
                sl_dist = atr * SL_ATR_MULT
                tp_dist = atr * TP_ATR_MULT

                if signal["action"] == "BUY":
                    sl_price = price - sl_dist
                    tp_price = price + tp_dist
                else:
                    sl_price = price + sl_dist
                    tp_price = price - tp_dist

                # Position sizing: risk 0.75% of account
                risk_amt = stats["balance"] * RISK_PER_TRADE
                qty = risk_amt / sl_dist if sl_dist > 0 else 0
                max_qty = stats["balance"] * 5 / price if price > 0 else 0
                qty = min(qty, max_qty)
                cost = (spread + slip * 2) * qty

                if qty <= 0:
                    continue

                horizon_end = (now + timedelta(hours=HORIZON)).isoformat()

                conn.execute(
                    "INSERT INTO trades (timestamp, symbol, action, entry_price, "
                    "stop_loss, take_profit, quantity, confidence, status, "
                    "horizon_end, spread_cost, notes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)",
                    (now.isoformat(), sym, signal["action"], price,
                     round(sl_price, 5), round(tp_price, 5),
                     round(qty, 4), signal["confidence"], horizon_end, round(cost, 2),
                     f"atr={atr:.5f}, R:R={tp_dist/sl_dist:.2f}")
                )
                conn.commit()

                action_color = "green" if signal["action"] == "BUY" else "red"
                console.print(
                    f"\n  [{action_color}]>>> {signal['action']} {sym} @ {price:.4f} "
                    f"| SL: {sl_price:.4f} | TP: {tp_price:.4f} "
                    f"| Conf: {signal['confidence']:.1%} | Qty: {qty:.2f}[/{action_color}]"
                )

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            console.print("\n\n[bold]Stopping bot...[/bold]")
            stats = _get_account_stats(conn)
            console.print(f"  Total trades: {stats['total_trades']}")
            console.print(f"  P&L: ${stats['total_pnl']:+,.2f}")
            console.print(f"  Win rate: {stats['win_rate']:.1f}%")
            console.print(f"  Balance: ${stats['balance']:,.2f}")
            console.print(f"\n  Trades saved to: {DB_PATH}")
            conn.close()
            break


def _check_exits(conn, models):
    """Check open trades for SL/TP hits and time exits."""
    now_str = datetime.now().isoformat()
    now = datetime.now()

    # Get ALL open trades
    cur = conn.execute(
        "SELECT id, symbol, action, entry_price, stop_loss, take_profit, "
        "quantity, spread_cost, horizon_end FROM trades WHERE status='open'"
    )

    for row in cur.fetchall():
        trade_id, sym, action, entry_price, sl, tp, qty, cost, horizon_end = row

        # Get current price
        try:
            df = yf.download(sym, period="1d", interval="1h", progress=False)
            if df.empty:
                continue
            if hasattr(df.columns, 'levels'):
                cols = {c[0].lower(): c for c in df.columns}
                current_high = float(df[cols.get('high', cols.get('close'))].iloc[-1])
                current_low = float(df[cols.get('low', cols.get('close'))].iloc[-1])
                current_close = float(df[cols.get('close')].iloc[-1])
            else:
                current_high = float(df.get("high", df["close"]).iloc[-1])
                current_low = float(df.get("low", df["close"]).iloc[-1])
                current_close = float(df.get("close", df["Close"]).iloc[-1])
        except Exception:
            continue

        exit_price = None
        exit_reason = None

        # Check SL hit
        if sl and sl > 0:
            if action == "BUY" and current_low <= sl:
                exit_price = sl
                exit_reason = "SL"
            elif action == "SELL" and current_high >= sl:
                exit_price = sl
                exit_reason = "SL"

        # Check TP hit
        if tp and tp > 0 and exit_price is None:
            if action == "BUY" and current_high >= tp:
                exit_price = tp
                exit_reason = "TP"
            elif action == "SELL" and current_low <= tp:
                exit_price = tp
                exit_reason = "TP"

        # Time exit (horizon reached)
        if exit_price is None and horizon_end and now_str >= horizon_end:
            exit_price = current_close
            exit_reason = "TIME"

        # Weekend close (Friday 20:00+ UTC)
        if exit_price is None and now.weekday() == 4 and now.hour >= 20:
            exit_price = current_close
            exit_reason = "WEEKEND"

        if exit_price is None:
            continue

        # Calculate P&L
        cost = cost or 0
        if action == "BUY":
            pnl = (exit_price - entry_price) * qty - cost
        else:
            pnl = (entry_price - exit_price) * qty - cost

        conn.execute(
            "UPDATE trades SET status='closed', actual_exit_price=?, pnl=?, exit_reason=? WHERE id=?",
            (round(exit_price, 5), round(pnl, 2), exit_reason, trade_id)
        )
        conn.commit()

        color = "green" if pnl > 0 else "red"
        console.print(
            f"\n  [{color}]<<< CLOSED ({exit_reason}) {action} {sym} @ {exit_price:.4f} "
            f"| P&L: ${pnl:+,.2f} | Entry: {entry_price:.4f}[/{color}]"
        )


def _display_dashboard(stats, models, iteration):
    """Display live dashboard."""
    t = Table(title=f"ML Paper Trader — Update #{iteration} (FunderPro Rules)", show_header=True)
    t.add_column("Metric", width=22)
    t.add_column("Value", width=30)

    balance = stats["balance"]
    pnl = stats["total_pnl"]
    daily_pnl = stats["daily_pnl"]
    pc = "green" if pnl >= 0 else "red"
    dc = "green" if daily_pnl >= 0 else "red"

    t.add_row("Balance", f"${balance:,.2f}")
    t.add_row("Total P&L", f"[{pc}]${pnl:+,.2f} ({pnl/ACCOUNT_SIZE*100:+.2f}%)[/{pc}]")
    t.add_row("Daily P&L", f"[{dc}]${daily_pnl:+,.2f}[/{dc}]")
    t.add_row("Total Trades", str(stats["total_trades"]))
    t.add_row("Open / Max", f"{stats['open_trades']} / {MAX_OPEN_TRADES}")
    t.add_row("Today / Max", f"{stats['trades_today']} / {MAX_TRADES_PER_DAY}")
    t.add_row("Win Rate", f"{stats['win_rate']:.1f}%")
    dd_pct = abs(pnl) / ACCOUNT_SIZE * 100 if pnl < 0 else 0
    dd_color = "red" if dd_pct > 5 else "yellow" if dd_pct > 3 else "green"
    t.add_row("Drawdown", f"[{dd_color}]{dd_pct:.1f}% / {MAX_TOTAL_DD_PCT}% max[/{dd_color}]")
    t.add_row("Models", ", ".join(models.keys()))

    console.print(t)
