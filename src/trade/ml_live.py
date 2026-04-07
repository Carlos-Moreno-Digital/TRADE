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
THOUGHT_LOG = Path("data/bot_thoughts.log")
SYMBOLS = ["EURUSD=X", "USDJPY=X"]  # BBMR validated 16yr: +$24,968, 17/17 years
# === ACCOUNT CONFIG (change for different prop firm tiers) ===
import os
_TIER = os.environ.get("PROP_TIER", "10K")  # "10K" or "250K"

if _TIER == "250K":
    ACCOUNT_SIZE = 250000.0
    RISK_PER_TRADE = 0.003   # 0.3% = $750/trade (conservative for 250K)
    MAX_OPEN_TRADES = 4      # More capital = more concurrent positions
    MAX_TRADES_PER_DAY = 8   # More instruments = more opportunities
    MAX_DAILY_LOSS_PCT = 2.5 # Tighter daily limit for larger account
    MAX_TOTAL_DD_PCT = 6.0   # Tighter total DD for larger account
    RETRAIN_INTERVAL = 21600 # Retrain every 6 hours (not 24h)
else:
    ACCOUNT_SIZE = 10000.0
    RISK_PER_TRADE = 0.003   # 0.3% = $30/trade (BBMR tested DD ~11% at 0.5%)
    MAX_OPEN_TRADES = 2
    MAX_TRADES_PER_DAY = 4
    MAX_DAILY_LOSS_PCT = 3.0
    MAX_TOTAL_DD_PCT = 7.0
    RETRAIN_INTERVAL = 86400 # Retrain every 24 hours

# === SHARED RULES ===
CONFIDENCE_THRESHOLD = 0.53
HORIZON = 8  # 8-hour prediction horizon
CHECK_INTERVAL = 300  # Check every 5 minutes
CONSEC_LOSS_COOLDOWN = 3600  # 60 min cooldown after 2 consecutive losses
SL_ATR_MULT = 1.5  # Stop loss at 1.5x ATR
TP_ATR_MULT = 2.5  # Take profit at 2.5x ATR → R:R = 1.67


def _log_thought(category: str, symbol: str, message: str, data: dict = None):
    """Log bot's reasoning to thought log for auditing."""
    THOUGHT_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [{category:12s}] {symbol:12s} | {message}"
    if data:
        details = " | ".join(f"{k}={v}" for k, v in data.items())
        entry += f" | {details}"
    with open(THOUGHT_LOG, "a") as f:
        f.write(entry + "\n")


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
    """Main loop: multi-agent pipeline with real-time prices."""
    from trade.agents.multi.orchestrator import Orchestrator

    console.print(Panel.fit(
        "[bold green]BBMR PAPER TRADER[/bold green]\n"
        f"Symbols: {', '.join(SYMBOLS)}\n"
        f"Strategy: Bollinger Band Mean Reversion (validated 16yr: +$24,968, 17/17 years)\n"
        f"Pipeline: BBMR → Risk Shield → Compliance\n"
        f"Risk: 0.3% per trade | Only trade when ADX < 20\n"
        f"Account: ${ACCOUNT_SIZE:,.0f} (paper) | Max {MAX_OPEN_TRADES} open | Max {MAX_TRADES_PER_DAY}/day\n"
        f"DD limits: {MAX_DAILY_LOSS_PCT}% daily / {MAX_TOTAL_DD_PCT}% total\n\n"
        "[dim]No broker needed — uses yfinance real-time prices[/dim]\n"
        "[dim]Ctrl+C to stop. All trades saved to data/ml_paper_trades.db[/dim]",
        title="BBMR Paper Trading",
        border_style="green",
    ))

    conn = _init_db()
    orchestrator = Orchestrator()
    last_train = {}
    last_candle = {}

    console.print("\n[bold]Verifying data access for BBMR (no training needed)...[/bold]")
    all_data = {}
    for sym in SYMBOLS:
        console.print(f"  Checking {sym}...", end=" ")
        try:
            df = yf.download(sym, period="1mo", interval="1h", progress=False)
            if not df.empty and len(df) > 200:
                if hasattr(df.columns, 'levels'):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                all_data[sym] = df
                console.print(f"[green]{len(df)} candles[/green]")
            else:
                console.print("[red]insufficient data[/red]")
        except Exception:
            console.print("[red]download error[/red]")

    if not all_data:
        console.print("[red]No data downloaded. Check internet connection.[/red]")
        return

    train_results = orchestrator.train_models(SYMBOLS, all_data)
    for sym, ok in train_results.items():
        status = "[green]OK[/green]" if ok else "[red]FAILED[/red]"
        console.print(f"    {sym}: {status}")
        if ok:
            last_train[sym] = datetime.now()

    if not any(train_results.values()):
        console.print("[red]No models trained successfully.[/red]")
        return

    console.print(f"\n[bold green]Bot running. Checking for signals every {CHECK_INTERVAL//60} minutes.[/bold green]")
    console.print("[dim]Press Ctrl+C to stop.\n[/dim]")

    iteration = 0
    while True:
        try:
            iteration += 1
            now = datetime.now()

            # Check and close expired trades (SL/TP/time)
            _check_exits(conn, {})

            # Get account stats
            stats = _get_account_stats(conn)

            # Display dashboard
            # BBMR doesn't need training, so no models dict for dashboard
            _display_dashboard(stats, {s: None for s in SYMBOLS}, iteration)

            # No retraining — BBMR is a fixed rule-based strategy

            # === MULTI-AGENT PIPELINE ===
            # All risk checks are now handled by Risk Shield + Compliance agents
            account_state = {
                "balance": stats["balance"],
                "daily_pnl": stats["daily_pnl"],
                "total_pnl": stats["total_pnl"],
                "open_trades": stats["open_trades"],
                "trades_today": stats["trades_today"],
                "consec_losses": stats["consec_losses"],
                "open_positions": [],
                "last_loss_time": None,
                "recent_trades": [],
            }

            # Get last loss time for cooldown
            cur_lt = conn.execute("SELECT timestamp FROM trades WHERE status='closed' AND pnl < 0 ORDER BY id DESC LIMIT 1")
            lt_row = cur_lt.fetchone()
            if lt_row:
                account_state["last_loss_time"] = lt_row[0]

            # Get recent trades for martingale detection + StoplossGuard
            cur_rt = conn.execute("SELECT quantity, pnl, symbol, exit_reason FROM trades WHERE status='closed' ORDER BY id DESC LIMIT 5")
            account_state["recent_trades"] = [{"quantity": r[0], "pnl": r[1], "symbol": r[2], "exit_reason": r[3]} for r in cur_rt.fetchall()]

            for sym in SYMBOLS:
                # Download latest data
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

                # Check for new candle
                latest_ts = str(df_latest.index[-1])
                if latest_ts == last_candle.get(sym):
                    continue
                last_candle[sym] = latest_ts

                # No duplicate positions
                cur = conn.execute("SELECT COUNT(*) FROM trades WHERE symbol=? AND status='open'", (sym,))
                if cur.fetchone()[0] > 0:
                    continue

                if len(df_latest) < 50:
                    continue

                # === RUN FULL PIPELINE: Alpha → Risk → Quant → Compliance ===
                _log_thought("EVALUATE", sym, f"New candle {latest_ts}, running pipeline")
                decision = orchestrator.evaluate(sym, df_latest, account_state)

                # Log the full pipeline reasoning
                for phase in orchestrator.pipeline_log[-1:]:
                    for p_entry in phase.get("phases", []):
                        _log_thought(
                            p_entry["agent"].upper()[:12],
                            sym,
                            f"{p_entry['status']}: {p_entry.get('rationale', '')[:200]}",
                            {"errors": p_entry.get("errors", [])} if p_entry.get("errors") else None,
                        )
                    _log_thought("DECISION", sym, f"Final: {phase.get('final', 'unknown')}")

                if decision.status_flag == "APPROVED":
                    p = decision.computational_payload
                    # BBMR uses max_holding_bars (24h), ML uses horizon_hours
                    sym_horizon = p.get("max_holding_bars") or p.get("horizon_hours", HORIZON)
                    horizon_end = (now + timedelta(hours=sym_horizon)).isoformat()

                    _log_thought("EXECUTE", sym, f"{p['action']} @ {p['entry_price']:.4f}", {
                        "SL": p.get("sl_price", 0), "TP": p.get("tp_price", 0),
                        "qty": p.get("quantity", 0), "conf": p.get("confidence", 0),
                        "horizon": sym_horizon, "R:R": p.get("rr_ratio", 0),
                        "risk_pct": p.get("risk_pct", 0), "regime": p.get("regime", "?"),
                    })

                    conn.execute(
                        "INSERT INTO trades (timestamp, symbol, action, entry_price, "
                        "stop_loss, take_profit, quantity, confidence, status, "
                        "horizon_end, spread_cost, notes) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)",
                        (now.isoformat(), sym, p["action"], p["entry_price"],
                         p.get("sl_price", 0), p.get("tp_price", 0),
                         p.get("quantity", 0), p.get("confidence", 0),
                         horizon_end, p.get("spread_cost", 0),
                         f"R:R={p.get('rr_ratio', 0):.2f} | {decision.economic_rationale[:100]}")
                    )
                    conn.commit()

                    account_state["open_trades"] += 1
                    account_state["trades_today"] += 1

                    ac = "green" if p["action"] == "BUY" else "red"
                    console.print(
                        f"\n  [{ac}]>>> {p['action']} {sym} @ {p['entry_price']:.4f} "
                        f"| SL: {p.get('sl_price', 0):.4f} | TP: {p.get('tp_price', 0):.4f} "
                        f"| Conf: {p.get('confidence', 0):.1%} | Qty: {p.get('quantity', 0):.2f}[/{ac}]"
                    )
                    console.print(f"  [dim]  Pipeline: {decision.economic_rationale[:120]}[/dim]")

                elif decision.status_flag == "NO_SIGNAL":
                    diag = orchestrator.get_diagnostic(sym, df_latest)
                    _log_thought("NO_SIGNAL", sym, diag[:200])
                    console.print(f"  [dim]{diag}[/dim]")

                elif decision.is_rejected():
                    reason = decision.errors[0] if decision.errors else decision.economic_rationale[:80]
                    _log_thought("REJECTED", sym, f"by {decision.agent_domain}: {reason}")
                    if decision.agent_domain != "alpha_generator":
                        console.print(
                            f"  [yellow]{sym}: BLOCKED by {decision.agent_domain} — {reason}[/yellow]"
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


def _check_exits(conn, _unused):
    """Check open trades for SL/TP/trailing stop hits and time exits."""
    now_str = datetime.now().isoformat()
    now = datetime.now()

    cur = conn.execute(
        "SELECT id, symbol, action, entry_price, stop_loss, take_profit, "
        "quantity, spread_cost, horizon_end FROM trades WHERE status='open'"
    )

    for row in cur.fetchall():
        trade_id, sym, action, entry_price, sl, tp, qty, cost, horizon_end = row

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
        sl_dist = abs(entry_price - sl) if sl and sl > 0 else 0

        # === TRAILING STOP (Chandelier Exit) ===
        # Move SL to breakeven after 1R profit, then trail at 1R behind
        if sl and sl > 0 and sl_dist > 0:
            if action == "BUY":
                profit_pips = current_close - entry_price
                if profit_pips >= sl_dist * 1.5:
                    # Trail at 1R behind current price
                    new_sl = current_close - sl_dist
                    if new_sl > sl:
                        conn.execute("UPDATE trades SET stop_loss=? WHERE id=?", (round(new_sl, 5), trade_id))
                        sl = new_sl
                elif profit_pips >= sl_dist:
                    # Move to breakeven + small buffer
                    new_sl = entry_price + sl_dist * 0.1
                    if new_sl > sl:
                        conn.execute("UPDATE trades SET stop_loss=? WHERE id=?", (round(new_sl, 5), trade_id))
                        sl = new_sl
            else:  # SELL
                profit_pips = entry_price - current_close
                if profit_pips >= sl_dist * 1.5:
                    new_sl = current_close + sl_dist
                    if new_sl < sl:
                        conn.execute("UPDATE trades SET stop_loss=? WHERE id=?", (round(new_sl, 5), trade_id))
                        sl = new_sl
                elif profit_pips >= sl_dist:
                    new_sl = entry_price - sl_dist * 0.1
                    if new_sl < sl:
                        conn.execute("UPDATE trades SET stop_loss=? WHERE id=?", (round(new_sl, 5), trade_id))
                        sl = new_sl

        # Check SL hit
        if sl and sl > 0:
            if action == "BUY" and current_low <= sl:
                exit_price = sl
                # If SL was trailed above entry, it's a win
                exit_reason = "TRAIL" if sl > entry_price else "SL"
            elif action == "SELL" and current_high >= sl:
                exit_price = sl
                exit_reason = "TRAIL" if sl < entry_price else "SL"

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

        _log_thought("EXIT", sym, f"{exit_reason}: {action} closed @ {exit_price:.4f}", {
            "entry": entry_price, "exit": exit_price, "pnl": round(pnl, 2),
            "sl_was": sl, "tp_was": tp, "held_bars": "?",
            "trailing_active": sl != entry_price if exit_reason == "TRAIL" else False,
        })

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
