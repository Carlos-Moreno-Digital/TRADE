"""Lightweight Telegram alert bot.

Sends trade signals, execution confirmations, and daily P&L summaries
to a Telegram chat via the Bot API. Zero dependencies beyond `requests`
(already in the project for Dukascopy downloads).

Setup:
  1. Talk to @BotFather on Telegram, create a bot, get the TOKEN.
  2. Start a chat with your bot, send /start.
  3. Get your chat_id via https://api.telegram.org/bot<TOKEN>/getUpdates
  4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.

Usage from the orchestrator:
    from trade.alerts.telegram_bot import TelegramAlert
    alert = TelegramAlert()       # reads .env
    alert.send_signal("EURUSD", "BUY", entry=1.1050, sl=1.1020, tp=1.1120)
    alert.send_daily_summary(pnl=+45.30, trades=3, wr=0.667)

The bot is FIRE-AND-FORGET: if Telegram is unreachable it logs the
failure and moves on. Trading logic is NEVER blocked by an alert.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import requests


def _load_env() -> dict[str, str]:
    """Read .env file without python-dotenv (one less dependency)."""
    env = {}
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


class TelegramAlert:
    """Stateless Telegram alert sender."""

    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
    ):
        env = _load_env()
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN") or env.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID") or env.get("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.token and self.chat_id)

    @property
    def is_configured(self) -> bool:
        return self._enabled

    def _send(self, text: str) -> bool:
        if not self._enabled:
            return False
        try:
            url = self.API.format(token=self.token)
            r = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }, timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Pre-built message templates
    # ------------------------------------------------------------------
    def send_signal(
        self,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        confidence: float | None = None,
        regime: str | None = None,
    ) -> bool:
        conf_txt = f"  Confidence: {confidence:.1%}\n" if confidence else ""
        reg_txt = f"  Regime: {regime}\n" if regime else ""
        msg = (
            f"*SIGNAL* {side.upper()} {symbol}\n"
            f"  Entry: {entry:.5f}\n"
            f"  SL: {sl:.5f}\n"
            f"  TP: {tp:.5f}\n"
            f"{conf_txt}{reg_txt}"
            f"  Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return self._send(msg)

    def send_execution(
        self,
        symbol: str,
        side: str,
        fill_price: float,
        qty: float,
    ) -> bool:
        msg = (
            f"*EXECUTED* {side.upper()} {symbol}\n"
            f"  Fill: {fill_price:.5f}\n"
            f"  Qty: {qty:,.0f}\n"
            f"  Time: {datetime.utcnow().strftime('%H:%M UTC')}"
        )
        return self._send(msg)

    def send_close(
        self,
        symbol: str,
        side: str,
        pnl: float,
        reason: str,
    ) -> bool:
        emoji = "+" if pnl >= 0 else ""
        msg = (
            f"*CLOSED* {side.upper()} {symbol}\n"
            f"  P&L: {emoji}${pnl:.2f}\n"
            f"  Reason: {reason}\n"
            f"  Time: {datetime.utcnow().strftime('%H:%M UTC')}"
        )
        return self._send(msg)

    def send_daily_summary(
        self,
        pnl: float,
        trades: int,
        wr: float,
        balance: float | None = None,
    ) -> bool:
        emoji = "+" if pnl >= 0 else ""
        bal_txt = f"\n  Balance: ${balance:,.2f}" if balance else ""
        msg = (
            f"*DAILY SUMMARY*\n"
            f"  P&L: {emoji}${pnl:.2f}\n"
            f"  Trades: {trades}\n"
            f"  Win rate: {wr:.1%}{bal_txt}\n"
            f"  Date: {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        return self._send(msg)

    def send_risk_alert(self, message: str) -> bool:
        return self._send(f"*RISK ALERT*\n{message}")

    def send_raw(self, text: str) -> bool:
        return self._send(text)
