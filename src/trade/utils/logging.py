"""Logging configuration for the trading system."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from trade.config import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure logging for the trading system."""
    config = config or LoggingConfig()

    # Remove default handler
    logger.remove()

    # Console handler
    if config.console:
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                   "<cyan>{extra[agent]:>15}</cyan> | <level>{message}</level>",
            filter=lambda record: "agent" in record["extra"],
        )
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            filter=lambda record: "agent" not in record["extra"],
        )

    # File handler
    if config.file:
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=config.rotation,
            retention=config.retention,
        )

    # Trade-specific log
    if config.trade_log_path:
        trade_path = Path(config.trade_log_path)
        trade_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(trade_path),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "TRADE" in record["message"] or "PAPER" in record["message"],
            rotation=config.rotation,
            retention=config.retention,
        )

    logger.info("Logging initialized")
