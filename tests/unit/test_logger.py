"""Tests for the AI Trading Lab logging utilities."""

from __future__ import annotations

import logging

import pytest

from utils.logger import get_logger


@pytest.fixture(autouse=True)
def cleanup_test_loggers():
    """Remove test loggers and their handlers after each test."""
    yield

    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("test_"):
            logger = logging.getLogger(name)

            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def test_get_logger_returns_logger():
    """get_logger should return a logging.Logger instance."""
    logger = get_logger("test_return_type")

    assert isinstance(logger, logging.Logger)


def test_logger_level_is_info():
    """Loggers should operate at INFO level by default."""
    logger = get_logger("test_level")

    assert logger.level == logging.INFO


def test_logger_has_console_and_file_handlers():
    """Logger should have exactly one console and one file handler."""
    logger = get_logger("test_handlers")

    assert len(logger.handlers) == 2
    assert any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    assert any(
        isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )


def test_logger_propagation_is_disabled():
    """Logger propagation should be disabled."""
    logger = get_logger("test_propagation")

    assert logger.propagate is False


def test_repeated_calls_do_not_create_duplicate_handlers():
    """Calling get_logger repeatedly should reuse the same configuration."""
    logger1 = get_logger("test_duplicate_handlers")
    logger2 = get_logger("test_duplicate_handlers")

    assert logger1 is logger2
    assert len(logger1.handlers) == 2
    assert len(logger2.handlers) == 2


def test_handlers_have_formatter():
    """Both handlers should have the expected formatter."""
    logger = get_logger("test_formatter")

    expected_format = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    for handler in logger.handlers:
        assert handler.formatter is not None
        assert handler.formatter._fmt == expected_format


def test_log_message_is_written_to_file():
    """A log message should be written to the central log file."""
    logger = get_logger("test_file_write")
    message = "Test message written to the log file."

    logger.info(message)

    from config.paths import LOGS_DIR

    log_file = LOGS_DIR / "ai_trading_lab.log"

    assert log_file.exists()
    assert message in log_file.read_text(encoding="utf-8")


def test_multiple_loggers_use_same_log_file():
    """Different loggers should write to the same central log file."""
    logger_a = get_logger("test_logger_a")
    logger_b = get_logger("test_logger_b")

    logger_a.info("Message from logger A")
    logger_b.info("Message from logger B")

    from config.paths import LOGS_DIR

    log_file = LOGS_DIR / "ai_trading_lab.log"
    content = log_file.read_text(encoding="utf-8")

    assert "Message from logger A" in content
    assert "Message from logger B" in content