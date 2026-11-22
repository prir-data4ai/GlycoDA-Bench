"""Shared colored-logging utilities for the benchmark suite."""
from __future__ import annotations

import logging
import sys

# ── ANSI color codes ──────────────────────────────────────────────
_RESET = "\033[0m"

_LEVEL_COLORS = {
    logging.DEBUG:    "\033[36m",   # Cyan
    logging.INFO:     "\033[32m",   # Green
    logging.WARNING:  "\033[33m",   # Yellow
    logging.ERROR:    "\033[31m",   # Red
    logging.CRITICAL: "\033[1;31m", # Bold Red
}

# Semantic tag colors (for _make_run_logger style print output)
TAG_COLORS = {
    "run":       "\033[34m",   # Blue
    "preflight": "\033[36m",   # Cyan
    "plan":      "\033[35m",   # Magenta
    "step":      "\033[32m",   # Green
    "replan":    "\033[33m",   # Yellow
    "exec":      "\033[32m",   # Green
    "warn":      "\033[33m",   # Yellow
    "error":     "\033[31m",   # Red
    "fail":      "\033[31m",   # Red
    "done":      "\033[1;32m", # Bold Green
    "skip":      "\033[33m",   # Yellow
    "retry":     "\033[33m",   # Yellow
    "metric":    "\033[36m",   # Cyan
}


def _is_color_terminal() -> bool:
    """Return True when stderr is a real color-capable terminal."""
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class ColorFormatter(logging.Formatter):
    """Logging formatter that adds ANSI colors based on log level.

    Automatically disables colors when output is not a terminal (e.g. piped to file).
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_color: bool | None = None,
    ):
        super().__init__(fmt, datefmt)
        self.use_color = use_color if use_color is not None else _is_color_terminal()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            # Create a copy to avoid mutating the original record
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{color}{record.levelname}{_RESET}"
            record.msg = f"{color}{record.msg}{_RESET}"
        return super().format(record)


def configure_colored_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with colored output.

    Safe to call from both the benchmark driver and the single-model subprocess.
    """
    use_color = _is_color_terminal()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_color=use_color,
        )
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


def colorize_tag(message: str, use_color: bool | None = None) -> str:
    """Add color to a ``[tag]`` prefixed log message for print-style loggers.

    Example::

        colorize_tag("[run] model=qwen3 dataset=gf_001")
        # → "\\033[34m[run] model=qwen3 dataset=gf_001\\033[0m"

    Returns the original message unchanged when colors are disabled.
    """
    if use_color is None:
        use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not use_color:
        return message

    # Extract the tag from "[tag]" prefix
    if message.startswith("["):
        bracket_end = message.find("]")
        if bracket_end > 0:
            tag = message[1:bracket_end].lower()
            color = TAG_COLORS.get(tag, "")
            if color:
                return f"{color}{message}{_RESET}"
    return message
