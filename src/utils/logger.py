
import logging
import uuid
import json
import sys
import logging.handlers
from pathlib import Path
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Optional
from src.config.settings import get_config, get_secrets


_LOG_DIR: Path = Path("logs")
_RUNS_DIR: Path = _LOG_DIR / "runs"
_DAILY_LOG_FILE: Path = _LOG_DIR / "daily.log"

_LOG_DIR.mkdir(parents=True, exist_ok=True)
_RUNS_DIR.mkdir(parents=True, exist_ok=True)
_DAILY_LOG_FILE.touch(exist_ok=True)

_RUN_START_TIME: str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
_RUN_LOG_FILE: Path = _RUNS_DIR / f"{_RUN_START_TIME}.log"


# Correlation ID for tracing logs across runs
_correlation_id: ContextVar[str] = ContextVar(
    "_correlation_id", default="no-request-context"
)


def set_correlation_id(cid: Optional[str] = None):
    """
    Set a unique correlation ID for the current execution context.
    """
    cid = cid or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid

def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get()


# ANSI colour codes — used only for terminal output in development
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_LEVEL_COLOURS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[35m",    # Magenta
}

# Formatter 1: Coloured output for terminal during development
class ColouredTerminalFormatter(logging.Formatter):
    """
    Coloured, human-readable output for the terminal.
    Each level gets a distinct colour so WARNING and ERROR stand out
    immediately while scanning a busy log stream.
    
    Example output:
      2026-03-22 14:05:31  INFO      src.ingestion.pipeline  Ingesting: hipaa_summary.md [regulations]
      2026-03-22 14:05:33  WARNING   src.query.pipeline      Auth failed: Not enough segments
      2026-03-22 14:05:34  ERROR     src.ingestion.pipeline  Failed to ingest RAG_BENCHMARK.pdf
    """

    def format(self, record: logging.LogRecord) -> str:
        colour  = _LEVEL_COLOURS.get(record.levelname, "")
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")   # Timestamp
        level   = f"{colour}{_BOLD}{record.levelname:<9}{_RESET}"            # Padded, coloured level
        name    = f"{_DIM}{record.name}{_RESET}"                             # Dimmed logger name
        message = record.getMessage()                                         # Log message

        # Append exception traceback on a new line if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return f"{ts}  {level}  {name}  {message}"



# Formatter 2: Plain text for log files, includes correlation ID for tracing
class PlainFileFormatter(logging.Formatter):
    """
    Plain human-readable text for log files.  No ANSI codes (they produce
    garbage in text editors and grep output).  Includes the correlation ID
    so you can trace a single request across thousands of log lines.

    Example output:
      [2026-03-22 14:05:31 UTC] [INFO    ] [src.ingestion.pipeline] Ingesting: hipaa_summary.md  |corr=36f63e2d
      [2026-03-22 14:05:34 UTC] [ERROR   ] [src.ingestion.pipeline] Failed to ingest RAG_BENCHMARK.pdf  |corr=36f63e2d
    """

    def format(self, record: logging.LogRecord) -> str:
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        level   = f"{record.levelname:<8}"                  # Left-padded to 8 chars for alignment
        name    = record.name                               # Logger name e.g. src.ingestion.pipeline
        message = record.getMessage()
        corr    = get_correlation_id()[:8]                  # First 8 chars of correlation ID — enough to identify

        line = f"[{ts}] [{level}] [{name}] {message}  |corr={corr}"

        # Append exception traceback indented below the log line
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line
    

# Formater 3: JSON for structured logging and machine parsing (not implemented yet)
class JsonFormatter(logging.Formatter):
    """
    Single-line JSON for production stdout.
    Compatible with ELK, Splunk, Datadog, and CloudWatch log ingestion.
    Each line is a complete, independently parseable JSON document.
    """

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "level":          record.levelname,
            "logger":         record.name,
            "message":        record.getMessage(),
            "module":         record.module,
            "function":       record.funcName,
            "line":           record.lineno,
            "correlation_id": get_correlation_id(),
        }
        if hasattr(record, "extra"):
            obj.update(record.extra)
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)
    


# Logger setup
def get_logger(name: str) -> logging.Logger:
    """
    Return a named, fully configured logger.

    Three handlers are attached:
      - stdout handler:   coloured text in dev, JSON in production
      - daily file:       plain text, rotates at midnight, 30-day retention
      - per-run file:     plain text, one file per process invocation

    Args:
        name: Always pass __name__ from the calling module so the logger
              hierarchy mirrors the package structure.

    Returns:
        Configured Logger instance (cached — safe to call many times).
    """
    cfg        = get_config()
    secrets    = get_secrets()
    log_level  = getattr(logging, secrets.log_level.upper(), logging.INFO)
    structured = cfg.operations["observability"].get("structured_logging", True)
    is_dev     = secrets.app_env == "development"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger                                 

    logger.setLevel(log_level)

    # ====================
    # Handler 1 — stdout
    # ====================
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)

    if is_dev:
        stdout_handler.setFormatter(ColouredTerminalFormatter())
    elif structured:
        stdout_handler.setFormatter(JsonFormatter())              
    else:
        stdout_handler.setFormatter(PlainFileFormatter())     

    logger.addHandler(stdout_handler)

    # ===================================================================
    # Handler 2 — daily rolling file: logs/app.log
    # Appends all runs on the same calendar day into one file.
    # Rotates at UTC midnight, keeps 30 days of history.
    # Rotated files are named: app.log.2026-03-21, app.log.2026-03-20, …
    # ===================================================================
    daily_handler = logging.handlers.TimedRotatingFileHandler(
        filename    = str(_DAILY_LOG_FILE),     # logs/app.log
        when        = "midnight",               # Rotate once per day
        interval    = 1,                        # Every 1 day
        backupCount = 30,                       # Keep 30 days of rotated files
        encoding    = "utf-8",
        utc         = True,                     # UTC-based rotation timing
    )
    daily_handler.setLevel(log_level)
    daily_handler.setFormatter(PlainFileFormatter())  
    daily_handler.suffix = "%Y-%m-%d"     

    logger.addHandler(daily_handler)

    # ===================================================================
    # Handler 3 — per-run file: logs/runs/YYYY-MM-DD_HH-MM-SS.log
    # Created once when the process starts (_RUN_LOG_FILE is module-level).
    # Never overwritten — each invocation gets its own timestamped file.
    # Mode "a" (append) is safe even if two processes start in the same second.
    # ===================================================================
    run_handler = logging.FileHandler(
        filename = str(_RUN_LOG_FILE),          # logs/runs/2026-03-22_14-05-31.log
        mode     = "a",                         # Append — never truncate
        encoding = "utf-8",
    )
    run_handler.setLevel(log_level)
    run_handler.setFormatter(PlainFileFormatter())

    logger.addHandler(run_handler)

    logger.propagate = False                    # Prevent double-logging via root logger
    return logger