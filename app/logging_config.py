import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone


_LOG_RECORD_BUILTINS = frozenset(logging.makeLogRecord({}).__dict__.keys())

class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge only extra fields passed via extra={...}, skipping all built-in LogRecord attrs
        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_BUILTINS:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    from app.config import get_settings
    settings = get_settings()

    handlers: list[logging.Handler] = []

    # Stdout — always JSON
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(_JsonFormatter())
    handlers.append(stdout_handler)

    # Timed rotating file handler
    import pathlib
    pathlib.Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f"{settings.log_dir}/app.log",
        when="midnight",
        backupCount=30,  # keep 30 days of logs
        encoding="utf-8",
        utc=True,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(_JsonFormatter())
    handlers.append(file_handler)

    logging.basicConfig(level=settings.log_level.upper(), handlers=handlers, force=True)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "openai", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
