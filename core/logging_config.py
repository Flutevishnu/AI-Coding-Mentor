import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.config import settings

_configured = False


def setup_logging() -> logging.Logger:
    """Configure root logging with a per-environment rotating file handler.

    Logs are written to ``<LOG_DIR>/<ENV>.log`` (e.g. ``logs/dev.log`` or
    ``logs/prod.log``) so development and production output never mix. A console
    handler is also attached for convenience.
    """
    global _configured

    logger = logging.getLogger("mentor")
    if _configured:
        return logger

    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{settings.ENV}.log"

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _configured = True
    logger.info("Logging initialized | env=%s | file=%s", settings.ENV, log_file)
    return logger


logger = setup_logging()
