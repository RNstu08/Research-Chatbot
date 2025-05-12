# Logging setup
import logging
import sys
from logging.handlers import RotatingFileHandler
# Corrected import path:
# We assume that when run.py executes, the 'research_chatbot' directory 
# (or its parent if run.py is in the root) is added to PYTHONPATH,
# or that Python's module resolution can find 'config' from 'app.utils'.
# A common way is to ensure your project root (research_chatbot) is the working directory.
from config import settings


def setup_logging():
    """Sets up logging configuration for the application."""

    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    logger = logging.getLogger() 
    logger.setLevel(settings.LOG_LEVEL.upper())

    # Clear existing handlers (if any, e.g., during hot-reloading in Streamlit)
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        settings.LOG_FILE, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info("Logging configured successfully.")


def get_logger(name: str) -> logging.Logger:
    """Utility function to get a logger instance for a specific module."""
    return logging.getLogger(name)