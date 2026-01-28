import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from config.settings import settings # Import settings instance

def setup_logging():
    """Configures logging for the application."""

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates if called multiple times
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    # Set console handler to only show WARNING and above
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    # --- File Handler (Optional, based on config) ---
    if settings.log_dir:
        try:
            os.makedirs(settings.log_dir, exist_ok=True)
            log_file = os.path.join(settings.log_dir, 'ethics_engine.log')

            # Rotating file handler (e.g., max 5MB per file, keep 5 backups)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setFormatter(log_format)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
            logger.info(f"Logging configured. Level: {settings.log_level}. Log file: {log_file}")

        except Exception as e:
            logger.error(f"Failed to configure file logging to {settings.log_dir}: {e}", exc_info=True)
            # Continue with console logging only
    else:
         logger.info(f"Logging configured. Level: {settings.log_level}. Console only (no log_dir specified).")


# Example usage (typically called once at application startup)
# if __name__ == "__main__":
#     setup_logging()
#     logging.info("Logging setup complete.")
#     logging.debug("This is a debug message.")
#     logging.warning("This is a warning.")
