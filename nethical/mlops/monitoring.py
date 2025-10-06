import logging
import os
import json

def setup_logger(logfile="logs/mlops.log", to_console=True):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is reused
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

        # Optional: Console handler
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(console_handler)
    return logger

def get_logger():
    return logging.getLogger("mlops")

def log_event(event, **kwargs):
    logger = get_logger()
    # Serialize kwargs for clarity
    logger.info(f"{event}: {json.dumps(kwargs)}" if kwargs else event)

# Example usage in training scripts:
# from nethical.mlops.monitoring import setup_logger, log_event
# setup_logger()  # Call once at the entrypoint of your app
# log_event("train_start", params=params)
