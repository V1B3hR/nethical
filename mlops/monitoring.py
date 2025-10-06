import logging

def setup_logger(logfile="logs/mlops.log"):
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger("mlops")

logger = setup_logger()

def log_event(event, **kwargs):
    logger.info(f"{event}: {kwargs}")

# Example usage in training scripts:
# from nethical.mlops.monitoring import log_event
# log_event("train_start", params=params)
