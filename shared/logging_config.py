import logging
import os

def setup_logger(name: str, log_file: str = "agent.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger