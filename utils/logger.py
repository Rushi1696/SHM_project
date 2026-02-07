"""Simple logger wrapper."""

import logging


def setup_logger(name=None, level=logging.INFO):
    """Setup and return a logger"""
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def get_logger(name=None):
    """Get or create a logger"""
    return setup_logger(name)
