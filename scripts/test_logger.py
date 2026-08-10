"""
Simple script to verify the logging system.
"""

from utils.logger import get_logger


logger = get_logger(__name__)

logger.debug("This is a DEBUG message.")
logger.info("This is an INFO message.")
logger.warning("This is a WARNING message.")
logger.error("This is an ERROR message.")
logger.critical("This is a CRITICAL message.")