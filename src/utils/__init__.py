import logging
import os
import sys

def setup_logging(level=None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, checks LOG_LEVEL environment variable.
               Defaults to INFO if neither is set.
    """
    if not logging.root.handlers:
        # Get level from parameter, environment variable, or default
        if level is None:
            level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
            level = getattr(logging, level_str, logging.INFO)
        elif isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

# Setup logging on import (can be overridden later)
setup_logging()

# Create and export a module-level logger instance
LOGGER = logging.getLogger(__name__)


