# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for Sarathi."""
import logging
import sys

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("Sim")
_default_handler = None

def _setup_logger(log_file: str = None): # If need log to file please apply to _setup_logger() before init_logger()
    _root_logger.setLevel(logging.DEBUG)

    global _default_handler
    if _default_handler is None:
        # Console handler
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush
        _default_handler.setLevel(logging.INFO)
        fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
        _default_handler.setFormatter(fmt)
        _root_logger.addHandler(_default_handler)

    # Add file handler if requested (only once)
    if log_file and not any(isinstance(h, logging.FileHandler) for h in _root_logger.handlers):
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        _root_logger.addHandler(fh)

    # Avoid double logging
    _root_logger.propagate = False

# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
# _setup_logger()
# logging.basicConfig(
#     level=logging.DEBUG,
#     format=_FORMAT
# )

def init_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
