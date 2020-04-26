import sys
import logging
import warnings


def get_logger(name):
    """
    Return a logger for current module.

    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                  datefmt="%Y-%m-%d - %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logfile = logging.FileHandler('run.log', 'w')
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)

    return logger


def disable_warnings():
    """
    Disable printing of warnings

    Returns
    -------
    None
    """
    warnings.simplefilter("ignore")
