import logging


def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with the specified name and level.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int
        The logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = (
        False  # Prevents the log messages from being propagated to the root logger
    )

    return logger
