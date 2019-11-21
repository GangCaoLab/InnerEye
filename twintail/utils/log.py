
LOGGING_FMT = "%(name)-20s %(levelname)-7s @ %(asctime)s: %(message)s"
LOGGING_DATE_FMT = "%m/%d/%y %H:%M:%S"


def set_logging_fmt(log_level=10, log_file=None):
    import sys
    import logging
    log = logging.getLogger()
    s_handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        fmt=LOGGING_FMT,
        datefmt=LOGGING_DATE_FMT
    )
    s_handler.setFormatter(formatter)
    log.addHandler(s_handler)
    if log_file:
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(formatter)
        log.addHandler(f_handler)
    log.setLevel(log_level)

