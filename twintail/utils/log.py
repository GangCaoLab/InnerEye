import typing as t

MSG_FMT = "%(name)-20s %(levelname)-7s @ %(asctime)s: %(message)s"
DATE_FMT = "%m/%d/%y %H:%M:%S"


def set_global_logging(log_level=10, log_file=None,
                       msg_fmt: str = MSG_FMT,
                       date_fmt: str = DATE_FMT):
    """Set the global logging formats.

    :param log_level: logging level.
    :param log_file: If specified will set a file handler.
    :param msg_fmt: Message format for logging.
    :param date_fmt: Date format in message.
    :return:
    """
    import sys
    import logging
    log = logging.getLogger()
    formatter = logging.Formatter(
        fmt=msg_fmt,
        datefmt=date_fmt
    )
    if not any([isinstance(h, logging.StreamHandler) for h in log.handlers]):
        s_handler = logging.StreamHandler(sys.stderr)
        s_handler.setFormatter(formatter)
        log.addHandler(s_handler)
    if log_file:
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(formatter)
        log.addHandler(f_handler)
    log.setLevel(log_level)


def print_arguments(print_func: t.Callable = print):
    """Print the arguments passed in a function.

    Use inside the function or method, for example:

    >>> def func1(a):
    ...     print_arguments()
    ...     return a

    :param print_func: Callable object used for print the arguments.
    :return:
    """
    import inspect
    f_back = inspect.currentframe().f_back
    func_name = f_back.f_code.co_name
    locals_ = f_back.f_locals
    if 'self' in locals_:
        func = getattr(locals_['self'], func_name)
    else:
        func = f_back.f_globals[func_name]
    sig = inspect.signature(func)

    is_init = ('self' in locals_) and (func_name == '__init__')
    if is_init and sig.parameters:
        cls_name = type(locals_['self']).__name__
        head = f"Initialize {cls_name} with arguments:"
    elif is_init:
        cls_name = type(locals_['self']).__name__
        head = f"Initialize {cls_name}."
    elif sig.parameters:
        head = f"Do {func.__name__} with arguments:"
    else:
        head = f"Do {func.__name__}."

    params = ""
    for name in sig.parameters:
        params += f"\n\t{name} = {repr(locals_[name])}"

    msg = head + params
    print_func(msg)

