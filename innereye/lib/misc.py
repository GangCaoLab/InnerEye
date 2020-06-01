import typing as t
import inspect

from collections import Iterable
import numpy as np


def flatten(items,
            ignore_tps=(str, bytes, np.ndarray)):
    """Flatten a Iterable object.

    :param items: Iterable object need to flatten.
    :param ignore_tps: Types not consider as iterable.
    :return: Generator yield values in flattened object.
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_tps):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def caller_caller() -> t.Tuple[t.Callable, t.Dict]:
    """Get caller's caller"""
    f_cc = inspect.currentframe().f_back.f_back
    func_name = f_cc.f_code.co_name
    locals_ = f_cc.f_locals
    if 'self' in locals_:
        func = getattr(locals_['self'], func_name)
    else:
        func = f_cc.f_globals[func_name]
    return func, locals_


def local_arguments(keywords=True) -> t.Union[t.Tuple, t.Dict]:
    """Get local arguments of a function.
    Need used within a function or method.

    :param keywords:
    :return: Tuple of arguments values or dict of (keywords, value) pair.
    """
    func, locals_ = caller_caller()
    sig = inspect.signature(func)

    names = list(sig.parameters)
    values = [locals_.get(n) for n in names]
    if keywords:
        return {n: v for n, v in zip(names, values)}
    else:
        return tuple(values)

