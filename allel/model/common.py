import numpy as np


def check_ndim(x, n):
    if x.ndim != n:
        raise ValueError('argument must have {} dimensions, found {}'.format(n, x.ndim))


def check_dtype(x, t):
    t = np.dtype(t)
    if x.dtype != t:
        raise ValueError('argument must have dtype {}, found {}'.format(t, x.dtype))


