import numpy as np
import dask.array as da


from .common import check_dtype, check_ndim
from . import ndarray as _ndarray, dask as _dask


def is_hdf5_like(x):
    return (hasattr(x, 'ndim') and
            hasattr(x, 'dtype') and
            hasattr(x, 'shape') and
            hasattr(x, 'chunks') and
            len(x.chunks) == x.ndim)


def normalize_genotype_array_data(x):
    if isinstance(x, np.ndarray):
        x = x.astype('i1', casting='safe', copy=False)
    elif isinstance(x, da.Array):
        pass
    elif is_hdf5_like(x):
        x = da.from_array(x, chunks=x.chunks)
    else:
        x = np.asarray(x).astype('i1', casting='safe', copy=False)
    check_ndim(x, 3)
    check_dtype(x, 'i1')
    return x


class GenotypeArray(object):

    def __init__(self, data):
        self._data = normalize_genotype_array_data(data)
        if isinstance(self._data, np.ndarray):
            self._algos = _ndarray
        elif isinstance(self._data, da.Array):
            self._algos = _dask

    @property
    def data(self):
        return self._data

    def is_called(self):
        """TODO docstring"""
        return self._algos.genotype_array_is_called(self._data)
