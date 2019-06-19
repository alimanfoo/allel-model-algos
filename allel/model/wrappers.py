import numpy as np
import dask.array as da


from . import ndarray as _ndarray, dask as _dask


def is_hdf5_like(x):
    return (hasattr(x, 'ndim') and
            hasattr(x, 'dtype') and
            hasattr(x, 'shape') and
            hasattr(x, 'chunks') and
            len(x.chunks) == x.ndim)


_further_advice = "For further advice, please raise an issue on GitHub at " \
                  "the following link: @@TODO URL."


def ensure_array_data(data):
    """Here we are fairly strict about the type of the incoming `data`
    argument, accepting a limited number of types, and raising an error if
    the type is different. We don't try to convert `data` to an accepted type
    ourselves, e.g., by calling `np.asarray(data)`, because this could lead
    to cryptic errors downstream. Rather we prefer to make the user aware
    if a type conversion is necessary, and rely on them to perform the type
    conversion in their code."""

    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, da.Array):
        return data
    elif is_hdf5_like(data):
        return da.from_array(data, chunks=data.chunks)
    else:
        raise ValueError(
            'The `data` argument must be a numpy array, or a dask array, or an '
            'h5py dataset, or a zarr array. The type of the `data` argument '
            'received was {}. It may be possible to convert the `data` '
            'argument to an accepted type, e.g., by calling `np.asarray('
            'data)`, but this may not work or may not be appropriate '
            'for large datasets. '
            .format(type(data)) + _further_advice
        )


def normalize_genotype_array_data(data):

    # ensure we have an accepted array type
    data = ensure_array_data(data)

    # check number of dimensions
    if data.ndim != 3:
        raise ValueError(
            'The `data` argument must be an array with 3 dimensions. The '
            '`data` argument received has {} dimensions. '
            .format(data.ndim) + _further_advice
        )

    # check dtype
    if data.dtype != np.dtype('i1'):
        raise ValueError(
            'The `data` argument must be an array with single byte integer ('
            'int8) dtype. The `data` argument receved has dtype {}. '
            .format(data.dtype) + _further_advice
        )

    return data


class GenotypeArray(object):
    """TODO docstring"""

    def __init__(self, data):
        """TODO docstring"""
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
