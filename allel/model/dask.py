import dask.array as da


from .common import check_dtype, check_ndim
from . import ndarray as _ndarray


def check_array_type(x):
    if not isinstance(x, da.Array):
        raise ValueError('argument must be a dask array, found {}'.format(type(x)))


def check_genotype_array(g):
    check_array_type(g)
    check_ndim(g, 3)
    check_dtype(g, 'i1')


def genotype_array_is_called(g):
    check_genotype_array(g)
    out = da.map_blocks(_ndarray.genotype_array_is_called, g, drop_axis=2)
    return out
