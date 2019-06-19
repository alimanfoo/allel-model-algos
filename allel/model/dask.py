import dask.array as da


from . import ndarray as _ndarray


def genotype_array_is_called(g):
    out = da.map_blocks(_ndarray.genotype_array_is_called, g, drop_axis=2)
    return out
