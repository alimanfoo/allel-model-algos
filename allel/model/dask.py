"""This is a low-level module providing routines for basic data manipulations
on dask arrays."""


import dask.array as da
import numpy as np


from . import ndarray as _ndarray


def require_genotype_dask_array(g):
    if not isinstance(g, da.Array):
        raise TypeError(
            'Bad type, expected dask array, found {}.'.format(type(g)))
    if g.dtype != np.dtype('i1'):
        raise TypeError(
            'Bad dtype, expected int8, found {}.'.format(g.dtype))
    if g.ndim != 3:
        raise ValueError(
            'Bad shape, expected 3 dimensions, found {}.'.format(g.ndim))


def genotype_array_is_called(g):
    require_genotype_dask_array(g)
    out = da.map_blocks(_ndarray.genotype_array_is_called, g, drop_axis=2)
    return out


def genotype_array_is_missing(g):
    require_genotype_dask_array(g)
    out = da.map_blocks(_ndarray.genotype_array_is_missing, g, drop_axis=2)
    return out


def genotype_array_is_hom(g):
    require_genotype_dask_array(g)
    out = da.map_blocks(_ndarray.genotype_array_is_hom, g, drop_axis=2)
    return out


def genotype_array_count_alleles(g, max_allele):
    require_genotype_dask_array(g)
    # TODO max_allele require int

    # determine output chunks - preserve axis 0; change axis 1, axis 2
    chunks = (g.chunks[0], (1,) * len(g.chunks[1]), (max_allele + 1,))

    def f(chunk):
        # compute allele counts for chunk
        ac = _ndarray.genotype_array_count_alleles(chunk, max_allele)
        # insert extra dimension to allow for summing
        ac = ac[:, None, :]
        return ac

    # map blocks and reduce via sum
    out = (
        da.map_blocks(f, g, chunks=chunks)
        .sum(axis=1, dtype='i4')
    )

    return out
