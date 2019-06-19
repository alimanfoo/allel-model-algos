"""This is a low-level module providing routines for basic data manipulations
on numpy arrays."""


import numpy as np
import numba


def require_genotype_ndarray(g):
    if not isinstance(g, np.ndarray):
        raise TypeError(
            'Bad type, expected numpy array, found {}.'.format(type(g)))
    if g.dtype != np.dtype('i1'):
        raise TypeError(
            'Bad dtype, expected int8, found {}.'.format(g.dtype))
    if g.ndim != 3:
        raise ValueError(
            'Bad shape, expected 3 dimensions, found {}.'.format(g.ndim))


def genotype_array_is_called(g):
    require_genotype_ndarray(g)
    out = np.ones(g.shape[:2], dtype=bool)
    genotype_array_is_called_impl(g, out)
    return out


@numba.njit(
    numba.void(numba.int8[:, :, :], numba.boolean[:, :]),
    nogil=True)
def genotype_array_is_called_impl(g, out):
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    ploidy = g.shape[2]
    for i in range(n_variants):
        for j in range(n_samples):
            for k in range(ploidy):
                if g[i, j, k] < 0:
                    out[i, j] = False
                    # no need to check other alleles
                    continue


def genotype_array_is_missing(g):
    require_genotype_ndarray(g)
    out = np.zeros(g.shape[:2], dtype=bool)
    genotype_array_is_missing_impl(g, out)
    return out


@numba.njit(
    numba.void(numba.int8[:, :, :], numba.boolean[:, :]),
    nogil=True)
def genotype_array_is_missing_impl(g, out):
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    ploidy = g.shape[2]
    for i in range(n_variants):
        for j in range(n_samples):
            for k in range(ploidy):
                if g[i, j, k] < 0:
                    out[i, j] = True
                    # no need to check other alleles
                    continue


def genotype_array_is_hom(g):
    require_genotype_ndarray(g)
    out = np.ones(g.shape[:2], dtype=bool)
    genotype_array_is_hom_impl(g, out)
    return out


@numba.njit(
    numba.void(numba.int8[:, :, :], numba.boolean[:, :]),
    nogil=True)
def genotype_array_is_hom_impl(g, out):
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    ploidy = g.shape[2]
    for i in range(n_variants):
        for j in range(n_samples):
            first_allele = g[i, j, 0]
            if first_allele < 0:
                out[i, j] = False
            else:
                for k in range(1, ploidy):
                    if g[i, j, k] != first_allele:
                        out[i, j] = False
                        # no need to check other alleles
                        continue


def genotype_array_count_alleles(g, max_allele):
    require_genotype_ndarray(g)
    # TODO max_allele require int
    out = np.zeros((g.shape[0], max_allele + 1), dtype='i4')
    genotype_array_count_alleles_impl(g, max_allele, out)
    return out


@numba.njit(
    numba.void(numba.int8[:, :, :], numba.int8, numba.int32[:, :]),
    nogil=True)
def genotype_array_count_alleles_impl(g, max_allele, out):
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    ploidy = g.shape[2]
    for i in range(n_variants):
        for j in range(n_samples):
            for k in range(ploidy):
                allele = g[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1
