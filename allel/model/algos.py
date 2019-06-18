import numpy as np
import numba


def check_ndarray(x):
    if not isinstance(x, np.ndarray):
        raise ValueError('argument must be a numpy array, found {}'.format(type(x)))


def check_ndim(x, n):
    if x.ndim != n:
        raise ValueError('argument must have {} dimensions, found {}'.format(n, x.ndim))


def check_dtype(x, t):
    t = np.dtype(t)
    if x.dtype != t:
        raise ValueError('argument must have dtype {}, found {}'.format(t, x.dtype))


def check_genotype_array(g):
    check_ndarray(g)
    check_ndim(g, 3)
    check_dtype(g, 'i1')


def genotype_array_is_called(g):
    check_genotype_array(g)
    out = np.ones(g.shape[:2], dtype=bool)
    _genotype_array_is_called(g, out)
    return out


@numba.jit(
    numba.void(numba.int8[:, :, :], numba.boolean[:, :]),
    nopython=True,
    nogil=True)
def _genotype_array_is_called(g, out):
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
    check_genotype_array(g)
    out = np.zeros(g.shape[:2], dtype=bool)
    _genotype_array_is_missing(g, out)
    return out


@numba.jit(
    numba.void(numba.int8[:, :, :], numba.boolean[:, :]),
    nopython=True,
    nogil=True)
def _genotype_array_is_missing(g, out):
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
    check_genotype_array(g)
    out = np.ones(g.shape[:2], dtype=bool)
    _genotype_array_is_hom(g, out)
    return out


@numba.jit(
    numba.void(numba.int8[:, :, :], numba.boolean[:, :]),
    nopython=True,
    nogil=True)
def _genotype_array_is_hom(g, out):
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
    check_genotype_array(g)
    out = np.zeros((g.shape[0], max_allele + 1), dtype='i4')
    _genotype_array_count_alleles(g, max_allele, out)
    return out


@numba.jit(
    numba.void(numba.int8[:, :, :], numba.int8, numba.int32[:, :]),
    nopython=True,
    nogil=True)
def _genotype_array_count_alleles(g, max_allele, out):
    n_variants = g.shape[0]
    n_samples = g.shape[1]
    ploidy = g.shape[2]
    for i in range(n_variants):
        for j in range(n_samples):
            for k in range(ploidy):
                allele = g[i, j, k]
                if 0 <= allele <= max_allele:
                    out[i, allele] += 1
