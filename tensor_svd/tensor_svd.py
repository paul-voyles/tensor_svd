import logging

import numpy as np
from scipy.sparse.linalg import eigs

_logger = logging.getLogger(__name__)


def svd_HO(data, rank, max_iter=100):
    """ Preforms a higher order SVD on some tensor with some rank defined by rank

    Parameters
    ----------
    data : numpy array
        R^n X N array of input data (R^n variables, N trials)
        All spatially uncorrelated axes are unwrapped.
    rank : numpy array
        The an array with length n which describes the eigenvectors for each dimension

    Returns
    -------

    factors : numpy array
    loadings : numpy array
    explained_variance : numpy array
    mean : numpy array or None (if center is None)
    """
    data_shape = np.shape(data)
    if len(data_shape) != len(rank):
        print("The rank should be the same size as the data shape")

    data0 = data   # might have to change to limit copying
    data_shape = np.shape(data) # p0
    dimensions = len(data_shape)        # d
    X = data # Data copy to reduce
    u_total = [0,0,0]
    ordered_indexes = np.argsort(data_shape) # getting the indicies from min len to max

    for k in ordered_indexes: # calculating initial SVD
        unfolded = unfold_axis(data, k) # unfolding along the minimum axis
        if np.shape(unfolded)[0] > np.shape(unfolded)[1]: # if the unfolded axis is less than the unfolded
            u_index = svd(unfolded, rank[k]) # svd along the min axis length
        else:
            u_index = eigs(A=unfolded*unfolded.T, k =rank[k])
        u_total[k] = u_index
        X = np.tensordot(X, u_index, axes=k) # This needs to be fixed!

    iter_count = 0

    while iter_count < max_iter: # converging the different SVD decompositions.
        iter_count += 1
        for k in range(0, dimensions):
            Y = data
            minus_k = range(0,dimensions).remove(k)  # every value except for k
            for j in minus_k:
                Y = np.tensordot(Y, u_total[j], axes=j)
            MY = unfold_axis(Y, k)
            u_index[k] = svd(MY, rank[k])

    X = data
    for k in ordered_indexes:
        X = np.tensordot(X,u_index[k], k)

    for k in range(0,dimensions):
        X = np.tensordot(X,u_index[k], k)

    return X