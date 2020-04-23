import logging

import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.utils.extmath import randomized_svd as fast_svd
from tensor_svd_support_func import unfold_axis, ttm


_logger = logging.getLogger(__name__)


def svd_HO(data, rank, max_iter=10):
    """ Preforms a higher order SVD on some tensor with some rank defined by rank

    Parameters
    ----------
    data : numpy array
        Higher-order noisy tensor with size (I1 , I2 , ... , IN) along N dimensions. N >1.
    rank : numpy array
        Integer array (R1, R2, R3, ... , Rn) describes the eigenvectors for each dimension
    max_iter: integer
        Number of iterations for higher-order orthogonal iteration (HOOI) algorithm

    Returns
    -------

    X: numpy array
        Denoised tensor with low rank and the same size of input data.
    U: list of numpy array
        List of orthogonal matrix for each dimension, each component in the list has size Ik x Rk.
    S: numpy array
        Core tensor of the low rank tensor X with size (R1, R2, R3, ... , Rn).
    """
    data_shape = np.shape(data)         # p0
    if len(data_shape) != len(rank):
        print("The rank should be the same size as the data shape")

    dimensions = len(data_shape)        # d
    ordered_indexes = np.argsort(data_shape) # getting the indicies from min len to max, initialization starts from smallest size

    ## Initialize U and Y with SVD
    U = [None] * dimensions # Generate an empty array to save all the U matrices with fixed length
    X = data
    for k in ordered_indexes: # calculating initial SVD
        unfolded = unfold_axis(X, k) # unfolding from the axis with minimum size
        [U[k], _ , _] = fast_svd(unfolded,rank[k])
        X = ttm(X, np.transpose(U[k]), k) # This needs to be fixed!

    ## Update U with HOOI
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        for k in range(0, dimensions):
            Y = data
            minus_k = list(range(0,dimensions))
            minus_k.remove(k)  # every value except for k, seems do it in one step will remove all the elements in the list.
            for j in minus_k:
                Y = ttm(Y, np.transpose(U[j]), j)
            MY = unfold_axis(Y, k)
            [U[k], _, _] = fast_svd(MY, rank[k])

    ## Use the determined U matrices to calculate core tensor and denoised tensor
    X = data
    for k in ordered_indexes:
        X = ttm(X,np.transpose(U[k]), k)  # Check this part.
    S = X   # core tensor
    for k in range(0,dimensions):
        X = ttm(X,U[k], k)

    return X, U, S