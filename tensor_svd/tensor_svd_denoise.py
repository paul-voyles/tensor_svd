import logging

import numpy as np
from sklearn.utils.extmath import randomized_svd as fast_svd

_logger = logging.getLogger(__name__)

def tensor_svd_denoise(data, rank):
    """ Wrapper to pre-process STEM data for tensor SVD denoise

    Parameters
    ----------
    data : numpy array
        3D or 4D noisy input data with first two dimensions being navigation dimensions.
        3D data for hyperspectral data or 4D data with reciprocal space unfolded into one dimension, 4D data for original 4D STEM data.
    rank: numpy array
        Integer array (R1, R2, R3) denoise ranks for hyperspectral data or 4D STEM data, R1, R2 for real space dimensions, R3 for energy or k dimension.
        Three elements for both 3D and 4D input data.
        If rank is empty, scree_plot function will be called to run scree tests and generate scree plots to help user determine ranks to use..

    Returns
    -------

    X: numpy array
        Denoised data with low rank and the same size of input data.


    Future to do list:
    1. Roughly estimate noise level and determine the number of iterations to use in HOOI algorithm.
    2. Atuomatically determine denoising ranks from eigenvalues.
    """
    # Case when SVD ranks are fed in the input, call svd_HO function to denoise

    if len(data.shape) == 3:  # hyperspectral data case, directly feed data to svd_HO function
        [X, _, _] = svd_HO(data, rank)
    if len(data.shape) == 4:    # Original 4D STEM data case, unfold reciprocal space dimensions into one dimension then feed to svd_HO function
        data = np.reshape(data, [data.shape[0], data.shape[1], data.shape[2]*data.shape[3]])
        [X, _, _] = svd_HO(data, rank)
    
    return X



def svd_HO(data, rank, max_iter=10):
    """ HOOI method to decompose high order tensor with given ranks

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
    svd_iter = 10
    data_shape = np.shape(data)         # p0

    # Check that number of dimensions match the number of rank numbers
    if len(data_shape) != len(rank):
        print("The rank should be the same size as the data shape")
        return data, [], []

    # Check that for each rank, the product of all the rest ranks are larger than this rank
    for k in range(len(rank)):
        prod = 1
        for i in range(len(rank)):
            if i != k:
                prod = prod * rank[i]
        if rank[k] > prod:
            print("The rank does not satisfy requirment of HOOI.")
            return data, [], []

    dimensions = len(data_shape)        # d
    ordered_indexes = np.argsort(data_shape) # getting the indicies from min len to max, initialization starts from smallest size

    ## Initialize U and Y with SVD
    U = [None] * dimensions # Generate an empty array to save all the U matrices with fixed length
    X = data
    for k in ordered_indexes: # calculating initial SVD
        unfolded = unfold_axis(X, k) # unfolding from the axis with minimum size
        [U[k], _ , _] = fast_svd(unfolded,rank[k],n_iter=svd_iter)
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
            [U[k], _, _] = fast_svd(MY, rank[k],n_iter=svd_iter)

    ## Use the determined U matrices to calculate core tensor and denoised tensor
    X = data
    for k in ordered_indexes:
        X = ttm(X,np.transpose(U[k]), k)  # Check this part.
    S = X   # core tensor
    for k in range(0,dimensions):
        X = ttm(X,U[k], k)

    return X, U, S

def weighted_tensor(data):
    """ weighting the whole tensor along each dimension following Jones et al., J. Chemometrics 22, 482, 2008 designed for Poisson noise.

    Parameters
    ----------
    data : numpy array
        Higher-order tensor with size (I1, I2, ... , IN) along N dimensions

    Returns
    -------
    data_weighted : numpy array
        Higher-order weighted tensor with size (I1, I2, ... , IN) along N dimensions

    weights: list of numpy arrays
        A list of N numpy arrays, each one is a 1D vector with length Ik, saving the weights along dimension k. Empty list will be returned if reverse_option = 1.

    """

    total_dim = len(data.shape)

    # Create weights tensor that has the same shape as the input data and save the weights at each element
    weights = np.ones(data.shape)

    # calculate the weight vector along each dimension
    for i in range(total_dim):
        # prepare dimension list, swap dimension 0 and target dimension i
        dim_list = list(range(0,total_dim))
        dim_list[0], dim_list[i] = dim_list[i], dim_list[0]
        dim_order = tuple(dim_list)

        # unfold the data and weight tensor into a 2D matrix
        data_unfold = np.transpose(data, dim_order)
        weights_unfold = np.transpose(weights, dim_order)
        new_shape = data_unfold.shape
        data_unfold = np.reshape(data_unfold,[data.shape[i],int(data.size/data.shape[i])])
        weights_unfold = np.reshape(weights_unfold,[data.shape[i],int(data.size/data.shape[i])])

        # calculate the weight vector (length ki) for target dimension using the unfolded data.
        temp = np.mean(data_unfold, axis = 1)
        temp = temp / np.sum(temp)
        
        # multiply the unfolded weight matrix then reshape back to the original shape
        weights_unfold = weights_unfold * temp.reshape([-1,1])
        weights_unfold = np.reshape(weights_unfold, new_shape)
        weights = np.transpose(weights_unfold, dim_order)

    data = np.multiply(data, weights)
    return data, weights

def unfold_axis(data, k):

    """ return matrix representation of a higher-order tensor along certain dimension

    Parameters
    ----------
    data : numpy array
        Higher-order tensor with size (I1 , I2 , ... , IN) along N dimensions.
    k : integer number
        Dimension index k (0 - N-1) along which the tensor will be unfolded

    Returns
    -------

    data_unfold : numpy array
        2D array with size (Ik, (I1 x I2 x ... x Ik-1 x Ik+1 x ... x IN))
        The second dimension is arranged in the order of k+1, k+2, ..., N, 1, 2, ..., k-1

    """

    target_dim = k
    total_dim = len(data.shape)

    dim_list = []
    for i in range(total_dim):
        dim_list.append((target_dim - i) % total_dim)
    dim_order = tuple(dim_list)

    data_unfold = np.transpose(data,dim_order)
    data_unfold = np.reshape(data_unfold,[data.shape[k],int(data.size/data.shape[k])])
    return data_unfold

def ttm(t, m, k):

    """ Preforms multiplication of a higher-order tensor by a matrix

    Parameters
    ----------
    t : numpy array
        Higher-order tensor with size (I1 , I2 , ... , IN) along N dimensions.
    m : numpy array
        2D matrix with size (Jk, Ik), the size of second dimension must be the same as the kth dimension in t, where k is the dimension index in the third input.
    k : integer
        Specify the dimension of t which will be multiplied by matrix m

    Returns
    -------

    t_mul : numpy array
        Higher-order tensor with size (I1 , I2 , ... , Ik-1, Jk, Ik+1, ... , IN) along N dimensions.
    """

    dim_list = []   # initialize a list to save dimension index to transpose the tensor reshapped from 2D matrix
    shape_list = [] # initialize a list to save the dimensions to reshape 2D matrix back to tensor
    total_dim = len(t.shape)
    for i in range(total_dim):
        dim_list.append((k - i) % total_dim)
        shape_list.append(t.shape[(k - i) % total_dim])
    dim_order = tuple(dim_list)
    shape_list[0] = m.shape[0]

    t_unfold = unfold_axis(t, k)
    t_mul = np.matmul(m, t_unfold)
    t_mul = np.reshape(t_mul,tuple(shape_list))
    t_mul = np.transpose(t_mul, dim_order)

    return t_mul

def scree_plots(t, ndim = []):
    
    """ Performs scree tests for each dimension of the input tensor

    Parameters
    ----------
    t : numpy array
        Higher-order tensor with size (I1 , I2 , ... , IN) along N dimensions.
    ndim : optional, list with N integer elements
        Number of components to calculate along each dimension. If not defined, the maximum size along each dimension will be used.

    Returns
    -------

    scree : list of N numpy arrays
        One array with size ndim[i] for each dimension saving the eigenvalues for this dimension.

    """
    total_dim = len(t.shape)
    if not ndim:   # case with no input ndim
        for i in range(total_dim):
            ndim.append(t.shape[i])
    elif len(ndim) != total_dim:    # case that input ndim does not agree with number of dimensions of the input tensor
        for i in range(total_dim):
            ndim.append(t.shape[i])
    else:   # check whether the number in ndim is less than the size of that dimension
        for i in range(total_dim):
            if ndim[i] > t.shape[i]:
                ndim[i] = t.shape[i]
    
    scree = []
    for i in range(total_dim):
        t_unfold = unfold_axis(t, i)
        [ _, e, _ ] = fast_svd(np.matmul(t_unfold,np.transpose(t_unfold)),ndim[i],n_iter=15)
        e = np.sqrt(e)
        e = np.real(e)
        scree.append(e)

    return scree