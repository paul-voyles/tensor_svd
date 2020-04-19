import numpy as np

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
    # def unfold_axis(data, target_dim):
    # data = np.linspace(1,10000,10000)
    # data = np.reshape(data,[10,10,10,10])
    # print(t[0,:,0,0])

    target_dim = k
    total_dim = len(data.shape)

    dim_list = []
    for i in range(total_dim):
        dim_list.append((target_dim - i) % total_dim)
    dim_order = tuple(dim_list)
    # print(dim_order)

    data_unfold = np.transpose(data,dim_order)
    data_unfold = np.reshape(data_unfold,[data.shape[k],int(data.size/data.shape[k])])
    # print(data_unfold[2,:])
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
    shape_order = tuple(shape_list)

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
    if ndim = []:   # case with no input ndim
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
        [ _, e, _ ] = fast_svd(np.matmul(t,np.transpose(t)),ndim[i])
        e = np.sqrt(e)
        e = np.real(e)
        scree.append(e)

    return scree