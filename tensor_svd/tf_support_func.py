import numpy as np
import tensorflow as tf

def SVD_tf(data, rank):
    """ Performs SVD similar to randomized SVD using tensorflow

    Parameters
    ----------
    data : numpy array
        2D matrix for SVD.
    rank : optional, list with N integer elements
        Number of left singular vectors to return

    Returns
    -------

    U : numpy array
        Numpy array with rank columns calculated from eigenvectors of data * data'.
    """
    data_sqr = tf.linalg.matmul(data, data, transpose_b=True)
    _, U = tf.linalg.eigh(data_sqr, name=None)
    U = U[:,::-1]   # Reverse e and U as they are both in non-decreasing order
    return np.asarray(U[:,0:rank]),[],[]