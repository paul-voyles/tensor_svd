import numpy as np
import time
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd as fast_svd
import tensorflow as tf
from tensor_svd_support_func import SVD_tf as fast_svd_2

t = np.random.rand(1000,500)
# e, U = np.linalg.eig(np.matmul(t,np.transpose(t)))
# e = np.sqrt(np.real(e))
# U = np.real(U)
U_approx, e_approx, V_approx = fast_svd(t,10)
U_tf, e_tf, V_tf = fast_svd_2(t,10)

# t_sqr = tf.linalg.matmul(t, t, transpose_b=True)
# e_tf, U_tf = tf.linalg.eigh(t_sqr, name=None)
# e_tf = e_tf[::-1]
# U_tf = U_tf[:,::-1]

print(np.linalg.norm((U[:,1]-U_approx[:,1])))
print(U[0:10,1])
print(U_approx[0:10,1])
print(e[0:10])
print(e_approx[0:10])