import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd as fast_svd
import scipy.io as sio

# Test 1: quantitatively compare randomized_svd output and matlab eig output
test = sio.loadmat('D:/2020/TensorSVD/ManuscriptRelated/Code/TensorSVD/Eig_test.mat')
t = test['test']
e_matlab = test['e']
U_matlab = test['U']

[U, e , _] = fast_svd(t, 10,n_iter=10)
e_truth, U_truth = np.linalg.eig(np.matmul(t,np.transpose(t)))
print(np.sqrt(np.diag(e_matlab)))
print(e)
print(np.sqrt(e_truth[0:10]))
print('finished')