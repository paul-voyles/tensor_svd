import numpy as np
import time
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd as fast_svd

t = np.random.rand(1000,500)
e, U = np.linalg.eig(np.matmul(t,np.transpose(t)))
e = np.sqrt(np.real(e))
U = np.real(U)
U_approx, e_approx, V_approx = fast_svd(t,100)

print(np.linalg.norm((U[:,1]-U_approx[:,1])))
print(U[0:10,1])
print(U_approx[0:10,1])
print(e[0:10])
print(e_approx[0:10])