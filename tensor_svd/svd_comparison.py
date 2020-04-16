import tensorflow as tf
import numpy as np
import time
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd as fast_svd

time_list_tf = []
size_list = []
for isize in range(8):
    t = tf.constant(np.random.rand(500*(isize+1),500*(isize+1)))
    t_sqr = tf.linalg.matmul(t,t)

    start = time.time()
    
    e, v = tf.linalg.eigh(
        t_sqr, name=None
    )

    end1 = time.time()
    print(end1 - start)
    time_list_tf.append(end1 - start)
    size_list.append(500*(isize+1))

time_list_np = []
time_list_scipy = []
time_list_sklearn = []
time_list_fastsvd = []

for isize in range(8):
    t = np.random.rand(500*(isize+1),500*(isize+1))
    start = time.time()
    e, v = np.linalg.eig(np.matmul(t,np.transpose(t)))
    end = time.time()
    time_list_np.append(end-start)
    print(end - start)

    start = time.time()
    e, v = eigs(np.matmul(t,np.transpose(t)))
    end = time.time()
    time_list_scipy.append(end - start)
    print(end - start)

    start = time.time()
    U, e, v = fast_svd(t,500*(isize+1))
    end = time.time()
    time_list_sklearn.append(end - start)
    print(end - start)

    start = time.time()
    U, e, v = fast_svd(t,100)
    end = time.time()
    time_list_fastsvd.append(end - start)
    print(end - start)

plt.plot(size_list, time_list_tf,'.', label = 'Tensorflow')
plt.plot(size_list, time_list_np,'.', label = 'Numpy')
plt.plot(size_list, time_list_scipy,'.', label = 'Scipy')
plt.plot(size_list, time_list_sklearn,'.', label = 'Sklearn')
plt.plot(size_list, time_list_fastsvd,'.', label = 'Fast SVD, 100 Components')
plt.legend(fontsize = 14)
plt.xlabel('Matrix Size',fontsize=14)
plt.ylabel('SVD Time (sec)',fontsize=14)
plt.show()

print('finished')