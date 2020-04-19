import numpy as np
from unfold_axis import unfold_axis

k = 1
# Test output shape with random tensor and matrix --- passed
# t = np.random.rand(2,3,4,5,6,7,8)
# m = np.random.rand(10,4)

# Test output values with tensor and identity matrix --- passed
# t = np.linspace(1,120,120)
# t = np.reshape(t,[2,3,4,5])
# m = np.identity(3)*2

# Test output values with tensor, matrix, and result calculated by tensor-toolbox in Matlab --- passed
t = np.load('D:/2020/TensorSVD/ManuscriptRelated/Code/TensorSVD/f.npy')
m = np.load('D:/2020/TensorSVD/ManuscriptRelated/Code/TensorSVD/m.npy')
result = np.load('D:/2020/TensorSVD/ManuscriptRelated/Code/TensorSVD/t_mul.npy')

dim_list = []
shape_list = []
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
print('Finished')