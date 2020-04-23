import numpy as np
from tensor_svd_support_func import unfold_axis, ttm, scree_plots
import matplotlib.pyplot as plt
from tensor_svd import svd_HO
import time
import scipy.io as sio

data = sio.loadmat('D:/2020/TensorSVD/ManuscriptRelated/SimulationData/DenoiseInput_fullsize/SiDislocation/Simulation_noisy_SiDisl_slc5_1000FPS.mat')
# data = sio.loadmat('D:/2020/TensorSVD/ManuscriptRelated/SimulationData/DenoiseInput_fullsize/STO/Simulation_noisy_STO_slice_5_1000FPS_fullsize.mat')
data = data['datacube'].astype(np.float)

# Test the output of scree_plots -- passed
# ndim = [100,100,100]
# scree = scree_plots(data,ndim)

# for i in range(len(scree)):
#     plt.plot(scree[i],'.',linewidth = 2)
# plt.show()

# print('Scree plots test finished')

# Test the tensor SVD output -- passed
start = time.time()
rank =[33,30,185]
# data = data[:,:,0:10000]
# rank = [7,7,30]
X,U,S = svd_HO(data,rank,max_iter=10)
end = time.time()
print(end - start)
print('SVD finished.')
np.save('SiDisl_1000FPS_denoised_python.npy',X)

# plt.subplot(121)
# plt.imshow(X[:,:,10].reshape(114,114))

# plt.subplot(122)
# plt.imshow(data[:,:,10].reshape(114,114))
# plt.show()

# Test quantitative match between python output and matlab output -- In Jupyter Notebook -- passed
# Max difference = 1%, mostly < 0.3%
# output_python = np.load('STO_1000FPS_denoised.npy')
# output_matlab = sio.loadmat('D:/2020/TensorSVD/ManuscriptRelated/SimulationData/DenoiseOutput_fullsize/STO/Simulation_tensor_STO_slice_5_1000FPS_fullsize.mat')
# output_matlab = output_matlab['est_HOOI']
# plt.imshow(output_python[:,:,100].reshape(114,114) - output_matlab[:,:,100].reshape(114,114))

# Test processing speed and benchmark against Matlab
# time_desktop = []
# for i in range(15):
#     data_cropped = data[:,:,0:1000*(i+1)]
#     start = time.time()
#     rank =[33,30,185]
#     X,U,S = svd_HO(data_cropped,rank,10)
#     end = time.time()
#     print(end - start)
#     time_desktop.append(end-start)
# np.save('time_desktop.npy',np.asarray(time_desktop_tf))