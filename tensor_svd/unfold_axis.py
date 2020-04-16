import numpy as np

dim = 1
t = np.linspace(1,1000,1000)
t = np.reshape(t,[10,10,10])
print(t[0,:,0])
# t1 = np.transpose(t,(0,2,1))
t1 = np.reshape(t,[10,100])
# print(t1[2,:])
# print(t[0,:,:])
# print(t[:,0,:])