import numpy as np

def unfold_axis(data, k):
    # def unfold_axis(data, target_dim):
    data = np.linspace(1,10000,10000)
    data = np.reshape(data,[10,10,10,10])
    # print(t[0,:,0,0])

    target_dim = k
    total_dim = len(data.shape)

    dim_list = []
    for i in range(total_dim):
        dim_list.append((target_dim - i) % total_dim)
    dim_order = tuple(dim_list)
    # print(dim_order)

    data_unfold = np.transpose(data,dim_order)
    data_unfold = np.reshape(data_unfold,[data.shape[0],int(data.size/data.shape[0])])
    # print(data_unfold[2,:])
    return data_unfold