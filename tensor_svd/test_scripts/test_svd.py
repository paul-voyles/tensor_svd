from unittest import TestCase
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
from tensor_svd.tensor_svd_denoise import tensor_svd_denoise

class test_svd(TestCase):
    def test_to_hs(self):
        n = np.load("Simulation_noisy_SiDisl_slice_5_1000FPS_cropped_100layers.npy")
        s = hs.signals.Signal1D(n)
        X = tensor_svd_denoise(s, (3,4,5))
        X.plot()
        plt.show()

