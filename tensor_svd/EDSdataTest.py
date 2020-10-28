import hyperspy.api as hs
path = 'D:/2020 Cornell/Tensor SVD improvment/102220/EDX/condition1/'
s = hs.load(path + "Scan1 Datacube.rpl")
s.plot()