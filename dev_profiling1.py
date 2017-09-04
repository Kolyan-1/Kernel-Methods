######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################

#
# This is for profiling HSIC estimators speed measurements
#

from Data.synthetic1 import blobs, gaussian, circle
from Utils.plotting  import plot2D, plot2
from Kernels.Kernels import GAUSSIAN_KERNEL
from TESTS.HSIC       import HSIC

import time

import numpy as np
# X,Y = blobs(100,0.2)
# X,Y = gaussian(100,1)
X = circle(1000,0.05)


# plot2(X)
# plot2(Y)

# print(X)

gaussian_params = 0.5

k = GAUSSIAN_KERNEL(X[:,0],gaussian_params)
l = GAUSSIAN_KERNEL(X[:,1],gaussian_params)

# k = GAUSSIAN_KERNEL(Y[:,0],gaussian_params)
# l = GAUSSIAN_KERNEL(Y[:,1],gaussian_params)

# print(k.get_matrix())
# print(l.get_matrix())

hsic = HSIC(k,l)
# hsic.debug1()
#
t0 = time.time()
est = hsic.get_estimate('brute')
t1 = time.time()
print("The Estimator is: %f and it took Time %f"%(est,t1-t0))


t0 = time.time()
est = hsic.get_estimate('direct',True)
t1 = time.time()
print("The Estimator is: %f and it took Time %f"%(est,t1-t0))

t0 = time.time()
est = hsic.get_estimate('centering_direct',True)
t1 = time.time()
print("The Estimator is: %f and it took Time %f"%(est,t1-t0))

t0 = time.time()
est = hsic.get_estimate('fast',True)
t1 = time.time()
print("The Estimator is: %f and it took Time %f"%(est,t1-t0))
# print(np.cov(X.T))
