######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################



from Data.synthetic1 import blobs, gaussian, circle
from Utils.plotting  import plot2D, plot2,plotfunc
from Kernels.Kernels import GAUSSIAN_KERNEL
from TESTS.HSIC      import HSIC, HSIC_TEST
from TESTS.TESTS     import TEST

import time

import numpy as np

# X= circle(1000,0.05)
X = gaussian(500,0.05)

gaussian_params = 0.1

#getting the Kernels
k = GAUSSIAN_KERNEL(X[:,0],gaussian_params)
l = GAUSSIAN_KERNEL(X[:,1],gaussian_params)

#getting the HSIC measure
hsic = HSIC(k,l)


#getting the treshholds
h0  = HSIC_TEST(hsic,0.05)

test = TEST(h0)

# gaussian_params_for_opt = [0.0001,0.001,0.003,0.006,0.009,0.01,0.02,0.05,0.08,0.1]
# gaussian_params_for_opt = [0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5]


# max, amax, powers = test.learn_kernel(gaussian_params_for_opt,gaussian_params_for_opt,'power')
#
# print(max)
# print(gaussian_params_for_opt[amax])
print(h0.get_power())
# plotfunc(powers)

# plotfunc(np.sort(dist))
