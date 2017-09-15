######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################



from Data.synthetic1 import blobs, gaussian, circle
from Utils.plotting  import plot2D, plot2,plotfunc
from NP.Kernels.Kernels import GAUSSIAN_KERNEL
from NP.TESTS.HSIC      import HSIC, HSIC_TEST
# from TESTS.TESTS     import

import time

import numpy as np

X = circle(10,0.05)


gaussian_params = 0.5

#getting the Kernels
k = GAUSSIAN_KERNEL(X[:,0],gaussian_params)
l = GAUSSIAN_KERNEL(X[:,1],gaussian_params)

#getting the HSIC measure
hsic = HSIC(k,l)

c = k.get_matrix()
a,b = hsic.get_central_matrix()
print(a)
print(c)

#getting the treshholds
h0  = HSIC_TEST(hsic,0.05)

tresh, dist = h0.get_treshold(True,1000)
powerb = h0.get_tstat()
powers = h0.get_tstat(True)

print(powerb)
print(powers)

print(tresh)




h0.reset(0.2,0.2)

tresh, dist = h0.get_treshold(True,1000)
powerb = h0.get_tstat()
powers = h0.get_tstat(True)

print(powerb)
print(powers)

print(tresh)

# plotfunc(np.sort(dist))
