######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################
from Kernels.Kernels import GAUSSIAN_KERNEL
from TESTS.HSIC      import HSIC, HSIC_TEST
from TESTS.TESTS     import TEST

import numpy as np



def main_np(X,Y,params,alpha):
    #getting the Kernels
    k = GAUSSIAN_KERNEL(X,gaussian_params)
    l = GAUSSIAN_KERNEL(Y,gaussian_params)

    #getting the HSIC measure
    hsic = HSIC(k,l)


    #getting the treshholds
    h0  = HSIC_TEST(hsic,0.05)

    #getting the test
    test = TEST(h0)


    return k,l,hsic,h0,test
