######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################

#
# This is a base file containing the base class for Kernels, which is not very exhaustive and concrete implementations of Kernels
#

import numpy as np



class Kernel(object):
    '''
    something like an abstract class
    '''
    def __init__(self,X,params):
        self._params = params
        self._X      = X
        self._N      = len(X)
        self._c      = False
        self._mat    = np.zeros([self._N,self._N])

    def __compute_matrix(self):
        pass

    def get_matrix(self):
        pass

    def get_N(self):
        return self._N

    def set_params(self,params):
        self._params = params
        self._c  = False




##############################################
# START OF GAUSSIAN

class GAUSSIAN_KERNEL(Kernel):
    '''
    Gaussian Kernel for:
    X rows are elemnts, column length is dimension
    '''
    def __init__(self,X,params):
        super(GAUSSIAN_KERNEL,self).__init__(X,params)

    def __compute_matrix(self):
        '''
        Function to compute matrix The Kernel matrix
        '''
        # constant for Kernel
        sigma = self._params
        mat_const = 1./sigma/sigma/2.0
        # filling diagonal
        np.fill_diagonal( self._mat , np.exp(0) )
        # computing the rest of the matrix (only upper triangle)
        for idx in range(self._N-1):
            for jdx in range(idx+1,self._N):
                self._mat[idx][jdx] = np.exp(  -mat_const * (   np.linalg.norm(  self._X[idx]-self._X[jdx]   )))
                self._mat[jdx][idx] = self._mat[idx][jdx]

        self.__c = True

    def get_matrix(self):
        '''
        returns the kernel matrix (and computes it, if not computed)
        '''
        if not self._c:
            self.__compute_matrix()
        return self._mat

# END OF GAUSSIAN
##############################################
