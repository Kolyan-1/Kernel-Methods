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
import numpy      as np
import tensorflow as tf


class Kernel(object):
    '''
    something like an abstract class
    '''
    def __init__(self,X,params,N):
        self._params = params # the real variable
        self._X      = X #place holder
        self._N      = N
        self._mat    = []

    def get_diff_mat(self):
        # splitting Kij=<xi-xj,xi-xj> = |xi|^2+|xj|^2 - 2<xi,xj>
        xtemp = tf.reduce_sum(tf.multiply(self._X,self._X),axis=1,keep_dims=True)
        x1    = tf.tile(xtemp,[1,self._N])
        x2    = tf.tile(tf.transpose(xtemp),[self._N,1])
        final = x1 + x2 - 2*tf.matmul(self._X,tf.transpose(self._X))
        return final
    def __compute_matrix(self):
        pass

    def get_matrix(self):
        pass

    def get_N(self):
        return self._N



##############################################
# START OF GAUSSIAN

class GAUSSIAN_KERNEL(Kernel):
    '''
    Gaussian Kernel for:
    X rows are elemnts, column length is dimension
    '''
    def __init__(self,X,params,N):
        super(GAUSSIAN_KERNEL,self).__init__(X,params,N)

    def __compute_matrix(self):
        '''
        Function to compute matrix The Kernel matrix
        '''
        # constant for Kernel
        sigma = self._params
        mat_const = 0.5/(sigma*sigma)

        # getting matrix_ij = ||xi-xj||_2^2
        diff_mat = self.get_diff_mat()
        # getting final matrix
        self._mat = tf.exp(- mat_const * diff_mat )


    def get_matrix(self):
        '''
        returns the kernel matrix (and computes it, if not computed)
        '''
        self.__compute_matrix()
        return self._mat

# END OF GAUSSIAN
##############################################
