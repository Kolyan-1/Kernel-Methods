######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################

import numpy as np

#
# This file is a way of learning the Kernel and performing a hypothesis test, by computin the test statistics
#

class TEST(object):
    '''
    main class

    test needs to have:
        get_tstat()
        get_estimate()
        reset(params1,params2)
        get_treshold
        get_power()
    '''
    def __init__(self,test):
        self.__test = test



    # #######################################
    # Optimise over the following parameters

    def learn_kernel(self,params_vec1,params_vec2,method='power'):
        '''
        finds the optimal kernel wrt to (power, test stat itself.. others maybe later)

        parmas1, params2 must be the same length
        '''

        if method=='power':
            vec = self.__learn_kernel_power(params_vec1,params_vec2)
        elif method=='tstat':
            vec = self.__learn_kernel_tstat(params_vec1,params_vec2)
        else:
            vec = []

        amax = np.argmax(vec)
        max  = np.max(vec)

        return max, amax, vec

    def __learn_kernel_power(self,params1,params2):
        '''
        power -
        '''
        num_ker = len(params1)
        powers  = np.zeros(num_ker)

        for idx in range(num_ker):
            self.__test.reset(params1[idx],params2[idx])
            powers[idx] = self.__test.get_power()

        return powers

    def __learn_kernel_tstat(self,params1,params2):
        '''
        tstat -
        '''
        num_ker = len(params1)
        powers  = np.zeros(num_ker)

        for idx in range(num_ker):
            self.__test.reset(params1[idx],params2[idx])
            powers[idx] = self.__test.get_tstat()

        return powers
