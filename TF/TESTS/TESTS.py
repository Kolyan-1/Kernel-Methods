######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################

import numpy as np
import tensorflow as tf
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

    def learn_kernel(self,method='power',learning_rate=0.01):
        '''
        finds the optimal kernel wrt to (power, test stat itself.. others maybe later)

        parmas1, params2 must be the same length
        '''

        if method=='power':
            loss = -self.__test.get_power()
        elif method=='measure':
            loss = -self.__test.get_estimate()
        else:
            loss = -self.__test.get_tstat()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(loss)
