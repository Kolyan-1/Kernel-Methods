######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################
from TF.Kernels.Kernels import GAUSSIAN_KERNEL
from TF.TESTS.HSIC      import HSIC, HSIC_TEST
from TF.TESTS.TESTS     import TEST

import time

import numpy as np
import tensorflow as tf


def main_tf(N,learningMethod='tstat',learningRate=0.01):
    X     = tf.placeholder(tf.float32, [None, None])
    Y     = tf.placeholder(tf.float32, [None, None])

    sigK   = tf.get_variable('sigmaK',initializer=tf.constant(0.3))
    sigL   = tf.get_variable('sigmaL',initializer=tf.constant(0.3))

    # clip_op1 = tf.assign(sigK, tf.clip_by_value(sigK, 1e-8, np.infty))
    # clip_op2 = tf.assign(sigL, tf.clip_by_value(sigL, 1e-8, np.infty))

    #getting the Kernels
    k = GAUSSIAN_KERNEL(X,sigK,N)
    l = GAUSSIAN_KERNEL(Y,sigL,N)

    # #getting the HSIC measure
    hsic = HSIC(k,l)
    #
    #getting the treshholds
    h0  = HSIC_TEST(hsic,0.05)

    test = TEST(h0)

    #
    # opt = test.learn_kernel('power',0.05)
    opt = test.learn_kernel(learningMethod,learningRate)

    return X,Y,k,l,hsic,h0,test,opt
