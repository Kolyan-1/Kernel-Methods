######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################



from Data.synthetic2 import blobs, gaussian, circle
from Utils.plotting  import plot2D, plot2,plotfunc
from TF.Kernels.Kernels import GAUSSIAN_KERNEL
from TF.TESTS.HSIC      import HSIC, HSIC_TEST
from TF.TESTS.TESTS     import TEST

import time

import numpy as np
import tensorflow as tf

N = 100
# X= circle(1000,0.05)
x,y = gaussian(N,0.05)

X     = tf.placeholder(tf.float32, [None, None])
Y     = tf.placeholder(tf.float32, [None, None])


sigK   = tf.get_variable('sigmaK',initializer=tf.constant(1.0))
sigL   = tf.get_variable('sigmaL',initializer=tf.constant(1.0))
clip_op1 = tf.assign(sigK, tf.clip_by_value(sigK, 1e-8, np.infty))
clip_op2 = tf.assign(sigL, tf.clip_by_value(sigL, 1e-8, np.infty))

#getting the Kernels
k = GAUSSIAN_KERNEL(X,sigK,N)
l = GAUSSIAN_KERNEL(Y,sigL,N)

# other = tf.shape(X)
mat = k.get_matrix()


C = mat - tf.reduce_mean(mat, axis=0, keep_dims=True)


# #getting the HSIC measure
hsic = HSIC(k,l)
#
#getting the treshholds
h0  = HSIC_TEST(hsic,0.05)

test = TEST(h0)


val = hsic.get_estimate()
#
# opt = test.learn_kernel('power',0.05)
opt = test.learn_kernel('power',0.05)



A,B = hsic.get_central_matrix()
other = A

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for idx in range(10):
    sess.run(clip_op1)
    sess.run(clip_op2)
    res = val.eval(session=sess,feed_dict={X:x,Y:y})
    newparamK = sigK.eval(session=sess)
    newparamL  = sigL.eval(session=sess)
    sess.run(opt,feed_dict={X:x,Y:y})
    # print(mat.eval(session=sess,feed_dict={X:x}))
    # print(A.eval(session=sess,feed_dict={X:x}))

    print("The new Value is: {} with params: {} and {}".format(res,newparamK,newparamL))


# print(mat.eval(session=sess,feed_dict={X:x,Y:y}))

#
