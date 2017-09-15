import tensorflow as tf
import numpy as np
x = tf.get_variable('a',initializer=tf.constant(np.array([[1.,2.,3.],[4.,5.,6.]])))
y = tf.get_variable('b',initializer=tf.constant(np.array([[1],[4]])))




sess = tf.Session()

sess.run(tf.global_variables_initializer())
a = tf.reduce_mean(x, axis=0, keep_dims=True)
c = x - a
d = c - tf.reduce_mean(c, axis=1, keep_dims=True)

print(sess.run(c))
print(sess.run(d))

# def tf_dot(x,y):
#     return tf.reduce_sum(x*y)
#
# x = tf.get_variable('a',initializer=tf.constant(np.array([1.0,2.0,3.0])))
#
# y = []
#
# y.append(tf_dot(x,x))
# y.append(tf_dot(x,2*x))
#
# z = tf.contrib.distributions.percentile(y,q=96)
#
# print(y)
#
# sess = tf.Session()
#
# sess.run(tf.global_variables_initializer())
#
# k = x*x
# print(k.eval(session=sess))
# print(z.eval(session=sess))
