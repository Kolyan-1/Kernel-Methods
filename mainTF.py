######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################



from TF.mainTF       import main_tf
from Data.synthetic2 import gaussian

import tensorflow as tf

# getting data
N   = 100
x,y = gaussian(N,0.1)

# getting model
X,Y,k,l,hsic,hsic_test,test,opt = main_tf(N)

# init tf
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# running optimisation
sess.run(opt,feed_dict={X:x,Y:y})
