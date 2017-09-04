######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################

#
# the bottom part of this file is not by me (as is indicated below)
#


import numpy as np
from sklearn.utils import check_random_state


def circle(n,var,rs=1):
    rs = check_random_state(rs)

    xvec = np.linspace(0,2*np.pi,n)

    X = np.cos(xvec) + rs.normal(0,var,n)
    Y = np.sin(xvec) + rs.normal(0,var,n)


    return np.reshape(X,[-1,1]), np.reshape(Y,[-1,1])

######################################
#
# THE CODE BELOW IS NOT MY CODE
# SOURCE GITHUB: https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/generate.py
#####################################





def gaussian(n,corr,rs=1):
    rs = check_random_state(rs)
    mu = np.zeros(2)
    correlation = corr
    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)

    return  np.reshape(Y[:,0],[-1,1]), np.reshape(Y[:,1],[-1,1])




def blobs(n, corr, rows=5, cols=5, sep=10, rs=1):
    rs = check_random_state(rs)
    # ratio is eigenvalue ratio
    correlation = corr

    # generate within-blob variation
    mu = np.zeros(2)
    sigma = np.eye(2)

    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)


    Y[:, 0] += rs.randint(rows, size=n) * sep
    Y[:, 1] += rs.randint(cols, size=n) * sep

    return  np.reshape(Y[:,0],[-1,1]), np.reshape(Y[:,1],[-1,1])
