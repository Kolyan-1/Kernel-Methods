######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################
import numpy as np
from sklearn.utils import check_random_state


# This is the base class for the data part of this Library
# The base class should be more seen as an abstract class, that provides the needed functionality


class BaseData(object):
    def __init__(self):
        self.X = np.array([])
        self.Y = np.array([])

    def assign(self,X,Y):
        self.X = X
        self.Y = Y


class Blobs(BaseData):

    def __init__(self,rows,columns,corr):
        self.rows       = rows
        self.columns    = columns
        self.corr       = corr
        super(Blobs,self).__init__()

    def __generate(self):
        # generate within-blob variation
        mu = np.zeros(2)
        sigma = np.eye(2)
        X = rs.multivariate_normal(mu, sigma, size=n)

        corr_sigma = np.array([[1, correlation], [correlation, 1]])
        Y = rs.multivariate_normal(mu, corr_sigma, size=n)

        # assign to blobs
        X[:, 0] += rs.randint(rows, size=n) * sep
        X[:, 1] += rs.randint(cols, size=n) * sep
        Y[:, 0] += rs.randint(rows, size=n) * sep
        Y[:, 1] += rs.randint(cols, size=n) * sep

        return X, Y
