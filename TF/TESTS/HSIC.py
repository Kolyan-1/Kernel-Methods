######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################

import tensorflow as tf

#
# This file contains code to compute HSIC and various Statistics associated with it.
# mainly the V-statistic in (Gretton 2007 NIPS) is used.
# However an extension to use approximate methods for large scale kernel methods is on the TODO list.
#
#

# #################################################################################################################
#
#
# HSIC CLASS
#
#


class HSIC(object):
    '''
    HSIC IS part of the measure class (not implemented this way yet), which needs to have a method called get_estimate()

    HSIC class that implements the estimator.

    We are assuming X and Y have equally many obersavtions

    '''
    def __init__(self,K,L):
        # kernels
        self.__kernelK   = K
        self.__kernelL   = L
        self.__N         = self.__kernelL.get_N()

        # matrices
        self.__K                 = self.__kernelK.get_matrix()
        self.__L                 = self.__kernelL.get_matrix()
        self.__Ktil, self.__Ltil = self.__centre_fast()

        # final HSIC estimate
        self.__estimate = 0.789
        self.__var_est  = 1.0

    # ################################
    #
    #
    # MAIN CALLABLE FUNCTIONS
    #
    #


    def get_estimate( self,perm_bool=False):
        # get final estimate (note, might invoke calculation)
        # permutation only works with fast/None
        self.__calculate_Estimate(perm_bool)
        return self.__estimate


    def get_variance( self ):
        self.__calculate_Variance_Estimate()
        return self.__var_est
    #
    # END OF MAIN CALLABLE FUNCTIONS
    # #################################





    # ################################
    # Functions to change/get kernels
    def set_K(self,kernel):
        self.__kernelK = kernel

    def set_L(self,kernel):
        self.__kernelK = kernel

    def get_central_matrix(self):
        return self.__Ktil, self.__Ltil
    #
    # ##################################




    # ################################
    # Various Internal Helper and Dev Functions
    def __get_matrix(self):
        # get new Kernel matrices
        self.__K                 = self.__kernelK.get_matrix()
        self.__L                 = self.__kernelL.get_matrix()
        self.__Ktil, self.__Ltil = self.__centre_fast()


    #
    # ##################################



    # ##########################################################
    # HELPERS for Estimators
    def __centre_fast(self):
        # centre them
        K_centre = self.__K - tf.reduce_mean(self.__K, axis=0, keep_dims=True)
        L_centre = self.__L - tf.reduce_mean(self.__L, axis=0, keep_dims=True)
        #
        K_centre = K_centre - tf.reduce_mean(K_centre, axis=1, keep_dims=True)
        L_centre = L_centre - tf.reduce_mean(L_centre, axis=1, keep_dims=True)

        return K_centre,L_centre
    #
    # #########################################################



    # #########################################################
    # Estimators for HSIC



    ####################################################################
    # currently, best estimator for HSIC itself:
    def __calculate_Estimate(self,perm_bool=False):
        '''
        currently fastes estimator
        '''

        if perm_bool:
            self.__estimate = tf.reduce_sum(tf.multiply(tf.transpose(self.__Ktil),tf.random_shuffle(self.__Ltil))) / (self.__N*self.__N)
        else:
            self.__estimate = tf.reduce_sum(tf.multiply(tf.transpose(self.__Ktil),self.__Ltil)) / (self.__N*self.__N)





    # #########################################################
    # Estimators for HSIC VARIANCE!


    def __calculate_Variance_Estimate(self):
        '''
        estimator for biased estimator
        '''
        # B matrix
        B = tf.multiply(self.__Ktil,self.__Ltil)
        B = tf.multiply(B,B)
        B = B - tf.diag(tf.diag_part(B))

        # final var estimate
        self.__var_est = tf.reduce_sum(B)

#
#
# END OF HSIC CLASS
#
#
# #################################################################################################################
# #################################################################################################################
# #################################################################################################################
# #################################################################################################################
# #################################################################################################################












# #################################################################################################################
# #################################################################################################################
# #################################################################################################################
# #################################################################################################################
# #################################################################################################################
#
#  HO CLASS, dealing with HSIC under H0
#
#
class HSIC_TEST(object):
    '''
    this class is responsible for various statistics under HSIC (H0, H1 etc.)

    measure needs to have the method: get_estimate(perm_bool=True,permutation="#herecomestherealpermutation")
    '''
    def __init__(self,measure,alpha):
        self.__measure = measure #HSIC in this case
        self.__alpha   = alpha
        self.__tresh   = 0.0
        self.__power   = 0.0
        self.__tstat   = 0.0


    # ########################################
    #
    # Main Callable functions
    #
    def get_estimate(self):
        return self.__measure.get_estimate()
        
    def get_treshold(self,get_dist=False,params=50):
        '''
        get's 1-alpha quantile of H0 distribution
        '''
        dist = self.__permutation_estimator(get_dist=get_dist,shuffles=params)
        return self.__tresh, dist

    def get_power(self,smallM=False):
        '''
        computes the power
        '''
        if smallM:
            self.__calculate_power_stat_smallM()
        else:
            self.__calculate_power_stat_largeM()
        return self.__power

    def get_tstat(self,smallM=False):
        '''
        computes the power
        '''
        if smallM:
            self.__calculate_t_stat_smallM()
        else:
            self.__calculate_t_stat_largeM()
        return self.__tstat


    # ##################################################################################
    #
    #
    # H0
    #
    #
    def __permutation_estimator(self,get_dist=False,shuffles=50):
        '''
        empirical estimator of HSIC_b values under H0
        '''
        dist = [None]*shuffles
        N    = self.__measure.get_N()

        # calculating the empirical dist
        for idx in range(shuffles):
            dist[idx]   = self.__measure.get_estimate(perm_bool=True)

        # getting the treshold
        self.__tresh = tf.stop_gradient(tf.contrib.distributions.percentile(dist,q=self.__alpha))

        if get_dist:
            return dist
        else:
            return []


    def __gamma_estimator(self,get_dist=False,shuffles=50):
        '''
        approximate distribution of HSIC_b using two moments gamma approximation
        (not implemented yet)
        '''
        pass





    # ##################################################################################
    #
    #
    # H1
    #
    #

    def __calculate_power_stat_smallM(self):
        _            = self.__permutation_estimator() #sets self.tresh

        N            = self.__measure.get_N()
        var          = tf.sqrt(self.__measure.get_variance())
        dist         = tf.distributions.Normal(loc=0., scale=1.)
        self.__power = dist.cdf(value=self.__measure.get_estimate()/var    -   self.__tresh/(N*var) )

    def __calculate_power_stat_largeM(self):
        dist         = tf.distributions.Normal(loc=0., scale=1.)
        self.__power = dist.cdf(value=self.__measure.get_estimate()/tf.sqrt(self.__measure.get_variance())) #approximates t dist


    def __calculate_t_stat_smallM(self):
        _            = self.__permutation_estimator() #sets self.tresh

        N            = self.__measure.get_N()
        var          = tf.sqrt(self.__measure.get_variance())
        self.__tstat = self.__measure.get_estimate()/var    -   self.__tresh/(N*var)


    def __calculate_t_stat_largeM(self):
        self.__tstat = self.__measure.get_estimate()/tf.sqrt(self.__measure.get_variance())

#
#
# END OF HSIC_TEST CLASS
#
#
# #################################################################################################################
