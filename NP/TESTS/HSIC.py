######################################
#
#  Nikolai Rozanov (C) 2017-Present
#
#  nikolai.rozanov@gmail.com
#
#####################################

import numpy as np
import scipy as sp
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
        # internal paramters
        self.__ce        = False
        self.__cm        = True
        self.__cv        = False

        # kernels
        self.__kernelK   = K
        self.__kernelL   = L
        self.__N         = self.__kernelL.get_N()

        # matrices
        self.__K                 = self.__kernelK.get_matrix()
        self.__L                 = self.__kernelL.get_matrix()
        self.__Ktil, self.__Ltil = self.__centre_fast()
        # final HSIC estimate
        self.__estimate  = 0.0
        self.__var_est   = 1.0


    # ################################
    #
    #
    # MAIN CALLABLE FUNCTIONS
    #
    #


    def get_estimate( self,type='None',force=False,perm_bool=False,permutation=[] ):
        # get final estimate (note, might invoke calculation)
        # permutation only works with fast/None
        if type=='fast' or perm_bool:
            self.__calculate_Estimate_fast(perm_bool,permutation)
        else:
            self.__calculate_Estimate(type,force)

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

    def reset_params_and_matrix(self,paramsK,paramsL):
        self.set_params(paramsK,paramsL)
        self.__get_matrix()

    def set_params(self,paramsK,paramsL):
        self.set_K_params(paramsK)
        self.set_L_params(paramsL)

    def set_K_params(self,params):
        self.__kernelK.set_params(params)

    def set_L_params(self,params):
        self.__kernelL.set_params(params)

    def get_KernelK(self):
        return self.__kernelK

    def get_KernelL(self):
        return self.__kernelL

    def get_N(self):
        return self.__N

    def get_central_matrix(self):
        return self.__Ktil, self.__Ltil
    #
    # ##################################




    # ################################
    # Various Internal Helper and Dev Functions
    def debug1(self):
        print(self.__H_mat)
        print(np.trace(np.dot(self.__H_mat,self.__H_mat))/self.__N/self.__N)

    def __get_matrix(self):
        # get new Kernel matrices
        self.__K                 = self.__kernelK.get_matrix()
        self.__L                 = self.__kernelL.get_matrix()
        self.__Ktil, self.__Ltil = self.__centre_fast()



    def __calculate_Estimate(self,type='None',force=False):
        '''
        wrapper function for various estimators, not very elegant at the moment
        '''

        if (not self.__ce) or force:
            if type=="direct":
                self.__calculate_Estimate_direct()
            elif type=="brute":
                self.__calculate_Estimate_brute()
            elif type=="centering_direct":
                self.__calculate_Estimate_centering_direct()
            else:
                self.__calculate_Estimate_fast()

    #
    # ##################################



    # ##########################################################
    # HELPERS for Estimators
    def __centre_slow(self):
        # centering matrix
        H_mat     = np.eye(self.__N) - (np.ones([self.__N,self.__N])/float(self.__N))

        # centre them
        K_tilde = np.dot(self.__K , H_mat)
        L_tilde = np.dot(self.__L , H_mat)

        return K_tilde,L_tilde

    def __centre_fast(self):
        # centre them
        K_centre = self.__K - np.sum(self.__K, axis=0, keepdims=True) / self.__N
        L_centre = self.__L - np.sum(self.__L, axis=0, keepdims=True) / self.__N

        K_centre = K_centre - np.sum(K_centre, axis=1, keepdims=True) / self.__N
        L_centre = L_centre - np.sum(L_centre, axis=1, keepdims=True) / self.__N

        return K_centre,L_centre
    #
    # #########################################################



    # #########################################################
    # Estimators for HSIC



    ####################################################################
    # currently, best estimator for HSIC itself:
    def __calculate_Estimate_fast(self,perm_bool=False,permutation=[]):
        '''
        currently fastes estimator
        '''
        # final estimate
        if perm_bool:
            self.__estimate = np.sum(self.__Ktil.T*self.__Ltil[permutation][permutation]) /self.__N/self.__N
        else:
            self.__estimate = np.sum(self.__Ktil.T*self.__Ltil) /self.__N/self.__N
        self.__ce = True





    # #####
    # older estimators, they recalculate centering matrices, for profiling while developing
    def __calculate_Estimate_brute(self):
        '''
        very crude function, bruteforce, not using any optimisation or approximation
        '''
        # centre matrix
        K_tilde,L_tilde = self.__centre_slow()

        # final estimate
        self.__estimate = np.trace(np.dot(K_tilde,L_tilde))/self.__N/self.__N
        self.__ce = True

    def __calculate_Estimate_direct(self):
        '''
        using a small trick and the elementwise product
        '''
        # centre matrix
        K_tilde,L_tilde = self.__centre_slow()

        # final estimate
        self.__estimate = np.sum(K_tilde.T*L_tilde) /self.__N/self.__N
        self.__ce = True

    def __calculate_Estimate_centering_direct(self):
        '''
        trick with centering and final trace product
        '''
        K_centre,L_centre = self.__centre_fast()

        # final estimate
        self.__estimate = np.sum(K_centre.T*L_centre) /self.__N/self.__N
        self.__ce = True

    # END OF ESTIMATORS for HSIC itself
    # ################################






    # #########################################################
    # Estimators for HSIC VARIANCE!


    def __calculate_Variance_Estimate(self):
        '''
        estimator for biased estimator
        '''
        # B matrix
        B = self.__Ktil*self.__Ltil
        B = B*B
        B = B - np.diag(np.diag(B))

        # final var estimate
        self.__var_est = np.sum(B)
        self.__cv      = True

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
        self.__measure = measure
        self.__alpha   = alpha
        self.__tresh   = 0.0
        self.__power   = 0.0
        self.__tstat   = 0.0


    # ########################################
    #
    # Main Callable functions
    #
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

    def reset(self,params1,params2):
        '''
        resets the parameters for the measure (in this case HSIC)
        '''
        self.__measure.reset_params_and_matrix(params1, params2)

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
        dist = np.zeros(shuffles);
        N    = self.__measure.get_N()

        # calculating the empirical dist
        for idx in range(shuffles):
            permutation = np.random.permutation(N)
            dist[idx]   = self.__measure.get_estimate(perm_bool=True,permutation=permutation)

        # getting the treshold
        dist         = np.sort(dist)
        self.__tresh = dist[int(np.round( (1-self.__alpha)*shuffles ))]

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
        var          = np.sqrt(self.__measure.get_variance())
        self.__power = sp.stats.norm.cdf( self.__measure.get_estimate(type='fast')/var    -   self.__tresh/(N*var) )

    def __calculate_power_stat_largeM(self):
        self.__power = sp.stats.norm.cdf(self.__measure.get_estimate(type='fast')/np.sqrt(self.__measure.get_variance()))


    def __calculate_t_stat_smallM(self):
        _            = self.__permutation_estimator() #sets self.tresh

        N            = self.__measure.get_N()
        var          = np.sqrt(self.__measure.get_variance())
        self.__tstat = self.__measure.get_estimate(type='fast')/var    -   self.__tresh/(N*var)


    def __calculate_t_stat_largeM(self):
        self.__tstat = self.__measure.get_estimate(type='fast')/np.sqrt(self.__measure.get_variance())

#
#
# END OF HSIC_TEST CLASS
#
#
# #################################################################################################################
