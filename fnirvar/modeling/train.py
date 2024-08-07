""" 
Script defining classes and functions to do estimation abd prediction of a factor + NIRVAR model.
"""

#!/usr/bin/env python3 
# USAGE: ./train_model.py 

import numpy as np

def eigenvalue_ratio():
    """
    Function to calculate the number of factors using the eigenvalue ratio method (Horenstein, 2013).
    """
    pass

def PCp_criteria():
    """
    Function to calculate the number of factors using the PCp criteria method (Bai & Ng, 2002).
    """
    pass

def PCp2_criteria(): 
    """
    Function to calculate the number of factors using the PCp2 criteria method (Bai & Ng, 2002).
    """
    pass

class FactorAdjustment():
    """
    Class to estimate the factors and loadings of the common component.
    Current estimation methods include:
    - Principal Component Analysis (Stock & Watson, 2002) for the static factor model.

    """
    def __init__(self,
                X : np.ndarray,
                r : int, 
                lF : int
                ) -> None:
        """
        :param X: Design matrix of shaoe (T, N) 
        :type X: numpy.ndarray 

        :param r: Number of factors 
        :type r: int 

        :param lF: Number of lags in the model for the factors 
        :type lF: int  
        """
        self.X = X 
        self.r = r 
        self.lF = lF

    @property
    def N(self):
        N = self.X.shape[1] 
        return N
    
    @property
    def T(self):
        T = self.X.shape[0] 
        return T
    
    @property
    def T_lF(self):
        T_lF = self.T - self.lF 
        return T_lF



    def loadings(self):
        """
        Function to estimate the loadings of the common component.
        """
        evecs = np.linalg.eigh(self.X.T @ self.X)[1][:, -self.r:] # Eigenvectors of the covariance matrix corresponding to the r largest eigenvalues 
        return evecs 
    
    def static_factors(self):  
        """
        Function to estimate the factors of the common component.
        """
        loadings = self.loadings()
        factors = self.X @ loadings / self.N 
        return factors 
    
    def factor_linear_model(self):
        """
        Function to estimate the coefficients of the linear model for the latent factors using OLS.
        """
        factors = self.static_factors()
        targets = factors[self.lF:, :] 

        factor_design = np.zeros((self.T_lF, int(self.r * self.lF)))
        for tau in range(self.T_lF):
            F_flat_tau = factors[tau:tau+self.lF, :].flatten() 
            factor_design[tau, :] = F_flat_tau

        factor_coefficients = np.zeros((self.r, self.r * self.lF))
        for f in range(self.r):
            factor_coefficients[f,:] = np.linalg.inv(factor_design.T @ factor_design) @ factor_design.T @ targets[:,f] # OLS estimation 
        return factor_coefficients
    
    def get_idiosyncratic_component(self):
        """
        Function to estimate the idiosyncratic component.
        """
        xi = self.X - self.static_factors() @ self.loadings().T
        return xi  
    
    def predict_common_component(self):
        """
        Function to predict then next day common component.
        """
        loadings = self.loadings()
        factors = self.static_factors()
        P_hat = self.factor_linear_model() 
        G_hat_t = factors[-self.lF:, :].flatten()[:, np.newaxis]
        predicted_common_component = loadings @ P_hat @ G_hat_t 
        return predicted_common_component 
    
    
        