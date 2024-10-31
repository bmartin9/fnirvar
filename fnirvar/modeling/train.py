""" 
Script defining classes and functions to do estimation and prediction of a factor + NIRVAR model.
"""

#!/usr/bin/env python3 
# USAGE: ./train_model.py 

import numpy as np
# from skrmt.ensemble.spectral_law import MarchenkoPasturDistribution
from numpy.linalg import svd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from scipy import linalg 
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import MultiTaskLassoCV

def eigenvalue_ratio_test(X, kmax=None):
    """
    Compute the Eigenvalue Ratio (ER) test statistic of Ahn and Horenstein (2013)
    to determine the number of factors in approximate factor models.
    
    Parameters:
    -----------
    X : ndarray
        The input data matrix of shape (n_samples, n_variables)
        Should be standardized (zero mean, unit variance)
    kmax : int, optional
        Maximum number of factors to consider
        If None, will be set to min(n_samples, n_variables) - 1
        
    Returns:
    --------
    k_er : int
        Estimated number of factors according to ER criterion
    er_ratios : ndarray
        Array of eigenvalue ratios for each k
    eigenvalues : ndarray
        Array of ordered eigenvalues
        
    References:
    -----------
    Ahn, S. C., & Horenstein, A. R. (2013). Eigenvalue ratio test for the number
    of factors. Econometrica, 81(3), 1203-1227.
    """
    
    # Get dimensions
    n, p = X.shape
    
    # Set default kmax if not provided
    if kmax is None:
        kmax = min(n, p) - 1
    
    # Compute sample covariance matrix
    S = np.dot(X.T, X) / n
    
    # Compute eigenvalues
    eigenvalues = linalg.eigvalsh(S)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
    
    # Initialize arrays
    er_ratios = np.zeros(kmax)
    
    # Compute ER ratios for k = 1 to kmax
    for k in range(kmax):
        er_ratios[k] = eigenvalues[k] / eigenvalues[k + 1]
    
    # Find k that maximizes the ER ratio
    k_er = np.argmax(er_ratios) + 1
    
    return k_er, er_ratios, eigenvalues

def ER(X : np.ndarray, kmax : int) -> None : 
    """ 
    :param X: Design matrix of shape (T,N)
    :type X: np.ndarray 

    :param kmax: maximum possible number of factors 
    :type kmax: int 

    :return k_star: chosen number of factors 
    :rtype int: 
    """

    T = X.shape[0] 
    N = X.shape[1] 

    if N > T:
        S = X@X.T / N*T 
    else:
        S = X.T@X / N*T 

    eigenvalues = linalg.eigvalsh(S)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order 

    er_ratios = np.zeros(kmax)
    for k in range(kmax):
        er_ratios[k] = eigenvalues[k] / eigenvalues[k + 1]

    k_star = np.argmax(er_ratios) + 1

    return k_star 

def GR(X : np.ndarray, kmax : int) -> None : 
    """ 
    :param X: Design matrix of shape (T,N)
    :type X: np.ndarray 

    :param kmax: maximum possible number of factors 
    :type kmax: int 

    :return k_star: chosen number of factors 
    :rtype int: 
    """

    T = X.shape[0] 
    N = X.shape[1] 

    if N > T:
        S = X@X.T / N*T 
    else:
        S = X.T@X / N*T 

    eigenvalues = linalg.eigvalsh(S)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order 
    V = np.cumsum(eigenvalues[::-1])[::-1]

    er_ratios = np.zeros(kmax)
    for k in range(kmax):
        V_k = V[k+1] 
        V_k_plus1 = V[k+2]
        er_ratios[k] = np.log(1 + (eigenvalues[k]/V_k)) / np.log(1 + (eigenvalues[k + 1]/V_k_plus1))

    k_star = np.argmax(er_ratios) + 1

    return k_star 
    

    



def minindc(X):
    ''' =========================================================================
     takes np <-> returns np
     DESCRIPTION
     This function finds the index of the minimum value for each column of a
     given matrix. The function assumes that the minimum value of each column
     occurs only once within that column. The function returns an error if
     this is not the case.

     -------------------------------------------------------------------------
     INPUT
               x   = matrix

     OUTPUT
               pos = column vector with pos(i) containing the row number
                     corresponding to the minimum value of x(:,i)

     ========================================================================= '''

    mins = X.argmin(axis=0)
    assert sum(X == X[mins]) == 1, 'Minimum value occurs more than once.'
    return mins

def baing(X,kmax,jj):
    #take in and return numpy arrays
    ''' =========================================================================
    DESCRIPTION
    This function determines the number of factors to be selected for a given
    dataset using one of three information criteria specified by the user.
    The user also specifies the maximum number of factors to be selected.

    -------------------------------------------------------------------------
    INPUTS
               X       = dataset (one series per column)
               kmax    = an integer indicating the maximum number of factors
                         to be estimated
               jj      = an integer indicating the information criterion used
                         for selecting the number of factors; it can take on
                         the following values:
                               1 (information criterion PC_p1)
                               2 (information criterion PC_p2)
                               3 (information criterion PC_p3)

     OUTPUTS
               ic1     = number of factors selected
               chat    = values of X predicted by the factors
               Fhat    = factors
               eigval  = eivenvalues of X'*X (or X*X' if N>T)

     -------------------------------------------------------------------------
     SUBFUNCTIONS USED

     minindc() - finds the index of the minimum value for each column of a given matrix

     -------------------------------------------------------------------------
     BREAKDOWN OF THE FUNCTION

     Part 1: Setup.

     Part 2: Calculate the overfitting penalty for each possible number of
             factors to be selected (from 1 to kmax).

     Part 3: Select the number of factors that minimizes the specified
             information criterion by utilizing the overfitting penalties calculated in Part 2.

     Part 4: Save other output variables to be returned by the function (chat,
             Fhat, and eigval).

    ========================================================================= '''
    assert kmax <= X.shape[1] and  kmax >= 1 and np.floor(kmax) == kmax or kmax == 99, 'kmax is specified incorrectly'
    assert jj in [1, 2, 3], 'jj is specified incorrectly'


    #  PART 1: SETUP

    T = X.shape[0]  # Number of observations per series (i.e. number of rows)
    N = X.shape[1]  # Number of series (i.e. number of columns)
    NT = N * T      # Total number of observations
    NT1 = N + T     # Number of rows + columns

    #  =========================================================================
    #  PART 2: OVERFITTING PENALTY
    #  Determine penalty for overfitting based on the selected information
    #  criterion.

    CT = np.zeros(kmax) # overfitting penalty
    ii = np.arange(1, kmax + 1)     # Array containing possible number of factors that can be selected (1 to kmax)
    GCT = min(N,T)                  # The smaller of N and T

    # Calculate penalty based on criterion determined by jj.
    if jj == 1:             # Criterion PC_p1
        CT[:] = np.log(NT / NT1) * ii * (NT1 / NT)

    elif jj == 2:             # Criterion PC_p2
        CT[:] = np.log(min(N, T)) * ii * (NT1 / NT)

    elif jj == 3:             # Criterion PC_p3
        CT[:] = np.log(GCT) / GCT * ii

    #  =========================================================================
    #  PART 3: SELECT NUMBER OF FACTORS
    #  Perform principal component analysis on the dataset and select the number
    #  of factors that minimizes the specified information criterion.
    #
    #  -------------------------------------------------------------------------
    #  RUN PRINCIPAL COMPONENT ANALYSIS
    #  Get components, loadings, and eigenvalues

    if T < N:
        ev, eigval, V = np.linalg.svd(np.dot(X, X.T))       #  Singular value decomposition
        Fhat0 = ev*np.sqrt(T)                               #  Components
        Lambda0 = np.dot(X.T, Fhat0) / T                    #  Loadings
    else:
        ev, eigval, V = np.linalg.svd(np.dot(X.T, X))       #  Singular value decomposition
        Lambda0 = ev*np.sqrt(N)                             #  Loadings
        Fhat0 = np.dot(X, Lambda0) / N                      #  Components
    #  -------------------------------------------------------------------------

    # SELECT NUMBER OF FACTORS
    # Preallocate memory
    Sigma = np.zeros(kmax + 1)          # sum of squared residuals divided by NT, kmax factors + no factor
    IC1 = np.zeros(kmax + 1)            # information criterion value, kmax factors + no factor

    for i in range(0, kmax) :           # Loop through all possibilites for the number of factors
        Fhat = Fhat0[:, :i+1]           # Identify factors as first i components
        lambda_ = Lambda0[:, :i+1]       #     % Identify factor loadings as first i loadings

        chat = np.dot(Fhat, lambda_.T)      #     % Predict X using i factors
        ehat = X - chat                 # Residuals from predicting X using the factors
        Sigma[i] = ((ehat*ehat/T).sum(axis = 0)).mean()    # Sum of squared residuals divided by NT

        IC1[i] = np.log(Sigma[i]) + CT[i]      #  Value of the information criterion when using i factors


    Sigma[kmax] = (X*X/T).sum(axis = 0).mean()  # Sum of squared residuals when using no factors to predict X (i.e. fitted values are set to 0)

    IC1[kmax] =  np.log(Sigma[kmax]) # % Value of the information criterion when using no factors

    ic1 = minindc(IC1) # % Number of factors that minimizes the information criterion
    # Set ic1=0 if ic1>kmax (i.e. no factors are selected if the value of the
    # information criterion is minimized when no factors are used)
    ic1 = ic1 *(ic1 < kmax) # if = kmax -> 0

    #  =========================================================================
    #  PART 4: SAVE OTHER OUTPUT
    #
    #  Factors and loadings when number of factors set to kmax

    Fhat = Fhat0[:, :kmax] # factors
    Lambda = Lambda0[:, :kmax] #factor loadings

    chat = np.dot(Fhat, Lambda.T) #     Predict X using kmax factors

    return ic1+1, chat, Fhat, eigval

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
        if self.N > self.T:
            factors = self.static_factors()
            loadings = self.X.T @ factors / self.T 
        else:
            loadings = np.linalg.eigh(self.X.T @ self.X )[1][:, -self.r:] # Eigenvectors of the covariance matrix corresponding to the r largest eigenvalues 
        return loadings
    
    def static_factors(self):  
        """
        Function to estimate the factors of the common component.
        """
        if self.N > self.T:
            factors = np.sqrt(self.T)*np.linalg.eigh(self.X @ self.X.T / self.N)[1][:, -self.r:] 
        else:
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
    
###### FUNCTIONS FOR COMPUTING BEST FIT MARCENKO-PASTUR DISTRIBUTION ######
    
def ks_statistic(sigma_squared, data, q):
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    mpl = MarchenkoPasturDistribution(beta=1, ratio=q, sigma=np.sqrt(sigma_squared) )
    theoretical_cdf = mpl.cdf(np.sort(data))
    ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
    return ks_stat

def exponential_weight_function(empirical_cdf, alpha=5):
    return np.exp(alpha * empirical_cdf)

# Weighted KS statistic function with exponential weights
def weighted_ks_statistic(sigma_squared, data, q, alpha=1):
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    mpl = MarchenkoPasturDistribution(beta=1, ratio=q, sigma=np.sqrt(sigma_squared) )

    theoretical_cdf = mpl.cdf(sorted_data)
    
    # Calculate weights using the exponential weight function
    weights = exponential_weight_function(empirical_cdf, alpha)
    
    # Calculate weighted discrepancies
    weighted_discrepancies = weights * np.abs(empirical_cdf - theoretical_cdf)
    
    # Calculate weighted KS statistic
    ks_stat = np.sum(weighted_discrepancies)
    return ks_stat

###### NIRVAR ESTIMATION AND PREDICTION ######

class NIRVAR():
    """ 
    Class to do NIRVAR estimation and prediction.
    """
    def __init__(self,
                Xi : np.ndarray,
                d : int = None, 
                K : int = None,
                embedding_method : str = "Pearson Correlation",
                gmm_random_int : int = 432
                ) -> None:
        """ 
        :param random_state: Random State object. The seed value is set by the user upon instantiation of the Random State object.
        :type: np.random.RandomState

        :param Xi: Design matrix of shaoe (T, N) 
        :type Xi: numpy.ndarray 

        :param d: embedding dimension
        :type d: int

        :param K: Number of Gaussian mixtures used for clustering
        :type K: int

        :param embedding_method: one of 'Pearson Correlation' (Default), 'Precision Matrix', 'Covariance Matrix' 
        :type embedding_method: str
        """
        self.Xi = Xi 
        self.embedding_method = embedding_method
        self.gmm_random_int = gmm_random_int 
        self.d = d if d is not None else self.marchenko_pastur_estimate()  
        self.K = K if K is not None else self.d 

    @property
    def T(self):
        T = self.Xi.shape[0] 
        return T

    @property
    def N(self):
        N = self.Xi.shape[1] 
        return N
        
    def pearson_correlations(self) -> np.ndarray: 
        """
        :return: The (N x N) Pearson correlation matrix. Shape = (N,N)
        :rtype: numpy.ndarray 
        """
        p_corr = np.corrcoef(self.Xi.T) 
        return p_corr
    
    def covariance_matrix(self) -> np.ndarray:
        """
            :return: The (N x N) sample covariance matrix. Shape = (N,N)
            :rtype: numpy.ndarray 
        """
            
        cov_mat = np.cov(self.Xi.T)  
        return cov_mat
    
    def inverse_correlation_matrix(self) -> np.ndarray:
        """
            :return: The (N x N) inverse correlation matrix. Shape = (N,N)
            :rtype: numpy.ndarray 
        """
        p_corr = np.corrcoef(self.Xi.T) 
        prec_mat = np.linalg.inv(p_corr)
        return prec_mat 
    
    def marchenko_pastur_estimate(self) -> int:
        """
        :return: Estimated number of "significant" dimensions
        :rtype: int
        """
        if self.embedding_method == 'Pearson Correlation':
            Sigma = self.pearson_correlations()
            eigenvalues = np.linalg.eigvals(Sigma)
            cutoff = (1 + np.sqrt(self.N/self.T))**2
            d_hat = np.count_nonzero(eigenvalues > cutoff) 
        elif self.embedding_method == 'Precision Matrix':
            Sigma = self.inverse_correlation_matrix()
            eigenvalues = np.linalg.eigvals(Sigma)
            ratio_limit = self.N/self.T 
            cutoff = ((1 - np.sqrt(ratio_limit))/(1 - ratio_limit))**2 
            d_hat = np.count_nonzero(eigenvalues < cutoff) 
        elif self.embedding_method == 'Covariance Matrix':
            Sigma = self.pearson_correlations()
            eigenvalues = np.linalg.eigvals(Sigma)
            cutoff = (1 + np.sqrt(self.N/self.T))**2
            d_hat = np.count_nonzero(eigenvalues > cutoff) 
        else:
            print("ERROR : Embedding method must be one of Pearson Correlation or Covariance Matrix or Precision Matrix")
        
        return d_hat 
    
    def embed(self):
        """ 
        :return: Embedded points. Shape = (N,d) 
        :return type: np.ndarray
        """
        if self.embedding_method == "Pearson Correlation":
            embedding_object = self.pearson_correlations()
        elif self.embedding_method == "Covariance Matrix":
            embedding_object = self.covariance_matrix()
        elif self.embedding_method == "Precision Matrix":
            embedding_object == self.inverse_correlation_matrix()
        else:
            print("ERROR : Embedding method in embed() must be one of Pearson Correlation or Covariance Matrix or Precision Matrix")

        U, S, Vh = svd(embedding_object,full_matrices=False) 
        sorted_indices = np.argsort(np.abs(S))[::-1]
        largest_indices = sorted_indices[:self.d]
        Vh_trun = Vh[largest_indices]
        S_trun = S[largest_indices]
        embedded_array =  Vh_trun.T@np.diag(S_trun) 
        return embedded_array 

    @staticmethod
    def groupings_to_2D(input_array : np.ndarray) -> np.ndarray:
        """ 
        Turn a 1d array of integers (groupings) into a 2d binary array, A, where 
        A[i,j] = 1 iff i and j have the same integer value in the 1d groupings array.

        :param input_array: 1d array of integers.
        :type input_array: np.ndarray

        :return: 2d Representation. Shape = (len(input_array),len(input_array))
        :rtype: np.ndarray
        """

        L = len(input_array)
        A = np.zeros((L,L)) 
        for i in range(L):
            for j in range(L): 
                if input_array[i] == input_array[j]:
                    A[i][j] = 1 
                else:
                    continue 
        
        return A 
    
    def gmm(self) -> np.ndarray:
        """
        GMM clustering. Number of clusters must be pre-specified. EM algorithm is then run.

        :return: similarity_matrix. A binary array with value 1 for the neighboring stocks in the same cluster and 0 otherwise. Shape = (N, N)
        :rtype: np.ndarray

        :return: labels. Array of integers where each integer labels a GMM cluster. Shape = (Q, N)
        :rtype: np.ndarray
        """
        embedding = self.embed()
        gmm_labels = GaussianMixture(n_components=self.K, random_state=self.gmm_random_int, init_params='k-means++').fit_predict(embedding)
        labels = gmm_labels
        similarity_matrix = self.groupings_to_2D(labels)

        return similarity_matrix, labels

    def covariates(self,constrained_array : np.ndarray) -> np.ndarray: 
        """ 
        :param constrained_array: Shape = (N,N) Some constraint on which neighbours to sum up to get a predictor along that feature.
        :type: np.ndarray 

        :rtype: np.ndarray 
        :return: Shape = (N,N,T) For each stock, we have a maximum (this max is not reached do to clustering regularisation) of NQ predictors. There are T training values for each predictor.
        :rtype: np.ndarray 
        """
        covariates = np.zeros((self.N,self.N,self.T),dtype=np.float32)
        for i in range(self.N):
            c = constrained_array[i,:,None]*(self.Xi.transpose(1,0)) 
            covariates[i] = c
        return covariates
    
    def ols_parameters(self,constrained_array : np.ndarray) -> np.ndarray:
        """ 
        :param constrained_array: Some constraint on which neighbours to sum up to get a predictor along that feature. Shape= (N,N)
        :type: np.ndarray

        :return: ols_params Shape = (N,N)
        :rtype: np.ndarray
        """
        ols_params = np.zeros((self.N,self.N)) 
        covariates = self.covariates(constrained_array=constrained_array)
        targets = self.Xi # shape = (T,N) 
        for i in range(self.N):
            ols_reg_object = LinearRegression(fit_intercept=False)
            x = covariates[i].reshape(-1,covariates[i].shape[-1],order='F').T[:-1,:] #shape = (T_train-1,NQ)
            non_zero_col_indices = np.where(x.any(axis=0))[0] #only do ols on stocks that are connected to node i
            x_reg = x[:,non_zero_col_indices]
            y = targets[1:,i] 
            ols_fit = ols_reg_object.fit(x_reg,y) 
            ols_params[i,non_zero_col_indices] = ols_fit.coef_  
        return ols_params 

    def predict_idiosyncratic_component(self):
        """ 
        Method to predict the next day idiosyncratic component.

        :return Xi_hat: predicted next day idiosyncratic component. Shape = (N,1) 
        :return type: np.ndarray  
        """
        similarity_matrix, labels = self.gmm() 
        phi_hat = self.ols_parameters(constrained_array=similarity_matrix)
        Xi_hat = phi_hat @ self.Xi[-1,:]
        return Xi_hat 
    
    def get_NIRVAR_gmm_labels(self):
        """ 
        Method to retrieve the NIRAR GMM cluster labels.

        :return: labels. Array of integers where each integer labels a GMM cluster. Shape = (Q, N)
        :rtype: np.ndarray
        """
        similarity_matrix, labels = self.gmm() 
        return labels 
    
class LASSO():
    "Class to do LASSO estimation and prediction."
    def __init__(self,
                Xi : np.ndarray,
                ) -> None :
        """ 
        :param Xi: Design matrix of shaoe (T, N) 
        :type Xi: numpy.ndarray 

        """
        self.Xi = Xi 

    @property
    def T(self):
        T = self.Xi.shape[0] 
        return T

    @property
    def N(self):
        N = self.Xi.shape[1] 
        return N
    
    def fit_lasso(self, alpha: float):
        """
        Fit LASSO regression with a given alpha.
        
        :param alpha: Regularization parameter for LASSO
        :type alpha: float
        """
        self.model = Lasso(alpha=alpha,fit_intercept=False) 
        predictors = self.Xi[:self.T-1]
        targets = self.Xi[1:] 
        self.model.fit(predictors, targets)

    def fit_lasso_cv(self, alpha_values: list):
        """
        Fit LASSO using cross validation to select the optimal alpha.
        
        :param alpha_values: List of alpha values to search over
        :type alpha_values: list
        """
        self.model = MultiTaskLassoCV(alphas=alpha_values,fit_intercept=False) 
        predictors = self.Xi[:self.T-1]
        targets = self.Xi[1:] 
        self.model.fit(predictors, targets)
        print(f"Penalty chosen by Cross Validation : {self.model.alpha_}") 

    def fit_lasso_bic(self, alpha_values: list):
        """
        Fit LASSO using BIC to select the optimal alpha.
        
        :param alpha_values: List of alpha values to search over
        :type alpha_values: list
        """
        predictors = self.Xi[:self.T-1]
        targets = self.Xi[1:] 

        best_bic = np.inf
        best_alpha = None
        best_model = None

        # Loop over alpha values and compute BIC
        for alpha in alpha_values:
            model = Lasso(alpha=alpha,fit_intercept=False)
            model.fit(predictors, targets)
            
            # Calculate BIC for the model
            rss = np.sum((targets - model.predict(predictors)) ** 2)
            bic = (self.T -1)* np.log(rss / (self.T -1)) + np.log(self.T-1) * np.sum(model.coef_ != 0)

            if bic < best_bic:
                best_bic = bic
                best_alpha = alpha
                best_model = model

        self.model = best_model
        print(f"Penalty chosen by BIC : {best_alpha}") 
        return best_alpha, best_bic

    def predict_idiosyncratic_component(self, X_new: np.ndarray):
        """
        Predict using the fitted LASSO model.
        
        :param X_new: New data matrix of shape (T_new, N)
        :type X_new: numpy.ndarray

        :return: Predictions for the new data
        :rtype: numpy.ndarray
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        # Ensure X_new is the correct shape (T_new, N)
        if X_new.shape[1] != self.N:
            raise ValueError(f"Expected X_new to have {self.N} features, but got {X_new.shape[1]}")
        return self.model.predict(X_new)
    
    

        