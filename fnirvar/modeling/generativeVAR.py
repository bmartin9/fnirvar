#!/usr/bin/env python3 
# USAGE: ./generativeSBM.py 

# Script to generate timeseries from the NIRVAR model 

import numpy as np
from sklearn.decomposition import TruncatedSVD 
from scipy.stats import invgamma
from scipy.stats import t as t_dist

class GenerateFNIRVAR:
    """
    This class simulates the factor process :math:`\\{F_t\\}` in a restricted dynamic factor
    model (FNIRVAR). It uses a VAR(:math:`l_F`) structure for the factors, along with
    optional user-provided or default coefficient matrices :math:`P` and shock mixing matrix
    :math:`N_0`. The default :math:`P` is scaled so that the block companion matrix has
    spectral radius :math:`\\rho_F < 1`, ensuring stationarity.

    :param l_F: The VAR order for the factors (number of lags).
    :type l_F: int

    :param T: The length of the time series to simulate.
    :type T: int

    :param r: The dimension of the factor vector :math:`F_t`.
    :type r: int

    :param q: The dimension of the common shocks :math:`u_t`.
    :type q: int

    :param P: Array of shape ``(l_F, r, r)`` containing the VAR coefficient matrices \
              :math:`[P_1, \\dots, P_{l_F}]``. If ``None``, default matrices will be generated.
    :type P: np.ndarray or None

    :param N0: A matrix of shape ``(r, q)`` that multiplies the shocks in the factor recursion. \
               If ``None``, a default matrix will be used.
    :type N0: np.ndarray or None

    :param rho_F: Desired spectral radius (in (0,1)) for the block companion matrix. Only used \
                  if :math:`P` is ``None``.
    :type rho_F: float

    :param random_state: Random State object for reproducibility. If ``None``, a new \
                         :class:`np.random.RandomState` is created.
    :type random_state: np.random.RandomState or None
    """

    def __init__(self, l_F, T, r, q,
                 P=None,
                 N0=None,
                 rho_F=0.95,
                 random_state=None):
        """
        Initialize the GenerateFNIRVAR class.

        :param l_F: The VAR order for the factors (number of lags).
        :type l_F: int

        :param T: The length of the time series to simulate.
        :type T: int

        :param r: The dimension of the factor vector :math:`F_t`.
        :type r: int

        :param q: The dimension of the common shocks :math:`u_t`.
        :type q: int

        :param P: Array of shape ``(l_F, r, r)`` containing the VAR coefficient matrices \
                  :math:`[P_1, \\dots, P_{l_F}]``. If ``None``, default matrices will be \
                  generated and scaled to have spectral radius :math:`rho_F`.
        :type P: np.ndarray or None

        :param N0: A matrix of shape ``(r, q)`` that multiplies the shocks in the factor recursion. \
                   If ``None``, a default matrix will be used.
        :type N0: np.ndarray or None

        :param rho_F: Desired spectral radius (in (0,1)) for the block companion matrix. Only used \
                      if :math:`P` is ``None``.
        :type rho_F: float

        :param random_state: Random State object. The seed value is set by the user upon \
                             instantiation of the Random State object.
        :type random_state: np.random.RandomState or None
        """
        self.l_F = l_F
        self.T = T
        self.r = r
        self.q = q
        self.rho_F = rho_F

        # Create a random state if none is provided
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

        # Assign or generate default P
        if P is None:
            P = self.default_P()
        if P.shape != (l_F, r, r):
            raise ValueError("P must have shape (l_F, r, r).")
        self.P = P

        # Assign or generate default N0
        if N0 is None:
            N0 = self.default_N0()
        if N0.shape != (r, q):
            raise ValueError("N0 must have shape (r, q).")
        self.N0 = N0

    def default_P(self):
        """
        Generate a default VAR coefficient tensor P of shape ``(l_F, r, r)``.
        Steps:
            1. Create random matrices from a normal distribution.
            2. Build the block companion matrix.
            3. Compute its spectral radius.
            4. Scale all P_k so that the new spectral radius = rho_F.

        :return: An array of shape ``(l_F, r, r)`` whose block companion matrix has spectral radius rho_F.
        :rtype: np.ndarray
        """
        # 1) Create random P of shape (l_F, r, r) with small entries
        scale = 0.1
        P_raw = self.random_state.normal(size=(self.l_F, self.r, self.r),scale=scale) 

        # 2) Build the block companion matrix
        comp_mat = self._build_companion(P_raw)

        # 3) Compute spectral radius
        eigvals = np.linalg.eigvals(comp_mat)
        current_rad = max(abs(eigvals))

        if current_rad == 0:
            # If random generation yields exactly 0 (extremely unlikely but possible),
            # we won't scale. That means P is 0 and trivially stable.
            return P_raw

        # 4) Scale
        scale_factor = self.rho_F / current_rad
        P_scaled = P_raw * scale_factor

        return P_scaled

    def _build_companion(self, P):
        """
        Build the block companion matrix of size (l_F*r, l_F*r) for a VAR(l_F) process.

        The top row is [P1, P2, ..., P_{l_F}],
        and the subdiagonal blocks form an identity structure for shifting the lags.

        :param P: An array of shape (l_F, r, r).
        :type P: np.ndarray

        :return: The block companion matrix of shape ((l_F*r), (l_F*r)).
        :rtype: np.ndarray
        """
        l_F, r, _ = P.shape
        dim = l_F * r
        comp_mat = np.zeros((dim, dim))

        # top row blocks
        for i in range(l_F):
            comp_mat[0:r, i*r:(i+1)*r] = P[i]

        # sub-diagonal identity blocks
        for i in range(1, l_F):
            row_start = i * r
            col_start = (i - 1) * r
            comp_mat[row_start:row_start+r, col_start:col_start+r] = np.eye(r)

        return comp_mat

    def default_N0(self):
        """
        Generate a default matrix N0 of shape ``(r, q)``.
        By default, we use an identity-like structure:
        - If r == q, returns the identity matrix of shape (r, r).
        - Otherwise, places 1's on the min(r,q) diagonal and 0 elsewhere.

        :return: A matrix of shape ``(r, q)``.
        :rtype: np.ndarray
        """
        if self.r == self.q:
            return np.eye(self.r, self.q)
        else:
            N0 = np.zeros((self.r, self.q))
            min_dim = min(self.r, self.q)
            for i in range(min_dim):
                N0[i, i] = 1.0
            return N0

    def generate_factors(self, burn_in=50):
        """
        Generate the factor process :math:`\\{F_t\\}` according to the recursion:

        .. math::
            F_t = \\sum_{k=1}^{l_F} P_k F_{t-k} + N_0 \\cdot u_t, \\quad u_t \\sim \\mathcal{N}(0, I_q).

        :param burn_in: Number of initial steps to discard for "burn-in" to reduce dependence on \
                        initial conditions.
        :type burn_in: int

        :return: A numpy array of shape ``(T, r)`` where each row is :math:`F_t^\\top`.
        :rtype: np.ndarray
        """
        total_length = self.T + burn_in
        F = np.zeros((total_length, self.r))

        # Generate shocks u_t ~ N(0, I_q)
        U = self.random_state.normal(size=(total_length, self.q))

        # Recursively compute the factor values
        for t in range(self.l_F, total_length):
            var_part = np.zeros(self.r)
            for k in range(1, self.l_F + 1):
                var_part += self.P[k - 1].dot(F[t - k])
            F[t] = var_part + self.N0.dot(U[t])

        return F[burn_in:]  # discard burn-in

    def generate_data(self, Lambda, xi = None):
        """
        Combine the generated factors :math:`F_t` with an idiosyncratic term :math:`\\xi_t`
        to produce the full observed data:

        .. math::
            X_t = \\Lambda F_t + \\xi_t.

        :param Lambda: A matrix of shape ``(N, r)`` containing the factor loadings.
        :type Lambda: np.ndarray

        :param xi: An array of shape ``(T, N)`` containing the idiosyncratic term for each time.
        :type xi: np.ndarray

        :return: A numpy array of shape ``(T, N)`` where each row is :math:`X_t^\\top`.
        :rtype: np.ndarray
        """
        F = self.generate_factors()
        X_factor = F.dot(Lambda.T)   # shape (T, N)
        if xi is None:
            print("No idiosyncratic term")
            return X_factor
        else:
            print("NIRVAR is the idiosyncratic term")
            return X_factor + xi


class GenerateNIRVAR():
    """ 
    :param random_state: Random State object. The seed value is set by the user upon instantiation of the Random State object.
    :type random_state: np.random.RandomState

    :param T: Number of observations (time points)
    :type T: int

    :param N: Number of Stocks
    :type N: int

    :param Q: Number of Features
    :type Q: int

    :param stock_names: List of stock names 
    :type stock_names: list

    :param feature_names: A list of feature names 
    :type feature_names: list

    :param B: The number of blocks in the SBM 
    :type B: int

    :param p_in: Probability of an edge forming between two in the same group for a particular feature
    :type p_in: float

    :param p_out: Probability of an edge forming between two in the different group for a particular feature
    :type p_out: float

    :param p_between: Probability of an edge forming between two over different features
    :type p_between: float

    :param categories: Dictionary with keys being the stock names and values being the corresponding groups
    :type categories: dict

    :param adjacency_matrix: Shape = (N,Q,N,Q). Gives the connections between stock-features. Entries are 1s or 0s.
    :type adjacency_matrix: np.ndarray

    :param phi_coefficients: Shape = (N,Q,N,Q). Gives the weighted connections between stock-features. Defines the VAR generative model. 
        Entries are real numbers. Has a spectral radius of <1.
    :type phi_coefficients: np.ndarray

    :param uniform_range: Distribution from which phi is sampled is U(-uniform_range,uniform_range)
    :type uniform_range: float

    :param innovations_variance: Variance of innovations, 
    :type innovations_variance: np.ndarray

    :param multiplier: Spectral radius of Phi
    :type multiplier: float

    :param global_noise: Variance of each innovation - set this if you want the same std for each stock innovation
    :type global_noise: float

    :param different_innovation_distributions: If False, the innovation distribution of each stock will be Normal(0,self.global_noise)
        If True, the innovation distribution of each stock will be Normal(0,sigma) with sigma ~ Inv-Gamma(3,2)
    :type different_innovation_distributions: bool

    :param phi_distribution: Shape = (NQ,NQ)
        A dense array of values for each Phi_{ij}^{(q)}. For example each Phi_{ij}^{(q)} could be drawn 
        from a some distribution that depends on the block membership of i and j. 
    :type phi_distribution: np.ndarray

    :param t_distribution: Whether you want t distributed innovations instead of normally distributed distributions
    :type t_distribution: bool

    :return: None
    :rtype: None
    """

    def __init__(self,
                 random_state : np.random.RandomState,
                 T : int, 
                 B : int,
                 N : int = None,
                 Q : int = None,
                 stock_names : list = None,
                 feature_names : list = None,
                 p_in : float = 0.9,
                 p_out : float = 0.05,
                 p_between : float = 0,
                 categories : dict = None,
                 adjacency_matrix : np.ndarray = None, 
                 phi_coefficients : np.ndarray = None,
                 uniform_range : float = 10,
                 innovations_variance : np.ndarray = None,
                 multiplier : float = 1,
                 global_noise : float = 1,
                 different_innovation_distributions : bool = False,
                 phi_distribution : np.ndarray = None,
                 t_distribution : bool = False
                 ) -> None:
 
        self.random_state = random_state
        self.T = T 
        self.B = B
        self.p_in = p_in
        self.p_out = p_out
        self.p_between = p_between
        self.uniform_range = uniform_range
        self.multiplier = multiplier
        self.different_innovation_distributions = different_innovation_distributions  
        self.global_noise = global_noise 
        self.t_distribution = t_distribution

        if N is None and stock_names is None:
            raise ValueError("You must specify either 'N' or 'stock_names'")
        if N is not None and stock_names is not None:
            if len(stock_names) != N:
                raise ValueError("Length of stock_names must be equal to N")
            self._N = N
            self._stock_names = stock_names 
        elif N is not None:
            self._N = N
            self._stock_names = ['{0}'.format(i) for i in range(N)] 
        elif stock_names is not None:
            self._stock_names = stock_names
            self._N = len(stock_names) 

        if Q is None and feature_names is None:
            raise ValueError("You must specify either 'Q' or 'feature_names'")
        if Q is not None and feature_names is not None:
            if len(feature_names) != Q:
                raise ValueError("Length of feature_names must be equal to Q")
            self._Q = Q
            self._feature_names = feature_names 
        elif Q is not None:
            self._Q = Q
            self._feature_names = ['{0}'.format(q) for q in range(Q)]  
        elif feature_names is not None:
            self._feature_names = feature_names
            self._Q = len(feature_names)  
        self.categories = categories if categories is not None else self.manual_categories()
        self.adjacency_matrix = adjacency_matrix if adjacency_matrix is not None else self.adjacency() 
        self.phi_distribution = phi_distribution if phi_distribution is not None else self.phi_blocks_distribution()
        self.phi_coefficients = phi_coefficients if phi_coefficients is not None else self.phi() 
        self.innovations_variance = innovations_variance if innovations_variance is not None else self.innovations_var()

    @property
    def N(self):
        return self._N 

    @N.setter
    def N(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("N must be a positive integer") 
        if value != len(self.stock_names):
            raise ValueError("Length of stock_names must be equal to N")
        self._N = value
        self._stock_names = ['{0}'.format(i) for i in range(value)] 

    @property
    def stock_names(self):
        return self._stock_names

    @stock_names.setter
    def stock_names(self, value):
        if not isinstance(value, list):
            raise ValueError("stock_names must be a list")
        if len(value) != self.N:
            raise ValueError("Length of stock_names must be equal to N")
        self._stock_names = value

    @property
    def Q(self):
        return self._Q 

    @Q.setter
    def Q(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Q must be a positive integer") 
        if value != len(self.feature_names):
            raise ValueError("Length of feature_names must be equal to Q")
        self._Q = value
        self._feature_names = ['{0}'.format(q) for q in range(value)] 

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        if not isinstance(value, list):
            raise ValueError("feature_names must be a list")
        if len(value) != self.Q:
            raise ValueError("Length of feature_names must be equal to Q")
        self._feature_names = value
    
    def random_categories(self):
        """ 
        Returns
        -------
        SBM_groupings : dict
        """
        block_labels = [np.argmax(self.random_state.multinomial(1,[1/self.B]*self.B)) for _ in range(self.N)]
        SBM_groupings = dict(zip(self.stock_names,block_labels)) 
        return SBM_groupings 
    
    def blocks(self) -> np.ndarray:
        """

        :return: blocks. Shape = (N,N). Represents the blocks defined by self.categories as a binary matrix
        :rtype: np.ndarray
        """
        SBM_groupings_matrix = np.zeros((self.N,self.N))
        SBM_groupings_values = list(self.categories.values())
        for i in range(self.N):
            for j in range(self.N):
                if SBM_groupings_values[i] == SBM_groupings_values[j]:
                    SBM_groupings_matrix[i][j] = 1 
                else:
                    continue 
        return SBM_groupings_matrix 
    
    def adjacency(self) -> np.ndarray:
        """ 

        :returns: adjacency_matrix 
            Shape = (N,Q,N,Q). Gives the adjacency matrix of the SBM defined by self.categories. 
            Connections within blocks are 1 with probability p_in.
            Connections in different blocks are 1 with probability p_out. 
            Connections between different features are 1 with probability p_between.
        :rtype: np.ndarray
        """
        blocks = self.blocks() 
        block_probabilities = np.where(blocks==1,self.p_in,self.p_out)
        P = self.p_between*np.ones((self.N,self.Q,self.N,self.Q)) 
        for q in range(self.Q):
            P[:,q,:,q] = block_probabilities 
        adjacency_matrix = self.random_state.binomial(1,P) 
        return adjacency_matrix
    
    def manual_categories(self):
        """ 
        :return: cat
            keys are the stock names, values are the block memberships
        :rtype: dict
        """
        vals = sorted([x%self.B for x in range(self.N)])
        keys = [str(x) for x in range(self.N)]
        cat = dict(zip(keys,vals))
        return cat
 
    def phi_blocks_distribution(self):
        """ 
        :return:  phi_dense.
            Shape = (NQ,NQ)
            Each Phi_{ij} ~ N(mean,1) where mean depends on the block membership of node i
        :rtype: np.ndarray
        """
        phi_dense = np.zeros((self.N,self.Q,self.N,self.Q))
        random_negative_mean = self.random_state.binomial(1,0.5,size=(self.B))
        random_negative_mean = np.where(random_negative_mean==1,-1,1)
        for i in range(self.N):
            mean = random_negative_mean[list(self.categories.values())[i]]*list(self.categories.values())[i]
            mean = 3*mean +10 
            phi_dense[i] = self.random_state.normal(loc=mean,scale=1,size=(self.Q,self.N,self.Q))
        phi_dense = np.reshape(phi_dense,(self.N*self.Q,self.N*self.Q),order='F')
        return phi_dense

    def phi(self) -> np.ndarray:
        """
        :return: phi 
            Shape = (N,Q,N,Q). Keeping zero edges, sample phi from a uniform distribution such that 
            the spectral radius of phi is <1 (for stationary solution of VAR model).
        :rtype: np.ndarray
        """
        connections = self.adjacency_matrix 
        connections = np.reshape(connections,(self.N*self.Q,self.N*self.Q),order='F')  
        phi = connections*self.phi_distribution 
        phi_eigs = np.linalg.eig(phi)[0]
        phi = (1/abs(np.max(phi_eigs)))*phi
        phi = self.multiplier*phi 
        phi_eigs = np.linalg.eig(phi)[0]
        phi = np.reshape(phi,(self.N,self.Q,self.N,self.Q),order='F')
        return phi 
    
    def innovations_var(self) -> np.ndarray: 
        """ 
        :return: var 
            Shape = (N,Q). Variance for each innovation
        :rtype: np.ndarray
        """
        if self.different_innovation_distributions:
            var = invgamma.rvs(a=3,loc=0,scale=2,size=(self.N,self.Q),random_state=self.random_state)
            return var 
        else:
            var = self.global_noise*np.ones((self.N,self.Q))
            return var 

    def generate(self) -> np.ndarray: 
        """ 
        :return: X_stored 
            Shape = (T,N,Q). Generated Time Series from VAR model.
        :rtype: np.ndarray
        """
        X_stored = np.zeros((self.T,self.N,self.Q))
        X = np.zeros((self.N,self.Q))
        if self.t_distribution:
            for t in range(self.T): 
                Z = t_dist.rvs(df=1,scale=self.global_noise,size=(self.N,self.Q))
                X = np.sum(np.sum(self.phi_coefficients*X,axis=2),axis=2) + Z 
                X_stored[t] = X

        else:
            for t in range(self.T): 
                Z = self.random_state.normal(0,np.sqrt(self.innovations_variance)) 
                X = np.sum(np.sum(self.phi_coefficients*X,axis=2),axis=2) + Z 
                X_stored[t] = X
        return X_stored 