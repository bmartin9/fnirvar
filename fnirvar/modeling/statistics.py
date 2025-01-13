""" 
Script containing methods to calculate various statistics given input predictions and targets.
"""

#!/usr/bin/env python3 
# USAGE: ./statistics.py 

import numpy as np 
from scipy.stats import spearmanr
from scipy.stats import rankdata


class benchmarking():

    """
    Class to compute daily benchmarking statistics when doing backtesting.

    :param predictions: Predicted returns for each day. Shape = (N) 
    :type predictions: np.ndarray 

    :param market_excess_returns: Excess returns on the prediction day. Equal to Raw returns minus SPY returns. Shape = (N) 
    :type market_excess_returns: np.ndarray 

    :param yesterdays_predictions: Predictions from the previous day. Shape = (N). Used to determine if a transaction occurred. 
    :type yesterdays_predictions: np.ndarray 

    :param transaction_cost: cost of a single transaction in bpts
    :type transaction_cost: int

    :return: None
    :rtype: None
    """
    def __init__(self,
                 predictions : np.ndarray,
                 market_excess_returns : np.ndarray,
                 yesterdays_predictions : np.ndarray,
                 transaction_cost : int = 0
                 ) -> None:
        
        self.predictions = predictions 
        self.market_excess_returns = market_excess_returns 
        self.yesterdays_predictions = yesterdays_predictions
        self.transaction_cost = transaction_cost 

    @property
    def N(self):
        N = self.predictions.shape[0] 
        return N
    
    def hit_ratio(self) -> float: 
        """ 
        :return: ratio
            The fraction of predictions with the same sign as market excess returns
        :rtype: float
        """
        is_correct_sign = np.sign(self.predictions)*np.sign(self.market_excess_returns) 
        is_corr_ones = np.where(is_correct_sign==1,1,0)
        ratio = np.sum(is_corr_ones)/(self.N)
        return ratio
    
    def long_ratio(self) -> float:
        """ 
        :return: long_ratio 
            The fraction of predictions with sign +1
        :rtype: float
        """
        prediction_sign = np.sign(self.predictions) 
        is_corr_ones = np.where(prediction_sign==1,1,0)
        long_ratio = np.sum(is_corr_ones)/(self.N)
        return long_ratio
    
    def corr_SP(self) -> float: 
        """ 
        :return: corr_SP 
            The Spearman correlation between your predictions and the target market excess returns
        :rtype: float 
        """
        rho_sp , p = spearmanr(self.predictions,self.market_excess_returns) 
        return rho_sp 
    
    def PnL(self, quantile : float) -> float: 
        """
        :param quantile: The top x largest (in magnitude) predictions where x ∈ [0,1].
        :type quantile: float 

        :return PnL: The quantile PnL where PnL is defined as \sum_{all_stocks} sign(predictions)*market_excess_returns
        :rtype PnL: float 
        """

        prediction_ranks = rankdata(np.abs(self.predictions),method='min') 
        cutoff_rank = self.N*(1-quantile) 
        quantile_predictions = np.where(prediction_ranks>=cutoff_rank,self.predictions,0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(signed_predictions*self.market_excess_returns)
        return PnL 
    
    def transaction_indicator(self):
        """ 
        :return: transaction_indicator
            1 if a transaction occured, 0 otherwise. Shape = (N)
        :rtype: np.ndarray
        """
        transaction_indicator = np.where(np.sign(self.predictions)-np.sign(self.yesterdays_predictions)==0,0,1)
        return transaction_indicator
    
    def weighted_PnL_transactions(self, weights : np.ndarray , quantile : float) -> float: 
        """ 
        :param quantile: The top x% largest (in magnitude) predictions where x ∈ [0,1].
        :type quantile: float 

        :param weights: The portfolio weightings for each stock. Shape = (N)
        :type weights: np.ndarray 

        :return PnL: The quantile PnL where PnL is defined as ∑_{all_stocks} sign(predictions)*market_excess_returns
        :rtype PnL: float 
        """

        prediction_ranks = rankdata(np.abs(self.predictions),method='min') 
        cutoff_rank = self.N*(1-quantile) 
        quantile_predictions = np.where(prediction_ranks>=cutoff_rank,self.predictions,0)
        quantile_weights = np.where(prediction_ranks>=cutoff_rank, weights, 0)
        quantile_transaction_indicator = np.where(prediction_ranks>=cutoff_rank,self.transaction_indicator(),0)
        signed_predictions = np.sign(quantile_predictions)
        PnL = np.sum(weights*(signed_predictions*self.market_excess_returns - (self.transaction_cost)*quantile_transaction_indicator)) 
        portfolio_size = np.sum(quantile_weights) 
        PnL = PnL/portfolio_size
        return PnL
    

