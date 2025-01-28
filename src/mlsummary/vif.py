"""Module for Variance Inflation Factor computation. 

Implemented methods:

- vif

"""
import numpy as np
from typing import List, Dict, Optional


__all__=['vif']


def vif(x: np.ndarray, 
        columns: Optional[List[int]]=None, 
        method='linregress') -> Dict:
    """Function to compute the Variance Inflation Factor (VIF).

    :param x: Input data
    :type x: np.ndarray.
    :param columns: Columns indexes to be considered.
    :type columns: List[int].
    :return: {columns, vif, tolerance}
    :rtype: Dict.

    Notes:
    Assume we have a list of exogenous variable [X1, X2, X3, X4].
    To calculate the VIF and Tolerance for each variable, we regress
    each of them against other exogenous variables. For instance, the
    regression model for X3 is defined as: 
        X3 ~ X1 + X2 + X4
    And then we extract the R-squared from the model to calculate:
        VIF = 1 / (1 - R-squared)
        Tolerance = 1 - R-squared
    The cutoff to detect multicollinearity:
        VIF > 10 or Tolerance < 0.1

    """

    __n, __m = x.shape
    if columns is None:
        columns = list(range(__m))
    
    if method == 'linregress':
        __vv = __linregress(x=x, columns=columns)
    elif method == 'inv':
        __corr = np.corrcoef(x, rowvar=False)
        try:
            __vv = np.linalg.inv(__corr).diagonal()
            __vv = np.abs(__vv)
        except np.linalg.LinAlgError:
            # __vv = np.linalg.pinv(__corr, hermitian=True).diagonal()
            u, s, v = np.linalg.svd(__corr)
            __vv = np.dot(v.transpose(), np.dot(np.diag(s**-1), u.transpose())).diagonal()
            __vv = np.abs(__vv)

    return {'columns': columns, 
            'vif': __vv, 
            'tolerance': 1./__vv}


def __linregress(
        x: np.ndarray, 
        columns: Optional[List[int]]=None
    ) -> np.ndarray:
    """Function to compute the variance inflation factors 
    using linear regression of 
        
        x[k] ~ x[1] + x[2] + ... + x[k-1] + x[k+1] + ... + x[m],
    
    where k < m, m is the number of variables. Then 
        
        VIF[k] = 1/(1 - Rsq[k]),  k=1,2,...,m 
    with Rsq[k] is the coefficient of determination of the regression 
    of k-th variable. For more details see: https://en.wikipedia.org/wiki/Variance_inflation_factor

    :param x: Input data
    :type x: np.ndarray.
    :param columns: Columns indexes to be considered.
    :type columns: List[int].
    :return: {columns, vif, tolerance}
    :rtype: Dict.
    """
    __n, __m = x.shape
    if columns is None:
        columns = list(range(__m))

    __vv = np.zeros(__m)
    for i in columns:
        __not_i = [j for j in columns if j != i]
        __A = np.column_stack([np.ones(__n), x[:, __not_i]])
        __y = x[:,i]
        __reg = np.linalg.lstsq(__A, __y, rcond=None)
        __sse = np.sum((__y - __A@__reg[0])**2)
        __tse = np.sum((__y - __y.mean())**2)
        __r2 = 1 - (__sse / __tse)
        __vv[i] = 1 / (1 - __r2) if (__r2 != 1.) else 1e15

    return __vv

