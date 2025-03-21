"""
Module to measure the degree of association of nominals variables.

Implemented methods:

    - association
    - association_matrix

"""
from typing import List, Union, Optional
import math
import scipy.stats as ss
from scipy.stats.contingency import crosstab
from itertools import combinations
import numpy as np
from joblib import Parallel, delayed, parallel_backend


__all__=['association', 'association_matrix']


# Code adapted from: https://github.com/scipy/scipy/blob/main/scipy/stats/contingency.py
def association(
        x: np.array, 
        method: str="cramer", 
        correction: bool=False, 
        lambda_: Optional[Union[float, str]]=None
    )->float:
    """Calculates degree of association between two nominal variables.

    The function provides the option for computing one of three measures of
    association between two nominal variables from the data given in a 2d
    contingency table: Tschuprow's T, Pearson's Contingency Coefficient
    and Cramer's V.
    
    :param observed: The array of observed values.
    :type observed: array-like.
    :param method: The association test statistic. Must be one of 
    	{"cramer", "tschuprow", "pearson"}. Default = "cramer".
    :type method: str.
    :param correction: Inherited from `scipy.stats.contingency.chi2_contingency()`.
    :type correction: bool, optional.
    :param lambda_: Inherited from `scipy.stats.contingency.chi2_contingency()`.
    :type lambda_: float or str, optional.
    :return: Value of the test statistic.
    :rtype: float.
    
    """
    arr=np.asarray(x)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("`x` must be an integer array.")
    
    if len(arr.shape) != 2:
        raise ValueError("This method only accepts 2d arrays")
    
    chi2_stat = ss.chi2_contingency(arr, correction=correction, lambda_=lambda_)
    phi2 = chi2_stat[0] / arr.sum()
    n_rows, n_cols = arr.shape
    if method == "cramer":
        value = phi2 / min(n_cols - 1, n_rows - 1)
    elif method == "tschuprow":
        value = phi2 / math.sqrt((n_rows - 1) * (n_cols - 1))
    elif method == "pearson":
        value = phi2 / (1 + phi2)
    else:
        raise ValueError("Invalid argument value: 'method' argument must be 'cramer', 'tschuprow', or 'pearson'")
    
    return math.sqrt(value)


def __get_association(x: np.ndarray, i: int, j: int, **kwargs):
    """Function to compute the degree of association of two columns (i, j) 
    of the input array `x`.
    
    :param x: Array of data
    :type x: np.ndarray.
    :param i: Index of the first column.
    :type i: int.
    :param j: Index of the second column.
    :type j: int.
    :param kwargs: Additional params to be passed to `association` function.
    :type kwargs: Dict.
    :return: Association value.
    :rtype: float.
    """
    __observed = crosstab(x[:,i], x[:,j])[1]
    
    return association(__observed, **kwargs)


def association_matrix(x: np.ndarray, n_jobs: int=-1, verbose: int=0, **kwargs):
    __n = x.shape[1]
    __indexes = combinations(list(range(__n)), 2)
    __mat = np.eye(__n)

    def __func(i: int, j: int):
        return (i, j, __get_association(x=x, i=i, j=j, **kwargs))
    
    with parallel_backend(backend='threading'):
        __parallel = Parallel(n_jobs=n_jobs, return_as='generator', 
                              verbose=verbose)
        __res = __parallel(delayed(__func)(*t) for t in __indexes)

    for (i, j, v) in __res:
        __mat[i,j] = v
        __mat[j,i] = v
    
    return __mat
