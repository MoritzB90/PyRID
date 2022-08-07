# -*- coding: utf-8 -*-
"""
@author: Moritz F P Becker
"""

import numpy as np
import numba as nb
from numba.experimental import jitclass
# from scipy.special import erf, erfc, erfcinv
from math import erfc as erfc_scalar


#%%

#---------------------------------------
# Weighted Random Choice
#---------------------------------------

@nb.njit
def bisect_right(a, x):
    
    """Return the index where to insert item x in list a, assuming a is sorted.
    
    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.
    
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    
    Reduced copy from Cpython library: https://github.com/python/cpython/blob/3.10/Lib/bisect.py
    
    Parameters
    ----------
    a : `array_like`
        Array which to bisect
    x : `float64`
        Value which to insert in the sorted list
    
    
    Returns
    -------
    `int64`
        Index where to isnert x
        
    """

    lo = 0
    hi = len(a) - 1
    # mid = 0

    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1

    return lo

@nb.njit
def random_choice(population = None, weights = None, cum_weights = None):
    
    """Returns a weighted random element from a given list if population is passed, othewise returns the respective index.
    
    Parameters
    ----------
    population : `array_like`
        population from which to pick an element.
    weights : `float64[:]`
        Weights
    cum_weights : `float64[:]`
        Cumulative weights
    
    Raises
    ------
    TypeError("random_choice() missing required argument 'weight' or 'cum_weight'")
        Error raised if neither a weight nor a cumulative weight is passed.
    
    Returns
    -------
    `int64`
        Random element from the population or its index.
    
    
    """
    
    if weights is not None:
        weights_cum = np.cumsum(weights)
        
        x = np.random.rand()*weights_cum[-1]
        
        index = bisect_right(weights_cum, x)
        
        if population is None:
            return index
        else:
            return population[index]
    
    elif cum_weights is not None:
        
        x = np.random.rand()*cum_weights[-1]
        
        index = bisect_right(cum_weights, x)
        
        if population is None:
            return index
        else:
            return population[index]
    
    else:
        raise TypeError("random_choice() missing required argument 'weight' or 'cum_weight'")


#%%


@nb.njit
def erfc(x):
    
    """Calculates the complementary error function of all elements in the 1D Array x using the erfc fucntion from the math library, which, hwoever, only works on scalars.
    
    Parameters
    ----------
    x : `float64[:]`
        Array of whose elements to calculate erfc.
    
    
    Returns
    -------
    `float64[:]`
        Complementary error function of array x.
    
    """
    
    a = np.empty(x.shape)
    
    for i in range(len(x)):
        a[i] = erfc_scalar(x[i])
        
    return a

#%%

@nb.njit
def dx_cum_prob(x):
    
    """Cumulative probability distribution for the distance away from a plane of a diffusing particle after it crossing the plane :cite:p:`Kerr2008`.
    
    Parameters
    ----------
    x : `float64`
        Normalized distance
    
    Raises
    ------
    NotImplementedError (just an example)
        Brief explanation of why/when this exception is raised
    
    Returns
    -------
    `float64`
        Probability of a diffusing particle being x (normalized by the diffusion length coefficient) away from a plane after it crossing the plane.
    

    """
    
    p = 1-np.exp(-x**2)+np.sqrt(np.pi)*x*erfc(x)
    
    return p 


#%%

spec = [
    ('x_list', nb.float64[:]),
    ('y_list', nb.float64[:]),
    ('slopes', nb.types.ListType(nb.float64)),
]

@jitclass(spec)
class Interpolate:
    
    """Interpolation
    
    Source: https://stackoverflow.com/questions/7343697/how-to-implement-linear-interpolation
    """
    
    def __init__(self, x_list, y_list):
        
        self.slopes = nb.typed.List.empty_list(nb.float64)
        
        # if np.any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
        #     raise ValueError("x_list must be in strictly ascending order!")
        self.x_list = x_list
        self.y_list = y_list
        intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
        for x1, x2, y1, y2 in intervals:
            self.slopes.append((y2 - y1) / (x2 - x1))

    def call(self, x):
        if not (self.x_list[0] <= x <= self.x_list[-1]):
            raise ValueError("x out of bounds!")
        if x == self.x_list[-1]:
            return self.y_list[-1]
        i = bisect_right(self.x_list, x) - 1
        return self.y_list[i] + self.slopes[i] * (x - self.x_list[i])
    


#%%

# if __name__=='__main__':
    

