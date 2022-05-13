#!/bin/python

import numpy as np
import pandas as pd

from statsmodels.tsa.ar_model import ar_select_order, AutoReg

class OnlineARIMA:
    '''
        Initialization
        @param series: 1D train data
    '''
    def __init__(self, series, maxlag=30, lrate=1.75, epsilon=10**(-0.5)):
        self.maxlag = maxlag
        self.lrate = lrate
        mod = ar_select_order(series, maxlag=maxlag)
        self.mask = mod.ar_lags
        res = AutoReg(series, lags = mod.ar_lags).fit()
        self.w = res.params[1:]
        self.window = series[-maxlag:].values.tolist()
        self.A_inv = np.eye(len(self.mask)) * epsilon
    
    '''
        Update
    '''
    def update(self, x):
        arr = self.window.copy()
        arr.append(0)
        arr = arr[::-1]
        mask = self.mask
        masked = np.array(arr)[mask]
        xhat = np.dot(masked, self.w)
        diff = xhat - x
        grad = 2 * diff * masked.reshape(1, -1)
        self.A_inv = self.A_inv - self.A_inv @ grad.T * grad * self.A_inv / (1 + grad * self.A_inv * grad.T)
        self.w = self.w - self.lrate * (grad * self.A_inv)[0]
        def update_array(array, update):
            updated = np.roll(array, -1)
            updated[-1] = update
            return updated
        self.window = update_array(self.window, x).tolist()
        return xhat