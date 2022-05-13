#!/bin/python

import numpy as np
import pandas as pd

class GPD:
    '''
        @param m period
    '''
    def __init__(self, x_train, m=1440, theta=0.01):
        self.theta=theta
        self.window = np.array(x_train.copy()[-m:])
    
    '''
        @return True/False if residual of current step is judged an outlier.
    '''
    def step(self, residual):
        def update_array(array, update):
            updated = np.roll(array, -1)
            updated[-1] = update
            return updated
        self.window = update_array(self.window, residual)
        # me
        miu  = pd.Series(self.window).mean()
        std  = pd.Series(self.window).std()
        sigma= miu ** 3 / 2 / (std ** 2 + 1)
        k    = miu ** 2 / 2 / (std ** 2 - 1)
        print(sigma)
        print(k)
        # X obeys GPD(sigma, k)
        def F(sigma, k, x):
            if (abs(k) < 1e-3):
                return 1 - np.exp(-x/sigma)
            return 1 - (1 - k * x / sigma) ** (1/k)
        x_range = np.linspace(self.window.min() - 1, self.window.max() + 1, 500)
        x_prob = pd.Series(x_range).apply(lambda x: F(sigma, k, x))
        idx = pd.Series(x_prob - (1-self.theta)).abs().argmin()
        print(x_range[idx])
        if residual > x_range[idx]:
            return 1
        return 0
    
    def plot(self):
        raise Exception("not implemented")