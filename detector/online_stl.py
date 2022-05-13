#!/bin/python

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

class OnlineSTL:
    '''
        df_train: 1D Series, shape=(1,4*m)
        m       ：period
    '''
    def __init__(self, df_train, m=1440):
        self.m = m
        # Initialize
        # A.shape=(1, 4m)
        self.A = np.array(df_train).copy()
        result=seasonal_decompose(self.A, model='additive', period=m)
        self.K = np.array(result.seasonal).copy()
        self.S = np.array(self.K[-m:]).copy()
        self.T = np.array(self.K[-m:]).copy()
        self.D = np.array((df_train - result.seasonal)[-m:]).copy()
        self.i = 4 * m

        self.asym_filter_4m = self.async_filter(4*m)
        self.asym_filter_m  = self.async_filter(m)
    
    '''
        Tri cube kernel filter
    '''
    def tri_cube(self, u):
        if 0 <= u < 1:
            return ( 1 - u ** 3 ) ** 3
        else:
            return 0
    
    def async_filter(self, n):
        asym_filter_n = np.zeros((n))
        for j in range(0, n):
            asym_filter_n[-j-1] = self.tri_cube(j / n)
        return asym_filter_n

    '''
        Xi: Recent data point
        i : sequential index
    '''
    def update(self, Xi):
        # Update
        def update_array(array, update):
            updated = np.roll(array, -1)
            updated[-1] = update
            return updated
        update_array(self.A, Xi)
        b = Xi
        def apply_trend_filter(filter, signal):
            return np.dot(filter, signal) / np.linalg.norm(x=filter, ord=1)
        t1= apply_trend_filter(self.asym_filter_4m, self.A)
        d1=b - t1
        self.i = self.i + 1
        r = (self.i-1) % self.m
        gamma = 0.7
        self.S[r] = gamma * d1 + (1-gamma) * self.S[r]
        update_array(self.K, self.S[r])
        t4= apply_trend_filter(self.asym_filter_4m, self.K)
        d5= b - t1 - t4
        self.T[r] = gamma * d5 + (1-gamma) * self.T[r]
        b = b - self.T[r]
        update_array(self.D, b)
        Ti=apply_trend_filter(self.asym_filter_m, self.D)
        Si=self.T[r]
        Ri=Xi - Ti - Si
        # return Ti, Si, Ri
        return Ri