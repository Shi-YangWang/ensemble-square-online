#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KernelDensity

class KDE:
    '''
        @param m period
    '''
    def __init__(self, x_train, m=1440, theta=0.01, bandwidth=1, kernel='epanechnikov'):
        self.theta=theta
        self.window = np.array(x_train.copy()[-m:])
        self.model = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    
    '''
        @return True/False if residual of current step is judged an outlier.
    '''
    def step(self, residual):
        def update_array(array, update):
            updated = np.roll(array, -1)
            updated[-1] = update
            return updated
        self.window = update_array(self.window, residual)
        self.model.fit(self.window[:, np.newaxis])
        x_range = np.linspace(self.window.min() - 1, self.window.max() + 1, 500)
        x_log_prob = self.model.score_samples(x_range[:, np.newaxis])
        x_prob = np.exp(x_log_prob)
        nx_prob = x_prob / x_prob.sum()
        idx = pd.Series(nx_prob.cumsum() - (1-self.theta)).abs().argmin()
        # cache plot parameters.
        self.x_range = x_range
        self.x_prob = x_prob
        if residual > x_range[idx]:
            return 1
        return 0
    
    def plot(self):
        plt.figure(figsize=(10, 10))
        r = plt.hist(
            x=self.window,
            bins=50,
            density=True,
            histtype='stepfilled',
            color='red',
            alpha=0.5,
            label='histogram',
        )
        plt.fill_between(
            x=self.x_range,
            y1=self.x_prob,
            y2=0,
            color='green',
            alpha=0.5,
            label='KDE',
        )
        plt.plot(self.x_range, self.x_prob, color='gray')
        # plt.vlines(x=2, ymin=0, ymax=r[0].max() + 0.01, color='k', linestyle='--', alpha=0.7)
        # plt.vlines(x=8, ymin=0, ymax=r[0].max() + 0.01, color='k', linestyle='--', alpha=0.7)
        plt.ylim(0, r[0].max() + 0.011)
        plt.legend(loc='upper right')
        plt.title('histogram and kde')
        plt.show()