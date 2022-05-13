#!/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Voting:
    def __init__(self):
        return
    
    def ensemble(self, series):
        if series.sum() * 2 >= series.count():
            return 1
        return 0

def main():
    voting = Voting()
    print(voting.ensemble(pd.Series([0, 0, 0])))
    print(voting.ensemble(pd.Series([1, 0, 0])))
    print(voting.ensemble(pd.Series([1, 0, 1])))