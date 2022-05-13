#!/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Merge:
    def __init__(self):
        return
    
    def ensemble(self, series):
        if series.sum() > 0:
            return 1
        return 0

def main():
    merge = Merge()
    print(merge.ensemble(pd.Series([0, 0, 0])))
    print(merge.ensemble(pd.Series([1, 0, 0])))
    print(merge.ensemble(pd.Series([1, 0, 1])))