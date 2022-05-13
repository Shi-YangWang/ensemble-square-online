#!/bin/python

import numpy as np
import matplotlib as plt
import pandas as pd
import xgboost as xgb

'''
    compute permutation entropy and visualize in color gradient.
'''
def render_cross_entropy(series, chunk_size=60, order=3):
    ## 计算交叉熵
    from pyentrp import entropy as ent
    perm_ent_series = []
    for i in range(0, series.count()-chunk_size):
        perm_ent_series.append(ent.permutation_entropy(series[i: i+chunk_size], order=order))
    ## 创建 color map
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    jet = plt.get_cmap('jet')
    min = pd.Series(perm_ent_series).min()
    max = pd.Series(perm_ent_series).max()
    len = pd.Series(perm_ent_series).count()
    cNorm  = colors.Normalize(vmin=min, vmax=max, clip=True)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    ## 映射 color map
    cols = []
    for i in range(0, len):
        cols.append(colors.rgb2hex(scalarMap.to_rgba(perm_ent_series[i])).upper())
    ## 可视化
    x = range(0, len)
    y = series[0:len]
    pd.DataFrame({'x': x, 'y': y}).plot.scatter(x='x', y='y', c=cols, colormap='jet')