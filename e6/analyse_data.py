#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import sys
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

#df = pd.read_csv("data.csv")

filename = sys.argv[1]
df = pd.read_csv(filename)

#From lecture notes
#ANalysis Of VAriance(ANOVA)
x1=df['qs1']
x2=df['qs2']
x3=df['qs3']
x4=df['qs4']
x5=df['qs5']
x6=df['merge1']
x7=df['partition_sort']

anova=stats.f_oneway(x1,x2,x3,x4,x5,x6,x7)
print(anova)
#print(anova.pvalue)

#Post Hoc Analysis
#From lecture notes
melt_df=pd.melt(df)
print("melt_df:",melt_df)

posthoc=pairwise_tukeyhsd(melt_df['value'],melt_df['variable'],alpha=0.05)
print("Posthoc:",posthoc)
print("Mean:", df.mean())
fig=posthoc.plot_simultaneous()
plt.show()

#So, merge1,partition_sort have different means
#partition_sort and qs1 have different means
#qs1 and qs2 have different means
#merge1 and qs1 have different means
# and so on





