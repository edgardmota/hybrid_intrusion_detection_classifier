#!/usr/bin/env python3
from scipy import stats
import pandas

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: not ('KNN' in x.upper()), sep='\t')
print(stats.mstats.kruskalwallis(df))
