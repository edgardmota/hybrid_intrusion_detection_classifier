#!/usr/bin/env python3
from scipy import stats
import pandas

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: not ('KNN' in x.upper() or 'SENSI' in x.upper()  or 'ESPECI' in x.upper()), sep='\t')
print("ACC {}".format(stats.mstats.kruskalwallis(df)))

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: 'SENSI' in x.upper(), sep='\t')
print("SENSI {}".format(stats.mstats.kruskalwallis(df)))

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: 'ESPECI' in x.upper(), sep='\t')
print("ESPECI {}".format(stats.mstats.kruskalwallis(df)))
