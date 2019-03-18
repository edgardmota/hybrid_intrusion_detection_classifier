#!/usr/bin/env python3
import scikit_posthocs as sp
import pandas

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: not ('KNN' in x.upper()), sep='\t')
df = df.melt(var_name='groups', value_name='values')
sp.posthoc_dunn(df, val_col='values', group_col='groups').to_csv(path_or_buf='final_comparison_dunn.csv')
