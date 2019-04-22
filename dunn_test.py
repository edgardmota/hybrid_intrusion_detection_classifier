#!/usr/bin/env python3
import scikit_posthocs as sp
import pandas

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: not ('KNN' in x.upper() or 'SENSI' in x.upper()  or 'ESPECI' in x.upper()), sep='\t')
df = df.melt(var_name='groups', value_name='values')
sp.posthoc_dunn(df, val_col='values', group_col='groups').to_csv(path_or_buf='acc_final_comparison_dunn.csv')

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: 'SENSI' in x.upper(), sep='\t')
df = df.melt(var_name='groups', value_name='values')
sp.posthoc_dunn(df, val_col='values', group_col='groups').to_csv(path_or_buf='sensi_final_comparison_dunn.csv')

df = pandas.read_csv('final_comparison.csv', usecols=lambda x: 'ESPECI' in x.upper(), sep='\t')
df = df.melt(var_name='groups', value_name='values')
sp.posthoc_dunn(df, val_col='values', group_col='groups').to_csv(path_or_buf='especi_final_comparison_dunn.csv')
