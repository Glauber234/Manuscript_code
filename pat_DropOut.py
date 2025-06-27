#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Apr  4 14:44:53 2022

@author: glauber
'''


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')


def clean(x):
    if x == 'hi':
        return 1
    elif x == 'lo':
        return 0


n_features = 40 
label = 'risk'

df_labels = pd.read_csv('labels.csv')
df_labels = df_labels[['patient_code', 'risk']]
df_labels = df_labels.loc[df_labels['risk'] != 'X']
df_labels.drop_duplicates(subset = ['patient_code'], inplace=True)  

df0 = pd.read_csv('distances.csv') 

Merged = pd.DataFrame()
df = df0.loc[df0['features (n)'] == n_features]

dfx = pd.pivot_table(df, values='distance', index=['pat1'], columns=['pat2']).reset_index()
dfx.rename(columns = {'pat1': 'patient_code'}, inplace = True)

#%%
  
dfx = pd.merge(dfx, df_labels, how='inner', on='patient_code')
dfx[label] = dfx[label].apply(lambda x: clean(x))


patient_code = dfx[['patient_code', 'risk']]

for patient in dfx['patient_code'].unique():
          
    df3 = dfx.copy()
    df3.drop([patient], axis = 'columns', inplace=True)
    df3 = df3.loc[df3['patient_code'] != patient]  
    patient_code2 = df3[['patient_code']]
    df3.drop(['patient_code', label], axis = 'columns', inplace = True)
    
    selected_data = df3.values
    # linkage = {‘ward’, 'complete', 'average', 'single'}, default=’ward’ 
    # affinity='euclidean' , 'precomputed', 'correlation'
    clustering_model = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='complete')
    clustering_model.fit(selected_data)
    yhat = clustering_model.labels_   
    patient_code2[patient] = yhat
    patient_code = pd.merge(patient_code, patient_code2, how='left', on='patient_code')


sample2 = open('temp.csv', 'w', newline = '\n') # windows
patient_code.to_csv(sample2, index=False, header=True, sep=',') 
sample2.close() 

#%%



#%%


