#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:03:38 2020

@author: glauber
"""
# !cut -f1 -d',' /home/glauber/Documents/mark_project/adelia_sampaio/patient/rank_3S.csv

import pandas as pd
import numpy as np


def ObervedEnrichment(df, net):
    df.columns = ['patient_sample','cluster'] # put in column names
    
    clusters = df.cluster.unique()
    
    Merged = pd.DataFrame() # empty dataframe
    
    for cluster in clusters:
        
        df2 = df.loc[df['cluster'] == cluster] # select rows
        top = df2.patient_sample # patients
        net2 = net[net.patient_sample.str.contains('|'.join(top))] # select patients
      
        count = pd.DataFrame(net2.phenotype.value_counts()) # series to dataframe
        count['cluster'] = cluster # add a new column with indexes from X1
        Merged = pd.concat([Merged,count], axis=0, sort=True) # merge two dataframes
    
    Index = Merged.index
    Merged['treatment'] = Index
    
    df2 = Merged.pivot(index='treatment', columns='cluster', values='phenotype') 
    
    return df2




def ExpectedEnrichment(cluster_members, runs, observed, input_net, selec):

    df = cluster_members
    df.columns = ['patient_sample','cluster'] # put in column names
    df['cluster'] = 'cluster_' + df['cluster'].astype(str) # change column values
    
    query0 = pd.read_csv(observed)
    #query0 = observed # observed values
    query1 = query0.add_prefix('cluster_') # change column name / replace column names
    query = query1.rename(columns={'cluster_treatment': 'phenotype'}) # change column name / replace column names
    uniqueClusters = query.columns
    
    
    net = input_net
   
    
    if selec == 0:
        net = net[net.treatment.str.contains('|'.join(['_DMSO_00_none']))] # single drug only
    
    net.drop('treatment', axis = 'columns', inplace = True) # remove column/drop column
    net.columns = ['sample_well','phenotype'] # put in column names
    #net = net1[~net1.sample_well.str.contains('|'.join(jaks))] # select nonJAKs
    
    uniquePhenotypes = pd.DataFrame(net.phenotype.value_counts())
    uniquePhenotypes.drop('phenotype', axis = 'columns', inplace = True) # remove column/drop column
    
     
    for Myquery in uniqueClusters: # list of clusters
        if Myquery != 'phenotype':
            selectedQuery = pd.DataFrame(query[Myquery])
            #selectedQuery = pd.DataFrame(query.iloc[:, Myquery])
            
            selectedQuery.index = query.phenotype # problem here (original)
            #selectedQuery.index = query.treatment # problem here
            ExpectedValues = randomizeNetwork(uniquePhenotypes, runs, df, net, Myquery)
            ExpectedValues = pd.concat([ExpectedValues, selectedQuery], axis=1, sort=True)
            Exp = ExpectedValues.fillna(0)
            Obs = Exp[Myquery]
            Exp1 = Exp.drop([Myquery], axis = 'columns')  
                           
            Zscore2 = pd.DataFrame(getZscore(Exp1, Obs))
            Zscore2.columns = [Myquery]
            
            uniquePhenotypes = pd.concat([uniquePhenotypes, Zscore2], axis=1, sort=True)
            
            ExpectedValues = pd.DataFrame()
            Exp = pd.DataFrame()
            Exp1 = pd.DataFrame()
            selectedQuery = pd.DataFrame()
            Obs = pd.DataFrame() 
     
    return uniquePhenotypes



def randomizeNetwork(uniquePhenotypes, runs, df, net, Myquery):
    
    for i in range(runs): # matrix iteration / array iteration
       
        df = df.loc[df['cluster'] == Myquery] # select rows
        top = df['patient_sample'] 
        
        net.sample_well = np.random.permutation(net.sample_well.values) # randomize column
        net2 = net[net.sample_well.str.contains('|'.join(top))] # select patients
           
        count = net2.phenotype.value_counts() 
         
        uniquePhenotypes = pd.concat([uniquePhenotypes, count], axis=1, sort=True) # merge two Series into a DataFrame (hash5.pl-like)  
        
    return uniquePhenotypes


def getZscore(ExpectedValues, Myquery):

    """ExpectedValues = array with expected values
    Myquery = array column with values
    Returns a column array with Z-scores
    """
    MeanArray = ExpectedValues.mean(axis=1) # mean row wise
    StdArray = ExpectedValues.std(axis=1) # SD row wise
    
    A = Myquery - MeanArray
    Zscore = divideByZero(A, StdArray) # Zscore = A/StdArray 

    return Zscore



def divideByZero(a, b): # this function deals with division by zero/divide by zero
    """ 
    1) ignore / 0, divideByZero( [-1, 0, 1], 0 ) -> [0, 0, 0] 
    2) a and b can be arrays
    """
    
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        
    return c


#%%

# clusters = 'cluster_membership_K7.csv'  # cluster membership
# clusters = '/home/glauber/Documents/mark_project/redo/experiment5/stability_K6.csv'  # cluster membership
clusters = 'stability_K5.csv'  # cluster membership

# input_net = '/home/glauber/Documents/mark_project/march/net_allTreatments.csv'
input_net = 'net_allTreatments.csv'
net = pd.read_csv(input_net)
net = net.loc[net['treatment'] == '_3S_DMSO_00_none']  
net.columns = ['patient_sample','treatment', 'phenotype'] # put in column names
    
cluster_membership = pd.read_csv(clusters) # patient_sample,cluster
df = cluster_membership[['patient_sample', 'run_0']]
#df.drop('Unnamed: 0', axis = 'columns', inplace = True)

# df['run_0'].value_counts()   # df['run_0'].value_counts().sort_index()


observed_df = ObervedEnrichment(df, net) 

sample = open('temp2.csv', 'w') # open file to write
observed_df.to_csv(sample, index=True, header=True, sep=',') # print dataframe
sample.close()  


runs_enrichment = 1000
observed = 'temp2.csv' # observed enrichment
selec_DMSO = 0 # 0 = DMSO    1 = JAKs


cluster_members = df
runs = runs_enrichment
observed = observed
input_net = net
selec = selec_DMSO

enrichment = ExpectedEnrichment(cluster_members, runs, observed, input_net, selec)


sample2 = open('new_enrichment_K6_30K.csv', 'w') # open file to write
enrichment.to_csv(sample2, index=True, header=True, sep=',') # print dataframe
sample2.close() 


#%%


#%%
