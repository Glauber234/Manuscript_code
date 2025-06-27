#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Apr  4 14:44:53 2022

@author: Glauber Costa Brito
'''



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import warnings
warnings.filterwarnings('ignore')
############################ PERFORMANCE ############################################
from sklearn.metrics import precision_score, recall_score, f1_score

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


df_dropout = pd.read_csv('perc_dropout_n40.csv')
df_dropout.drop([label], axis = 'columns', inplace = True)
df_dropout = df_dropout[['patient_code', '% high']]
df_dropout.fillna(10, inplace = True) 

df = pd.read_csv('distances.csv')
df = df.loc[df['features (n)'] == n_features]
#%%
dfx = pd.pivot_table(df, values='distance', index=['pat1'], columns=['pat2']).reset_index()
dfx.rename(columns = {'pat1': 'patient_code'}, inplace = True) 

dfx = pd.merge(dfx, df_labels, how='inner', on='patient_code')
dfx = pd.merge(dfx, df_dropout, how='inner', on='patient_code')


dfx[label] = dfx[label].apply(lambda x: clean(x))

df3 = dfx.copy()

y = df3[[label]].values
df3.drop(['patient_code', label, '% high'], axis = 'columns', inplace = True)

X = df3.values
#%%

from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# create a list of base-models
def get_models():
 	models = list()
 	models.append(LogisticRegression(solver='liblinear'))
 	models.append(DecisionTreeClassifier())
 	models.append(SVC(gamma='scale', probability=True))
 	models.append(GaussianNB())
 	models.append(KNeighborsClassifier())
 	models.append(AdaBoostClassifier())
 	models.append(BaggingClassifier(n_estimators=10))
 	models.append(RandomForestClassifier(n_estimators=10))
 	models.append(ExtraTreesClassifier(n_estimators=10))
	
 	return models

# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models):
	meta_X, meta_y = list(), list()
	# define split of data
	kfold = KFold(n_splits=10, shuffle=True)
	# enumerate splits
	for train_ix, test_ix in kfold.split(X):
		fold_yhats = list()
		# get data
		train_X, test_X = X[train_ix], X[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
		# fit and make predictions with each sub-model
		for model in models:
			model.fit(train_X, train_y)
			yhat = model.predict_proba(test_X)
			# store columns
			fold_yhats.append(yhat)
		# store fold yhats as columns
		meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)


# fit all base models on the training dataset
def fit_base_models(X, y, models):
	for model in models:
		model.fit(X, y)


# fit a meta model
def fit_meta_model(X, y):
	model = LogisticRegression(solver='liblinear')
	model.fit(X, y)
	return model


# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
	for model in models:
		yhat = model.predict(X)
		acc = accuracy_score(y, yhat)       
		prec = precision_score(y, yhat)
		rec = recall_score(y, yhat)
		f1 = f1_score(y, yhat)
		print(model.__class__.__name__, acc, prec, rec, f1, file = sample, sep=',')
        


def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
		yhat = model.predict_proba(X)
		meta_X.append(yhat)
	meta_X = hstack(meta_X)
	# predict
	return meta_model.predict(meta_X)


sample = open('temp.csv', 'w') # open file to write
print('model', 'accuracy', 'precision', 'recall', 'F1', file = sample, sep=',')


for m in range(0, 5):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify = y)
    # print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
    # get models
    models = get_models()
    # get out of fold predictions
    meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, models)
    # print('Meta ', meta_X.shape, meta_y.shape)
    # fit base models
    fit_base_models(X_train, y_train, models)
    # fit the meta model
    meta_model = fit_meta_model(meta_X, meta_y)
    # evaluate base models
    evaluate_models(X_val, y_val, models)
    # evaluate meta model
    yhat = super_learner_predictions(X_val, models, meta_model)
    
    ensemble_acc = accuracy_score(y_val, yhat)
    ensemble_prec = precision_score(y_val, yhat)
    ensemble_rec = recall_score(y_val, yhat)
    ensemble_f1 = f1_score(y_val, yhat)

    print('ensemble', ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1, file = sample, sep=',') # accuracy_score


sample.close()

##%% take the mean and std
import pandas as pd
df = pd.read_csv('temp.csv')

df2 = df.groupby(['model']).agg(
    {'accuracy': 'mean',
     'precision': 'mean',
     'recall': 'mean',
     'F1': 'mean'
    }).reset_index()


sample2 = open('mean.csv', 'w', newline = '\n') 
df2.to_csv(sample2, index=False, header=True, sep=',') 
sample2.close() 

df4 = df.groupby(['model']).agg(
    {'accuracy': 'std',
     'precision': 'std',
     'recall': 'std',
     'F1': 'std'
    }).reset_index()

sample2 = open('std.csv', 'w', newline = '\n') 
df4.to_csv(sample2, index=False, header=True, sep=',') 
sample2.close() 

#%%



#%%

