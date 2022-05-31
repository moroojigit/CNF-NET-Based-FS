# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:05:12 2022

@author: Reza
"""

import numpy as np
import pickle as pk
import os
import sklearn
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt 
from scipy.io import savemat

def Best_Comb(Results):
    Avg_metrics = []
    for index, each_result in Results.items():
        Avg_metrics.append(abs(each_result["avg_score"]))
    Best_com_index = Avg_metrics.index(min(Avg_metrics))
    return Results[Best_com_index+1]['feature_idx'], np.asarray(Avg_metrics) 
    
#### Loading Dataset     
X = pk.load(open(os.path.join(os.getcwd(),"c499_X.pk"), 'rb'))
Y = pk.load(open(os.path.join(os.getcwd(),"c499_Y.pk"), 'rb'))

X = np.asarray(X)
Y = np.asarray(Y)

### Feature Selection 
from sklearn.neighbors import KNeighborsRegressor
Model = KNeighborsRegressor()
sfs1 = SFS(Model,
           k_features = 45, 
           forward = True,
           cv = 10,
           scoring = sklearn.metrics.SCORERS['neg_mean_squared_error'])

sfs1.fit(X,Y)
Results = sfs1.subsets_
SF, Metrics = Best_Comb(Results)

#SF = Results[3]['feature_idx']
fig = plot_sfs(sfs1.get_metric_dict())
plt.savefig('my_plot.png')
plt.close

Mydic = {"Selected_Features":SF, "Metrics":Metrics}
savemat('SF.mat',Mydic)

#### Removing nonselected features 
#X_train = X_train[:,SF]
#X_test = X_test[:,SF]
#
#### Normalization on training and testing sets 
#X_train = (X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
#X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
#
#### Loading LDA Classifier 
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#Model = LinearDiscriminantAnalysis()
#
#### Training the classifier model 
#Model.fit(X_train,L_train)
#
#### Predicting the testing samples
#Predict = Model.predict(X_test)
#
#### Evaluting the model 
#Score = Model.score(X_test,L_test)
#
