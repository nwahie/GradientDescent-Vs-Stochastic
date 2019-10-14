# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:34:40 2019

@author: nitin
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score,f1_score, precision_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# Batch Gradient
eta = 0.2
m=100
theta= np.random.randn(2,1)
for i in range(1000):
    grad = 2/m*X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta*grad

# Stochastic Gradient
epoch=50
t0,t1 = 5,50
theta1= np.random.randn(2,1)
def learning_schedule(t):
    return t0/(t+t1)
for i in range(epoch):
    for j in range(100):
        index = np.random.randint(m)
        x = X_b[index:index+1]
        y1 = y[index:index+1]
        grad = 2*x.T.dot(x.dot(theta1)-y1)
        eta = learning_schedule(m*i+j)
        theta1 = theta1- eta*grad
from sklearn.linear_model import SGDRegressor

reg =   SGDRegressor(n_iter=50,eta0=0.1, penalty=None )
reg.fit(X,y.ravel())
reg.intercept_,reg.coef_


   