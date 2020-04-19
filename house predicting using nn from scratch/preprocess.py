# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:31:49 2020

@author: subham
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split




def load_data():
    datset=pd.read_csv('housepricedata.csv')
    dataset=datset.values
    X = dataset[:,0:10]
    Y = dataset[:,10]
    min_max=preprocessing.MinMaxScaler()
    X_scale=min_max.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3)
    return X_train, X_test, Y_train, Y_test