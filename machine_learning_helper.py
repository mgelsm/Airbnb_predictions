# import 
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  
% matplotlib inline
import random
from datetime import datetime
from pandas.tools.plotting import scatter_matrix
from scipy.sparse import coo_matrix
import copy
import sklearn.neighbors, sklearn.linear_model, sklearn.ensemble, sklearn.naive_bayes # Baseline classification techniques
import scipy.io # Import data


# @name buildFeatureMat
# @arg[in] df : cleaned dataframe
# @return data frame as one-hot vector
def buildFeatsMat(df):
    df = df.drop(['id','date_first_booking','timestamp_first_active','date_account_created'])
    if 'country_destination' in df.columns: 
        df = df.drop(['country_destination'])
    feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
             'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    for f in feats:
        df_dummy = pd.get_dummies(df[f], prefix=f)
        df = df_train_users.drop([f], axis=1)
        df = pd.concat((df, df_dummy), axis=1)
    return df

# @name buildTargetMat
# @arg[in] cleaned data frame
# @return target vector as scalar
def buildTargetMat(df):
    labels = df['country_destination'].values
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(labels)   
    return y

# @name trainRandForest
# @arg[in] X_train : df of features (one_hot representation) for training
# @arg[in] X_train : target vector (scalar representation)
# @return rand_forest_model : trained model
def trainRandForest(X_train, y_train):
    rand_forest_model = sklearn.ensemble.RandomForestClassifier(n_estimators=100,max_depth=6)
    rand_forest_model.fit(X_train,y_train)
    return rand_forest_model


# @name predictCountries
# @arg[in] model (sklearn)
# @arg[in] X_test = df of features (one_hot representation) for testing
# @return y : predicted countries
def predictCountries(model,X_test):
    y = model.predict(X_test)
    return y 

