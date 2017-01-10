# import 
import pandas as pd
import numpy as np
import random
from datetime import datetime
from pandas.tools.plotting import scatter_matrix
from scipy.sparse import coo_matrix
import copy
import sklearn.neighbors, sklearn.linear_model, sklearn.ensemble, sklearn.naive_bayes # Baseline classification techniques
from sklearn import preprocessing
import scipy.io # Import data
from xgboost.sklearn import XGBClassifier


# @name buildFeatureMat
# @arg[in] df_train : cleaned dataframe of training users
# @arg[in] df_test : cleaned dataframe of testing users
# @return df_train, df_test : data frame as one-hot vector
def buildFeatsMat(df_train, df_test, df_sessions):
    df_train_len = df_train.shape[0]
    df_train = df_train.drop(['country_destination'],axis=1)
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, df_sessions, on='id', how='left', left_index=True)
    df_all = df_all.fillna(-1)
    df_all = df_all.drop(['id', 'date_first_booking','timestamp_first_active','date_account_created'], axis=1)
    
    feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
             'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    
    for f in feats:
        df_dummy = pd.get_dummies(df_all[f], prefix=f)
        df_all = df_all.drop([f], axis=1)
        df_all = pd.concat((df_all, df_dummy), axis=1)
        
    df_train = df_all[:df_train_len]
    df_test = df_all[df_train_len:]
    return (df_train,df_test)

# @name buildTargetMat
# @arg[in] cleaned data frame
# @return target vector as scalar
def buildTargetMat(df):
    labels = df['country_destination'].values
    label_enc = preprocessing.LabelEncoder()
    y = label_enc.fit_transform(labels)   
    return (y,label_enc)

# @name trainRandForest
# @arg[in] X_train : df of features (one_hot representation) for training
# @arg[in] X_train : target vector (scalar representation)
# @return rand_forest_model : trained model
def trainRandForest(X_train, y_train):
    xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
    #rand_forest_model = sklearn.ensemble.RandomForestClassifier(n_estimators=10,max_depth=6)
    xgb.fit(X_train,y_train)
    return xgb


# @name predictCountries
# @arg[in] model (sklearn)
# @arg[in] X_test = df of features (one_hot representation) for testing
# @return y : predicted countries
def predictCountries(model,X_test):
    y = model.predict(X_test)
    return y 

