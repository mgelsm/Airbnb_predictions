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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer


# @name buildFeatureMat
# @arg[in] df_train : cleaned dataframe of training users
# @arg[in] df_test : cleaned dataframe of testing users
# @arg[in] df_sessions : cleaned dataframe of sessions
# @return df_train, df_test : dataframe as one-hot vector
def buildFeatsMat(df_train, df_test, df_sessions):
    
    # Concat train and test dataset so that the feature engineering and processing can be the same on the whole dataset
    df_train_len = df_train.shape[0]
    df_train = df_train.drop(['country_destination'],axis=1)
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    
    ## ---- Feature Engineering ---- ####
    # Features Session
    df_all = pd.merge(df_all, df_sessions, on='id', how='left', left_index=True)
    df_all = df_all.fillna(-1)
    
    # Feature date_account_created
    dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    df_all['dac_year'] = dac[:,0].astype(np.int8)
    df_all['dac_month'] = dac[:,1].astype(np.int8)
    df_all['dac_day'] = dac[:,2].astype(np.int8)

    # Feature timestamp_first_active
    tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
    df_all['tfa_year'] = tfa[:,0].astype(np.int8)
    df_all['tfa_month'] = tfa[:,1].astype(np.int8)
    df_all['tfa_day'] = tfa[:,2].astype(np.int8)
    
    
    #### ---- Feature Processing ---- ####
    # Drop transformed and useless features
    df_all = df_all.drop(['id','date_first_booking','timestamp_first_active','date_account_created'], axis=1)
    
    # Categorical features
    feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
             'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    
    # Convert  categorical features to dummy
    for f in feats:
        df_dummy = pd.get_dummies(df_all[f], prefix=f).astype(np.int8)
        df_all = df_all.drop([f], axis=1)
        df_all = pd.concat((df_all, df_dummy), axis=1)
    
    # Split again train and test dataset
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
    y = y.astype(np.int8)
    return (y,label_enc)


# @name predictCountries
# @arg[in] model (sklearn)
# @arg[in] X_test = df of features (one_hot representation) for testing
# @return y : predicted countries
def predictCountries(model,X_test,test_users_len):
    y = model.predict_proba(X_test)  
    #Taking the 5 classes with highest probabilities
    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(test_users_len):
        idx = id_test[i]
        ids += [idx] * 5
        cts += (np.argsort(y_pred[i])[::-1])[:5].tolist()
    return (ids,cts)


# @arg[in] y_pred : countries predicted by model.predict proba. Example : y_pred = model.predict_proba(X_test)  
# @arg[in] id_test : id of users example: df_test_users['id']
# @return cts : list of 5 countries per user
def get5likelycountries(y_pred, id_test):
    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(len(id_test)):
        idx = id_test[i]
        ids += [idx] * 5
        cts += (np.argsort(y_pred[i])[::-1])[:5].tolist()
    return cts,ids

