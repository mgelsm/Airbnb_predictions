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
    df_all['dac_year'] = dac[:,0]
    df_all['dac_month'] = dac[:,1]
    df_all['dac_day'] = dac[:,2]

    # Feature timestamp_first_active
    tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
    df_all['tfa_year'] = tfa[:,0]
    df_all['tfa_month'] = tfa[:,1]
    df_all['tfa_day'] = tfa[:,2]
    
    
    #### ---- Feature Processing ---- ####
    # Drop transformed and useless features
    df_all = df_all.drop(['id','date_first_booking','timestamp_first_active','date_account_created'], axis=1)
    
    # Categorical features
    feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
             'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
    
    # Convert  categorical features to dummy
    for f in feats:
        df_dummy = pd.get_dummies(df_all[f], prefix=f)
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
    return (y,label_enc)

#####TODO : DELETE

# @name trainXGB : Gradient boosting decision trees
# @arg[in] X_train : df of features (one_hot representation) for training
# @arg[in] X_train : target vector (scalar representation)
# @return xgb : trained model
def trainXGB(X_train, y_train):
    xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
    xgb.fit(X_train,y_train)
    return xgb

#####TODO : DELETE

# @name trainRandForest
# @arg[in] X_train : df of features (one_hot representation) for training
# @arg[in] X_train : target vector (scalar representation)
# @return rand_forest_model : trained model
def trainRandForest(X_train, y_train,n_estimators,max_depth):
    rand_forest_model = sklearn.ensemble.RandomForestClassifier(n_estimators=10,max_depth=6)
    rand_forest_model.fit(X_train,y_train)
    return rand_forest_model


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

#####TODO : DELETE

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

#####TODO : DELETE

def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


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


#####TODO : DELETE

# @arg[in] cts : list of 5 countries per user
# @arg[in] y_labels : Label of the user. (True Destination)
# @return ndcg_score_final : ndcg score for this batch of data
def computeNDCG_batch(cts, y_labels):

    # Transform results into a single array. The best prediction comes first, the least accurate comes last.
    # There are 5 predictions per user
    predictions = np.zeros((len(y_labels), k))
    for i in range(len(y_true)):
        for j in range(k):
            if y_labels[i] == cts[i+j]:
                predictions[i,j] = 1

    # Compute the ndcg score for each user            
    score_array = []
    for array in predictions:
        score = machine_learning_helper.ndcg_at_k(array, 5 , method = 1)
        score_array.append(score)

    # NDCG is the mean of all the user scores         
    ndcg_score_final = np.mean(score_array)
    return ndcg_score_final