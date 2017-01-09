# Functions to preprocess Train_user and Test_user
# Pecoraro Cyril

import pandas as pd
import numpy as np
import os
from IPython.display import Image
from IPython.core.display import HTML 
import matplotlib.pyplot as plt  
% matplotlib inline
import random
from datetime import datetime
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import copy


def ageCleaning(df):
    # Finding users who put their birthdate instead of age in original dataframe
    df_birthyear = df[(df['age']>=1926) & (df['age']<=2001)]

    # Converting to age
    df_birthyear = copy.deepcopy(df_birthyear)
    df_birthyear['age'] = 2016-df_birthyear['age']

    # Replacing in original dataframe
    df.loc[(df['age']>=1926) & (df['age']<=2001), 'age'] = df_birthyear

    # Assigning a -1 value to invalid ages
    df = copy.deepcopy(df)
    df.loc[((df['age']<15) | (df['age']>90)), 'age'] = -1

    # Counting invalid ages:
    OutOfBoundsAgePercentage = round(100*len(df.loc[(df['age'] == -1), 'age'])/len(df),2)
    print('Percentage of users with irrelevant age',OutOfBoundsAgePercentage,'%')
    
    # Counting NaN ages:
    nanAgePercentage = round(100*df.isnull().values.sum()/len(df),2)
    print('Percentage of users with NaN age',nanAgePercentage,'%')
    
    # Assigning a -1 value to NaN ages
    df = copy.deepcopy(df)
    df['age'].fillna(-1) 
    print('All the invalid or missing age where replaced by value -1')
    
    return df

def exportInvalidAge(df):
    #invalid age
    df_invalid_age = df.loc[(df['age']==-1), ['id']]
    df = df[df['age'] != -1]

    #not specified age
    df_invalid_age= pd.concat([df_invalid_age, (df[df['age'].isnull()])])
    df.dropna(subset=['age'],inplace = True)

    #export
    pd.DataFrame(df_invalid_age, columns=list(df_invalid_age.columns)).to_csv('invalid_age_user_id.csv', index=False, encoding="utf-8") 
    print('file saved')
    return df
    
    
def plotAge(df):
    df.id.groupby(df.age).count().plot(kind='bar', alpha=0.4, color='b',figsize=(20,10))
    plt.ylabel('Number of  users')
    plt.show()
    
    
def plotGender(df):
    df.id.groupby(df.gender).count().plot(kind='bar', alpha=0.4, color='b',figsize=(20,10))
    plt.ylabel('Number of  users')
    plt.show()
    
def cleanGender(df):
    #Assign unknown to category '-unknown-' and '-other-'
    df.loc[df['gender']=='-unknown-', 'gender'] = 'UNKNOWN'
    df.loc[df['gender']=='OTHER', 'gender'] = 'UNKNOWN'
    return df

def cleanFirst_affiliate_tracked(df):
    df.loc[df['first_affiliate_tracked'].isnull(), 'first_affiliate_tracked'] = 'untracked'
    
    
    
    