# Functions to preprocess Train_user and Test_user
# Pecoraro Cyril

import pandas as pd
import numpy as np
import os
from IPython.display import Image
from IPython.core.display import HTML 
import matplotlib.pyplot as plt  
import random
from datetime import datetime
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import copy
import pylab


# clean age
# @arg(in) df : DataFrame
# @arg(in) type : 'k' : unvalid data to be replaced by -1. 'd' : unvalid data to be deleted
def cleanAge(df, type):
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
    
    if(type == 'k'):
        # Counting invalid ages:
        OutOfBoundsAgePercentage = round(100*len(df.loc[(df['age'] == -1), 'age'])/len(df),2)
        print('Percentage of users with irrelevant age',OutOfBoundsAgePercentage,'%')

        # Counting NaN ages:
        nanAgePercentage = round(100*df['age'].isnull().values.sum()/len(df),2)
        print('Percentage of users with NaN age',nanAgePercentage,'%')

        # Assigning a -1 value to NaN ages
        df = copy.deepcopy(df)
        df['age'].fillna(-1, inplace = True) 
        print('All the invalid or missing age were replaced by value -1')
        
    if(type == 'd'):
        df = df[df['age'] != -1]
        
    return df

# clean ageBucket
# @arg(in) df : DataFrame
def cleanAgeBucket(df):
    df.drop(df.ix[df['age_bucket'].isin(['0-4','5-9','10-14','90-94','95-99','100+'])].index, inplace= True)
    return df

# preprocess display of travelers per country given age and sex
# @arg(in) df : DataFrame
def travellerCountryProcess(df):
    # remove  year
    if 'year' in df:
        df.drop(['year'],axis=1,inplace = True)

    # Compute number of people by age (previously, people were characterized by age AND sex)
    df_destination_age = df.groupby(['country_destination','age_bucket','gender'],as_index=False).sum()

    # Compute total number of people by country 
    df_destination_total = df_destination_age.groupby('country_destination').sum().reset_index()

    # Incorpore total in the df_destination_age dataframe
    df_destination_age = df_destination_age.merge(df_destination_total, on='country_destination')
    df_destination_age=df_destination_age.rename(columns = {'population_in_thousands_y':'population_total_in_thousands',
                                                            'population_in_thousands_x':'population_in_thousands'})

    # Compute share of people by age and destination 
    df_destination_age['proportion_%']=np.round(df_destination_age['population_in_thousands']/
                                                 df_destination_age['population_total_in_thousands']*100, 
                                                 decimals=1)

    # Index dataframe by country of destination then age
    df_destination_age_male = df_destination_age.loc[df_destination_age['gender'] == 'male']
    df_destination_age_female = df_destination_age.loc[df_destination_age['gender'] == 'female']
    
    return df_destination_age_male,df_destination_age_female,df_destination_total
    
# Display of travelers proportion per country given age and sex
# @arg(in) df_destination_age_male : DataFrame  of males  
# @arg(in) df_destination_age_female : DataFrame  of females  
def travellerProportionCountryPlot(df_destination_age_male,df_destination_age_female):
    #male in blue
    fig, axes = plt.subplots(nrows=3, ncols=2)
    for (i, group), ax in zip(df_destination_age_male.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="proportion_%", title=str(i),ax=ax ,kind='line',color='b',label='male' )
        ax.set_ylim([0, 6])
        ax.set_ylabel('percentage')
    plt.tight_layout()

    #female in red
    for (i, group), ax in zip(df_destination_age_female.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="proportion_%", title=str(i),ax=ax ,kind='line',color='r',label='female')
        ax.set_ylim([0, 6])
    plt.tight_layout()
    plt.show()
    
# Display number of travelers number per country given age and sex
# @arg(in) df_destination_age_male : DataFrame  of males  
# @arg(in) df_destination_age_female : DataFrame  of females  
def travellerNumberCountryPlot(df_destination_age_male,df_destination_age_female):
    #male in blue
    fig, axes = plt.subplots(nrows=3, ncols=2)
    for (i, group), ax in zip(df_destination_age_male.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="population_in_thousands", title=str(i),ax=ax ,kind='line',color='b',label='male' )
        ax.set_ylim([100, 4000])
        ax.set_ylabel('people in thousands')
    plt.tight_layout()


    #female in red
    for (i, group), ax in zip(df_destination_age_female.groupby("country_destination"), axes.flat):
        group.plot(x='age_bucket', y="population_in_thousands", title=str(i),ax=ax ,kind='line',color='r',label='female') 
        ax.set_ylim([100, 4000])
        ax.set_ylabel('people in thousands')
    plt.tight_layout()
    plt.title('salut')
    plt.show()
    
# Display number of travelers number per country
# @arg(in) df_destination_total : DataFrame of all travellers 
def destinationTotalPlot(df_destination_total):
    ax = df_destination_total.sort_values('population_in_thousands',ascending=False).plot(x='country_destination', y='population_in_thousands',kind='bar')
    ax.set_ylabel('people in thousands')
    ax.set_title('Destination of travelers')
    plt.show()

    
# export a list of id with the invalid ages to .csv file
# @arg(in) df : DataFrame
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
    
# plot age
# @arg(in) df : DataFrame    
def plotAge(df):
    df.id.groupby(df.age).count().plot(kind='bar', alpha=0.4, color='b',figsize=(20,10),logy=True)
    plt.ylabel('Number of  users, log scale')
    plt.show()
    
# plot gender
# @arg(in) df : DataFrame    
def plotGender(df):
    df.id.groupby(df.gender).count().plot(kind='bar', alpha=0.4, color='b',figsize=(20,10))
    plt.ylabel('Number of  users')
    plt.show()
    
# clean gender
# @arg(in) df : DataFrame
def cleanGender(df):
    #Assign unknown to category '-unknown-' and '-other-'
    df.loc[df['gender']=='-unknown-', 'gender'] = 'UNKNOWN'
    df.loc[df['gender']=='OTHER', 'gender'] = 'UNKNOWN'
    return df

# clean First_affiliate_tracked
# @arg(in) df : DataFrame
def cleanFirst_affiliate_tracked(df):
    df.loc[df['first_affiliate_tracked'].isnull(), 'first_affiliate_tracked'] = 'untracked'
    return df

# Clean Date_first_booking
def cleanDate_First_booking(df):
    df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])
    return df

def plotDate_First_booking_months(df):
    df.id.groupby([df.date_first_booking.dt.month]).count().plot(kind="bar")
    plt.xlabel('Month')
    plt.ylabel('Number of bookings')
    plt.title('Number of bookings over the months of the year')
    plt.show()
    
def plotDate_First_booking_years(df):
    df.date_first_booking.value_counts().plot(kind='line', linewidth=1,figsize=(15,10))
    plt.ylabel('Number of bookings')
    plt.title('Number of bookings throughout time')
    plt.show()
    
def plotDate_First_booking_months(df):
    df.id.groupby([df.date_first_booking.dt.month]).count().plot(kind="bar")
    plt.xlabel('Month')
    plt.ylabel('Number of bookings')
    plt.title('Number of bookings over the months of the year')
    plt.show()
    
def plotsth(df):
    df.id.groupby([df.date_first_booking.dt.month]).count().plot(kind="bar")
    plt.xlabel('Month')
    plt.ylabel('Number of bookings')
    plt.title('Number of bookings over the months of the year')
    plt.show()
    
def computeDate_First_booking_weekdays(df):
    weekday = []
    for date in df.date_first_booking:
        weekday.append(date.weekday())
    return pd.Series(weekday)

def plotDate_First_booking_weekdays(df):
    sns.barplot(x = df.value_counts().index, y=df.value_counts().values)
    plt.xlabel('Week Day')
    plt.title(s='Number of bookings per day in the week')
    
#export file to csv
# @arg(in) filename : name of file in String, with .csv at the end 
# @arg(in) df : DataFrame
def saveFile(df, filename):
    pd.DataFrame(df, columns=list(df.columns)).to_csv(filename, index=False, encoding="utf-8") 
    print('file saved')    
    