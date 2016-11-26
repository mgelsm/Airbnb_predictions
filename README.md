# Airbnb Challenge : Where will a new guest book their first travel experience?

# Abstract

This project will be developped within the framework of the course An Introduction to Data Science.
It is inspired by a Kaggle challenge proposed in February 2016 via their [website] (https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data).
It aims at determining where a new guest will book their first travel experience.
By accurately predicting this information, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.

# Description

We are given a list of users along with their demographics, web session records, and some summary statistics. We will try to predict which country a new user's first booking destination will be. All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.

The training and test sets are split by dates. In the test set, we will predict all the new users with first activities after 12/05/15. In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010.

The evaluation metric to determine the accuracy of the model is NDCG (Normalized discounted cumulative gain). The destination country predictions will be ordered such that the most probable destination country goes first.

In the ideal case, an interactive visualization map of each destination country with a user type will be presented.   

# Feasibility 

The features used are all non intrusive and completely anonymized. 
A possible problem would be that the features used are not relevant to predict the first destination of a new AirBnb customer. 

# Deliverables

A Jupyter Notebook and the data used for the analysis will be handout. 

# Timeline

This project will be presented in January 2016.
