# Airbnb Challenge : Where will new guests book their first travel experience?

This project is inspired by a Kaggle challenge proposed in February 2016 via their [website] (https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data).
It aims at determining where new guests will book their first travel experience.
By accurately predicting this information, Airbnb can share more personalized content with their community, decrease the average time to first booking, and better forecast demand.


## Description

We are given a list of users along with their demographics, web session records, and some summary statistics. We will try to predict which country a new user's first booking destination will be. All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.

The training and test sets are split by dates. In the test set, we will predict all the new users with first activities after 12/05/15. In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010.

The evaluation metric to determine the accuracy of the model is  [NDCG (Normalized discounted cumulative gain)](https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain). The destination country predictions will be ordered such that the most probable destination country goes first.


## Data
The data are real and from Airbnb. For a complete description, see the file `Data.md`.

# Scoring
We will implement a model that will be able to obtain the maximum score at the  [NDCG (Normalized discounted cumulative gain)](https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain). The NDCG is given by the following formula:

![equation](http://latex.codecogs.com/gif.latex?DCG_k%3D%5Csum_%7Bi%3D1%7D%5Ek%5Cfrac%7B2%5E%7Brel_i%7D-1%7D%7B%5Clog_2%7B%5Cleft%28i+1%5Cright%29%7D%7D)

![equation](http://latex.codecogs.com/gif.latex?nDCG_k%3D%5Cfrac%7BDCG_k%7D%7BIDCG_k%7D)

where *k* = 5, *rel_i* is the relevance of the result at position  *i*, *IDCG_k* is the maximum possible (ideal) *DCG* for a given set of queries. All *NDCG* calculations are relative values on the interval 0.0 to 1.0.


## Results

Using Cross-Validationt of ind the optimal parameters, 4 models were tried:
- Random Forest
  - Depth 16, Trees 600
  - NDCG : 0.86686
  
- XGBOOST
  - Depth 7, gamma 0.7, estimators 75, learning rate 0.1, multisoftprob
  - Best model : NDCG 0.86987, ranked 699, top 45%
  
- 2 layers stacking
  - Bad results because the first layer model has a F score of only 60%.
[![Screen Shot 2017-02-06 at 11.17.19.png](https://s27.postimg.org/j1iec25mr/Screen_Shot_2017_02_06_at_11_17_19.png)](https://postimg.org/image/wv6r13y7z/)
  
- Voting
  - Combine 3 models above
  - Soft voting
  - Bad results



** Best model XGBOOST : NDCG 0.86987, ranked 699, top 45% **
