#!/usr/bin/env python

# We'll use the pandas library to read CSV files into dataframes
import math
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn.cross_validation import train_test_split
import collections
import operator
import xgboost as xgb
import json

def save_user_purchase_list(user_cps):
    with open('../data/user_cps.json', 'w') as f:
        json.dump(user_cps, f)

def load_user_purchase_list():
    with open('../data/user_cps.json') as f:
        user_cps = json.load(f)
    return user_cps

def process_user_purchase_list():
    transcations = pd.read_csv('../data/coupon_detail_train.csv')

    print "transcations.shape = ", transcations.shape
    print "max date = ", np.max(transcations['I_DATE'])
    print "min date = ", np.min(transcations['I_DATE'])
    
    print 'Processing user purchase list'
    user_cps = {}
    for index, row in transcations.iterrows():
        
        if index % 10000 == 0: print index

        user_hash = row['USER_ID_hash']
        coupon_hash = row['COUPON_ID_hash']
        if user_hash not in user_cps:
            user_cps[user_hash] = [coupon_hash]
        else:
            user_cps[user_hash].append(coupon_hash)
    return user_cps

### clean the dataset
### 1. drop date related fields
### 2. drop coupon_id_hash
### 3. label encoding
def clean_dataset(d):
    c = d.copy()
    
    # drop date related fields
    c.drop('DISPFROM', axis=1, inplace=True)
    c.drop('DISPEND', axis=1, inplace=True)
    c.drop('VALIDFROM', axis=1, inplace=True)
    c.drop('VALIDEND', axis=1, inplace=True)

    # drop coupon hash
    c.drop('COUPON_ID_hash', axis=1, inplace=True)

    # encoding category/area range

    # impute missing values
    # mean value
    c['VALIDPERIOD'].fillna(c['VALIDPERIOD'].mean(), inplace=True)
    # usable on all days
    c['USABLE_DATE_MON'].fillna(1, inplace=True)
    c['USABLE_DATE_TUE'].fillna(1, inplace=True)
    c['USABLE_DATE_WED'].fillna(1, inplace=True)
    c['USABLE_DATE_THU'].fillna(1, inplace=True)
    c['USABLE_DATE_FRI'].fillna(1, inplace=True)
    c['USABLE_DATE_SAT'].fillna(1, inplace=True)
    c['USABLE_DATE_SUN'].fillna(1, inplace=True)
    c['USABLE_DATE_HOLIDAY'].fillna(1, inplace=True)
    c['USABLE_DATE_BEFORE_HOLIDAY'].fillna(1, inplace=True)

    a = np.array(c)

    #TODO(zxi) using better encoder for categorical fields
    for i in range(a.shape[1]):
        if type(a[1,i]) is str:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(a[:,i])
            a[:,i] = lbl.transform(a[:,i])

    # index = c.columns.get_loc("large_area_name")

    # print c['large_area_name'].shape
    # print a[:, index].shape

    scaler = preprocessing.StandardScaler()
    a = scaler.fit_transform(a)
    # a = preprocessing.normalize(a)

    return a

users = pd.read_csv("../data/user_list.csv")
print 'users.shape = ', users.shape

# find withdrawn users, who will not able to perchase after 
withdrawn_users = np.array(users['WITHDRAW_DATE'].notnull())
print 'withdrawn_users.count = ', np.sum(withdrawn_users)

train_coupon = pd.read_csv('../data/coupon_list_train.csv')
test_coupon = pd.read_csv('../data/coupon_list_test.csv')

print "train_coupon shape = ", train_coupon.shape
print "test_coupon shape = ", test_coupon.shape

all_coupon = pd.concat([train_coupon, test_coupon], ignore_index=True)
print "all_coupon shape = ", all_coupon.shape

cleaned_coupon = clean_dataset(all_coupon)
print cleaned_coupon[:, 0:9]
print 'cleaned_coupon shape = ', cleaned_coupon.shape

train = cleaned_coupon[0:train_coupon.shape[0], :]
test = cleaned_coupon[train_coupon.shape[0]:cleaned_coupon.shape[0], :]
print 'cleaned_train.shape = ', train.shape
print 'cleaned_test.shape = ', test.shape

D = pairwise_distances(train, test, metric='cosine')
print "D.shape = ", D.shape
print "D.max, D.min = ", np.max(D), np.min(D)

train_hash_to_index = dict(zip(train_coupon['COUPON_ID_hash'], train_coupon.index))
test_hash_to_index = dict(zip(test_coupon['COUPON_ID_hash'], test_coupon.index))


#print cleaned_coupon

# user_cps = process_user_purchase_list(transcations)
# save_user_purchase_list(user_cps)

user_cps = load_user_purchase_list()
print "Total purchased users = ", len(user_cps.keys())

recommands = []
user_index = -1
skipped = 0

THRASHOLD = 0.3

for user_hash, withdrawn_date in zip(users['USER_ID_hash'], users['WITHDRAW_DATE']):
    user_index += 1
    recommand = []
    
    #TODO(implement recommandation)
    # print withdrawn_date, type(withdrawn_date)
    if type(withdrawn_date) is not str and user_hash in user_cps:
        # normal users
        # ...
        # ...
        # pcs =         
        train_indics = [train_hash_to_index[pc_hash] for pc_hash in user_cps[user_hash]]
        
        count = len(train_indics)
        # print 'purchased count = ', count
        min_dist = np.min(D[train_indics,:], axis=0)
        selected_index = min_dist < THRASHOLD
        selected_cp = test_coupon[selected_index]['COUPON_ID_hash']
        # print 'selected_cp.shape = ', selected_cp.shape
        selected_dist = min_dist[selected_index]

        sorted_selected_index = np.argsort(selected_dist)
        # print 'sorted_selected_index.shape = ', sorted_selected_index.shape, 'min = ', np.min(sorted_selected_index), 'max = ', np.max(sorted_selected_index)
        # print 'sorted_selected_index', sorted_selected_index

        sorted_selected_dist = selected_dist[sorted_selected_index]
        sorted_selected_cp = selected_cp[sorted_selected_index]
        
        recommand = list(selected_cp)
        print 'purcahse count = ', count, 'recomanded count = ', len(recommand)

    else:        
        skipped += 1

    recommands.append(" ".join(recommand))

print 'skipped = ', skipped

preds = pd.DataFrame({"USER_ID_hash": users['USER_ID_hash'], "PURCHASED_COUPONS": recommands})
preds = preds.set_index('USER_ID_hash')
preds.to_csv('prediction.csv')
