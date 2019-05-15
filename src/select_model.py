# -*_ coding: utf-8 -*-
"""
Created on Apr 11 10:50:12 2019
@author: Youngmi Huang
@email: cyeninesky3@gmail.com

模型选择与调参
"""

import csv
import time
import numpy as np 
import pandas as pd 
import pickle
import datetime
import lightgbm as lgb
from datetime import date
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tools.utils import *
from tools.prepro import *


def lgb_grid_search(X_train, y_train, X_valid, y_valid):
    depth_paras = np.arange(6,15,2)
    auc_scores = []
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
    for depth in depth_paras:
        params = {'learning_rate': 0.01, 
            'max_depth': depth, 
            'boosting': 'gbdt', 
            'objective': 'binary', 
            'metric': 'auc', 
            'is_training_metric': True, 
            'seed': 42}
        print('Now is depth: {}'.format(depth))
        model = lgb.train(params, train_set= lgb_train, num_boost_round=2000,
                valid_sets=[lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_valid, label=y_valid)], verbose_eval=100, early_stopping_rounds=100)
        y_valid_pred = model.predict(X_valid)
        auc_score = roc_auc_score(y_valid, y_valid_pred)
        auc_scores.append(auc_score)
    # 寫入
    with open('../csv/model/model_params.txt' , 'w', newline='') as f:
        wr = csv.writer(f, delimiter=' ')
        wr.writerow(['lgb', depth_paras, auc_scores])
    
    # 讀取
    # with open('../csv/model/model_paras.txt' , newline='') as f:
    #     rows = csv.reader(f, delimiter=' ')
    #     for row in rows:
    #         print(row)

    idx_max = np.argmax(auc_scores)
    best_depth = int(depth_paras[idx_max])
    print(int(depth_paras[idx_max]))
    return best_depth


def lgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # best_depth = lgb_grid_search(X_train, y_train, X_valid, y_valid)
    params = {'learning_rate': 0.01, 
            'max_depth': 12, 
            'boosting': 'gbdt', 
            'objective': 'binary', 
            'metric': 'auc', 
            'is_training_metric': True, 
            'seed': 42}
    
    model = lgb.train(params, train_set=lgb.Dataset(X_train, label=y_train), num_boost_round=2000,
                valid_sets=[lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_valid, label=y_valid)],
                verbose_eval=100, early_stopping_rounds=100)
    
    # plot_feature_importance(model)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    return y_valid_pred, y_test_pred

def xgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    params = {'learning_rate': 0.1, 
            'n_estimators': 2000, 
            'max_depth': 10, 
            'min_child_weight': 1, 
            'seed': 0,
            'subsample': 0.8, 
            'colsample_bytree': 0.8, 
            'gamma': 0, 
            'reg_alpha': 0, 
            'reg_lambda': 1}

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_train, y_train), (X_valid, y_valid)], 
            verbose=100, early_stopping_rounds=1000)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
    return y_valid_pred, y_test_pred

def logreg_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_valid_pred = model.predict_proba(X_valid)
    y_test_pred = model.predict_proba(X_test)
    return y_valid_pred[:, 1], y_test_pred[:, 1]

def rf_model(X_train, y_train, X_valid, y_valid, X_test, y_test):
    params = {'n_estimators': 1200,
            'max_depth': 10}
    
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    y_valid_pred = rf.predict_proba(X_valid)
    y_test_pred = rf.predict_proba(X_test)
    return y_valid_pred[:, 1], y_test_pred[:, 1]


