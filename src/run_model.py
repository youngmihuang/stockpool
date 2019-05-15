# -*_ coding: utf-8 -*-
"""
Created on Apr 28 00:13:10 2019
@author: Youngmi Huang
@email: cyeninesky3@gmail.com

产生选股模型的每月上涨概率(performance_hist)、更新后的股票池(codelist_info)
"""
__all__     = ['main_run_model']
__version__ = '0.1.0'


from tools.utils import *
from tools.prepro import *
from create_feature import create_label
import os
import yaml
import time
import numpy as np 
import pandas as pd 
import pickle
import datetime
import lightgbm as lgb
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from create_feature import get_total_features, create_industry
from change_data import batch_update, get_codelist_info
from select_model import lgb_grid_search, lgb_model

path = './config.yml'
with open(path, encoding='utf-8') as f:
    config = yaml.load(f)


file_mkt            = config['raw_update']['daily_mkt']
file_mkt_compose    = config['base']['mkt_compose']
dm_basic_5          = config['dm_generate']['generate_dm_basic_5']
dm_total_feats      = config['dm_generate']['generate_total_feats']
model_path          = config['path']['model_dir']
select_strategy     = config['select_stocks']['strategy_name']
blacklist           = config['select_stocks']['blacklist']
performance         = config['select_stocks']['performance']
code_list_info      = config['select_stocks']['result_for_RL']

basic_5_path        = os.path.join(model_path, dm_basic_5)
feature_path        = os.path.join(model_path, dm_total_feats)
hs300_compose_path  = os.path.join(model_path, file_mkt_compose[0])
sh50_compose_path   = os.path.join(model_path, file_mkt_compose[1])
hs300_average_path  = os.path.join(model_path, file_mkt[1])
sh50_path           = os.path.join(model_path, file_mkt[2])
blacklist_path      = os.path.join(model_path, blacklist)
performance_path    = os.path.join(model_path, performance)


def load_features():
    with open(feature_path, 'rb') as f:
        df = pickle.load(f)
    return df

def load_hs300_average():
    df = pd.read_csv(hs300_average_path)
    return df 

def load_sh50():
    df = pd.read_csv(sh50_path)
    return df

def _pool_comp(filepath, dt):
    df = pd.read_csv(filepath)
    comp_dt = max(df[ df['date']<=dt].date)
    pool = df[ df['date'] == comp_dt]
    pool_list = [i[:-3] for i in pool.code]
    return pool_list

def load_pool_comp(dt):
    pool_hs300 = _pool_comp(hs300_compose_path, dt) 
    pool_sh50 = _pool_comp(sh50_compose_path, dt) 
    pool_list = sorted(list(set(pool_hs300+pool_sh50)))
    return pool_list, pool_hs300, pool_sh50

def get_input(df, indices):
    num_cols = [i for i in df.columns if i not in ['code', 'date', 'returns_next1m_close', 'label', 'returns_next1m_open', 'returns_mkt', 'close_mkt', 'excess_returns_daily', 'revised_label']]
    X = df.loc[indices, num_cols]
    # 预测上涨
    y = (df.loc[indices, 'returns_next1m_open'] >0).astype(int)
    # 预测下跌
    # y = (df.loc[indices, 'returns_next1m_open'] <0).astype(int)
    return X, y

def _select_exclude_list():
    with open(blacklist_path, 'r') as f:
        exclude_list = list(set([line.split(',')[0].strip() for index, line in enumerate(f) if index > 0]))
    return exclude_list
    
def select_by_prob_hs300(df, select_dt, topN=20):
    _ , pool_hs300, pool_sh50 = load_pool_comp(select_dt)
    exclude_list = _select_exclude_list()
    pool_hs300 = [i for i in pool_hs300 if i not in exclude_list]

    df = df[df['code'].isin(pool_hs300)]
    df = df.sort_values(by=['y_test_pred'], ascending=False).reset_index(drop=True)
    select_code = df['code'][:topN].tolist()
    select_dict = {}
    for i in range(len(df)):
        select_dict[df['code'][i]] = df['y_test_pred'][i]
    return select_code, select_dict

def select_by_prob_sh50(df, select_dt, topN=20):
    _ , pool_hs300, pool_sh50 = load_pool_comp(select_dt)
    exclude_list = _select_exclude_list()
    pool_sh50 = [i for i in pool_sh50 if i not in exclude_list]

    df = df[df['code'].isin(pool_sh50)]
    df = df.sort_values(by=['y_test_pred'], ascending=False).reset_index(drop=True)
    select_code = df['code'][:topN].tolist()
    select_dict = {}
    for i in range(len(df)):
        select_dict[df['code'][i]] = df['y_test_pred'][i]
    return select_code, select_dict

def select_strategy_codelist(df, select_dt, N):
    alpha_hs300, prob_hs300 = select_by_prob_hs300(df, select_dt, N)
    alpha_sh50, prob_sh50 = select_by_prob_sh50(df, select_dt, N)

    strategy = {}
    if len(alpha_hs300) > 0:
        strategy[select_strategy[0]] = alpha_hs300, prob_hs300
    if len(alpha_sh50) > 0:
        strategy[select_strategy[1]] = alpha_sh50, prob_sh50
    return strategy

def get_win_lose_ratio(df):
    codeList = list(set(df.code))
    mkt, raw = [], []
    for code in codeList:
        r = np.array(df[df['code'] == code]['returns'])
        r_mkt = np.array(df[df['code'] == code]['returns_mkt'])
        raw.append(holding_period_returns(r))
        mkt.append(holding_period_returns(r_mkt))
    result = np.array(raw) - np.array(mkt) 
    win_lose_ratio = sum(win_lose)/len(win_lose)
    return win_lose_ratio

def get_select_performance(test, feats, select_dt, start_dt, end_dt, N):
    strategy = select_strategy_codelist(test, select_dt, N)
    names = [i for i in strategy.keys()]
    cum_returns, excess_returns, max_drawdowns, info_ratios, calmr_ratios, sharpe_ratios, win_lose_ratios = [], [], [], [], [], [], []

    print('Next month: %s %s' %(start_dt, end_dt))
    for name in names:
        if name == select_strategy[0]:
            mkt = load_hs300_average()
        else:
            mkt = load_sh50()

        mask1 = feats['date'].between(start_dt, end_dt)
        mask2 = feats['code'].isin(strategy[name][0])
        select = feats[(mask1 & mask2)]

        if len(select) == 0:
            win_lose_ratios.append(0)
            cum_returns.append(0)
            excess_returns.append(0)
            max_drawdowns.append(0)
            info_ratios.append(0)
            calmr_ratios.append(0)
            sharpe_ratios.append(0)

        else:
            # 个股表现的胜率
            select = pd.merge(select, mkt, how='left', on='date').dropna()
            win_lose_ratios.append(get_win_lose_ratio(select))
            
            # 以下为股票池表现的计算
            select = select.groupby(['date'])['returns'].mean().reset_index()
            select = pd.merge(select, mkt, how='left', on='date').dropna()
            cum_returns.append(holding_period_returns(select['returns']))
            excess_returns.append(get_excess_returns(select['returns'], select['returns_mkt']))
            max_drawdowns.append(get_max_drawdown(select['returns']))
            info_ratios.append(get_info_ratio(select['returns'], select['returns_mkt']))
            calmr_ratios.append(get_calmr_ratio(select['returns']))
            sharpe_ratios.append(get_sharpe_ratio(select['returns'], select['returns_mkt']))
            # plot_cumulative_return(select['date'], select['returns'], select['returns_mkt'], name)

    df = pd.DataFrame()
    df['select_code'] = [i[0] for i in strategy.values()]
    df['prob'] = [i[1] for i in strategy.values()]
    df['select_dt'] = select_dt
    df['start_dt'] = start_dt
    df['end_dt'] = end_dt
    df['win_lose_ratio'] = win_lose_ratios
    df['cum_returns_1m'] = cum_returns
    df['excess_returns'] = excess_returns
    df['max_drawdown'] = max_drawdowns
    df['info_ratio'] = info_ratios
    df['calmr_ratio'] = calmr_ratios
    df['sharpe_ratio'] = sharpe_ratios
    df['remarks'] = names 
    return df

def save_performance(df):
    cols = ['select_dt', 'start_dt', 'end_dt', 'cum_returns_1m', 'excess_returns','max_drawdown', 'info_ratio', 
            'calmr_ratio', 'sharpe_ratio', 'win_lose_ratio', 'precision', 'auc', 'select_code', 'prob','remarks']
    df = df[cols]
    df.to_csv(performance_path, index=False)


def main_run_model(start_m, end_m, N=20):
    mkt = load_hs300_average()
    start_end_dates = get_start_end_dt(mkt, start_m, end_m)
    select_dates = sorted(list(start_end_dates.keys()))
    num_select_dates = len(select_dates)
    
    feats = load_features()
    feats = feats[feats['date']>= '2014-03-31']  # original

    time_starts = time.time()
    feats_all = preprocessing(feats)
    print('feats processing successful, spend: {}'.format(time.time()-time_starts))

    time_model_starts = time.time()
    total = pd.DataFrame()
    for select_dt in select_dates: 
        # 股票池組成
        start_dt = start_end_dates[select_dt][0]
        end_dt = start_end_dates[select_dt][1]
        
        pool_list, _ , _ = load_pool_comp(select_dt)
        pool_feats_all = feats_all[feats_all['code'].isin(pool_list)]

        # 選股時間
        test = feats_all[feats_all['date'] == select_dt].reset_index(drop=True)
        if len(test) == 0:
            print('no trading dt')
            trading_dt = list(set(feats_all[feats_all['date'] < select_dt].date))
            select_dt = max(trading_dt)
            test = feats_all[feats_all['date'] == select_dt].reset_index(drop=True)
            train = feats_all[feats_all['date'] < select_dt]

        else:
            train = feats_all[feats_all['date'] < select_dt]

        print('Now is: %s, start: %s, end: %s' %(select_dt, start_dt, end_dt))
        train = train[train['label'].isin([1,-1])].reset_index(drop=True)
        print(len(train))
        train_indices, val_indices = train_test_split(train.index.values, test_size=0.1, random_state=6)
        X_train, y_train = get_input(train, train_indices)
        X_valid, y_valid = get_input(train, val_indices)
        X_test, y_test = get_input(test, test.index.values)
        
        y_valid_pred, y_test_pred = lgb_model(X_train, y_train, X_valid, y_valid, X_test, y_test)
        auc_score = auc_scores(y_valid, y_valid_pred)        
        precision = precision_scores(y_valid, y_valid_pred)

        # 检验选股模型选出的股在下个月的表现
        test['y_test_pred'] = y_test_pred
        df = get_select_performance(test, feats, select_dt, start_dt, end_dt, N)    
        df['precision'] = precision
        df['auc'] = auc_score
        total = pd.concat([total, df], axis=0, sort=False)

    save_performance(total)
    print('model training and predicting Done, spending: {}'.format(time.time()- time_model_starts))
    
    if not os.path.exists(code_list_info):
        batch_update()
    else:
        codelist_info = get_codelist_info()
    print('codelist_info Done')
    return codelist_info

if __name__ == '__main__':
    main_run_model(201712, 201811, 20)
    print('All Done')
