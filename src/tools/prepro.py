# -*_ coding: utf-8 -*-
'''
Editor: Youngmi huang
Update: 2019/02/26
'''

import numpy as np 
import pandas as pd 
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import date

def preprocessing(df):
    inds = create_adj_roe(df)
    df = pd.merge(df, inds, how='left', on=['code', 'date', 'industry', 'roe_q'])
    
    # standardization
    df = df.drop(['close', 'industry', 'roe_q'], axis=1)
    cols = [i for i in df.columns if i not in ['code', 'date', 'returns', 'returns_next1m_close', 'label', 'returns_next1m_open', 'revised_label', 'excess_returns_daily']]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    
    # get beta
    beta = get_beta(df)
    df = pd.merge(df, beta, how='left', on=['code', 'date'])
    return df 

def create_adj_roe(df):
    cols = ['industry', 'code', 'date', 'roe_q']
    df = df[cols]
    inds_df = pd.DataFrame(columns=cols)
    inds = list(set(df.industry))
    for ind in inds:
        single = df[df['industry'] == ind]
        tmp = single.set_index(['industry', 'code','date'])
        tmp = tmp.unstack(level=2)
        mtx = tmp.values
        mtx_mean = np.nanmean(mtx, axis=0)
        mtx_std = is_pos(np.nanstd(mtx, axis=0)) # 如果是0要等於1

        mtx_trans = ((mtx - mtx_mean)/mtx_std).reshape(-1)
        mtx_trans = mtx_trans[~np.isnan(mtx_trans)]
        single['roe_adj'] = mtx_trans
        inds_df = pd.concat([inds_df, single], axis=0, sort=False)
    return inds_df

def is_pos(arr):
    r=[]
    for i in arr:
        if i == 0:
            r.append(1)
        else:
            r.append(i)
    return r

def load_hs300_average():
    df = pd.read_csv('../csv/model/hs300_average.csv')
    return df

def get_beta(df):
    n=22
    return_cols = ['code', 'date', 'beta']
    hs300 = load_hs300_average()

    df = df.copy()
    df = df[['code', 'date', 'returns']]
    df = pd.merge(df, hs300, how='left', on='date')
    df['xy'] = df['returns'] * df['returns_mkt']
    roll = df.groupby('code').rolling(window=n)
    df['cov_xy'] = ((roll.xy.mean() - roll.returns.mean() * roll.returns_mkt.mean())*n/(n-1)).reset_index(drop=True)
    df['var_y'] = roll.returns_mkt.var().reset_index(drop=True)
    df['beta'] = (df['cov_xy']/df['var_y'])
    df = df[return_cols].fillna(0)
    return df

def select_by_beta(df, topN=20):
    mask1 = df['y_test_pred']>0.5
    mask2 = df['beta']>0
    df = df[mask1 & mask2]
    select = df.sort_values(by=['beta'])[['code', 'beta', 'y_test_pred']]
    
    total = len(df)
    group = 5
    sub_num = round(total/group)
    strategies ={}

    if sub_num < topN:
        num_every_group = round(total/(group+1))
        print('num_every_group:{} '.format(num_every_group))
        diff = topN -2*num_every_group
        for i in range(group-1):
            add = list(select['code'][i*sub_num: (i+2)*sub_num+diff])
            strategies[i] = add
        last = select['code'][-topN:].tolist()
        strategies[group-1] = last

    elif sub_num > topN:
        num_every_group = round(total/(group))
        for i in range(group):
            add = select[i*num_every_group :(i+1)*num_every_group]
            add = list(add.sort_values(by=['y_test_pred'], ascending=False)['code'][:topN])
            strategies[i] = add        
    else:
        for i in range(group):
            add = list(select['code'][i*topN :(i+1)*topN])
            strategies[i] = add             
    return strategies
