# -*_ coding: utf-8 -*-
"""
Created on Apr 28 00:13:10 2019
@author: Shao Xuan Huang
@email : HUANGSHOAXUAN801@pingan.com.cn

计算并输出 RL 训练所需特徵
"""
__all__     = ['deliver_data']
__version__ = '0.1.0'

import os
import yaml
import pickle
import time
import csv
import pandas as pd 
import numpy as np 
import tushare as ts
import datetime

path = './config.yml'
with open(path, encoding='utf-8') as f:
    config = yaml.load(f)

all_mkt_code        = config['base']['all_mkt_code']
all_industry        = config['base']['industry']

basic_stock_path    = config['path']['daily_stock_dir']
new_stock_path      = config['path']['daily_new_dir']
index_data_path     = config['path']['daily_index_dir']
deliver_data_path   = config['path']['daily_deliver_dir']
model_path          = config['path']['model_dir']

mood_path           = config['raw_update']['daily_ola']
amount_path         = os.path.join(index_data_path, config['dm_generate']['cal_amount'])
vol_5d_path         = os.path.join(index_data_path, config['dm_generate']['cal_5d_vol'])
vol_22d_path        = os.path.join(index_data_path, config['dm_generate']['cal_22d_vol'])
vol_change_path     = os.path.join(index_data_path, config['dm_generate']['cal_vol_change'])
error_path          = os.path.join(index_data_path, config['dm_generate']['error_log'])

hs300_compose_path  = os.path.join(model_path, config['base']['mkt_compose'][0])
sh50_compose_path   = os.path.join(model_path, config['base']['mkt_compose'][1])

def get_all_stock():
    with open(all_mkt_code, 'r') as f:
        codeList = f.read().split('\n')
    codeList = [i[1:-1]+'.csv' for i in codeList]
    return codeList

def get_list_info():
    """
    :return: 正常交易股票(包含停牌)以及上市日期, 以及退市股票
    """
    codeList = get_all_stock()
    pro = ts.pro_api()
    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    data['list_date'] = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in data.list_date]
    data = data.rename(columns={'symbol':'code'})
    existList = data.ts_code.tolist()
    stopList = [i for i in codeList if i[:-4] not in existList]
    return existList, stopList        

def make_amount_data():
    codeList = get_all_stock()
    origin = pd.read_csv(amount_path)
    origin_max_dt = max(origin.date)
    today_dt = datetime.date.today().strftime('%Y-%m-%d')

    if origin_max_dt == today_dt:
        print('amount data already up-to-date.')
        return 0
    
    print('Update make_amount_data Start...')
    error_code = []
    result = pd.DataFrame()    
    for code in codeList:
        try:
            df = pd.read_csv(os.path.join(basic_stock_path, code))[['date', 'amount']]
            df = df[df['date'] > origin_max_dt].set_index('date')
            df = df.rename(columns={'amount':code[:-4]})
            result = pd.concat([result, df], axis=1, sort=False)
        except:
            error_code.append(code)
    result = result.reset_index().rename(columns={'index':'date'})
    result = pd.concat([origin, result], axis=0, sort=False)
    result = result.drop_duplicates().sort_values(by=['date'])
    result.to_csv(amount_path, index=False)

    with open(error_path, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(error_code)            
    end = time.time()
    print('Update make_amount_data Done.')

def make_volatility_5_22_data():
    codeList = get_all_stock()
    origin_5 = pd.read_csv(vol_5d_path)
    origin_22 = pd.read_csv(vol_22d_path)
    origin_max_dt = max(origin_5.date)
    origin_max_dt_next = (datetime.datetime.strptime(origin_max_dt, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    today_dt = datetime.date.today().strftime('%Y-%m-%d')

    
    if origin_max_dt == today_dt:
        print('volatility 5 and 22 data already up-to-date.')
        return 0

    print('Update make_volatility_5_22 Start...')
    volatility_all_05= pd.DataFrame()
    volatility_all_22= pd.DataFrame()
    origin_max_dt_past = (datetime.datetime.strptime(origin_max_dt, '%Y-%m-%d') + datetime.timedelta(days=-40)).strftime('%Y-%m-%d')

    error_code = []
    for code in codeList:
        try:
            df = pd.read_csv(os.path.join(basic_stock_path, code))[['date', 'close']]
            df = df[df['date']> origin_max_dt_past].set_index('date')
            df = df.sort_index()
            df['close']  = pd.to_numeric(df['close'])
            df['log_ret']= np.log(df['close']/df['close'].shift(1))
            df = df.dropna()
            df['volatility'] = df['log_ret'].rolling(window=5,center=False).std()*np.sqrt(252)
            df['volatility_22'] = df['log_ret'].rolling(window=22,center=False).std()*np.sqrt(252)
            volatility_all_05[str(code[:-4])] = df['volatility']
            volatility_all_22[str(code[:-4])] = df['volatility_22']
        except:
            error_code.append(code)
            print(error_code)
    volatility_all_05 = volatility_all_05.reset_index()  
    volatility_all_05 = volatility_all_05[volatility_all_05['date']> origin_max_dt]
    volatility_all_05 = pd.concat([origin_5, volatility_all_05], axis=0, sort=False).reset_index(drop=True)
    volatility_all_05 = volatility_all_05.drop_duplicates().sort_values(by=['date'])

    volatility_all_22 = volatility_all_22.reset_index()        
    volatility_all_22 = volatility_all_22[volatility_all_22['date']> origin_max_dt]
    volatility_all_22 = pd.concat([origin_22, volatility_all_22], axis=0, sort=False).reset_index(drop=True)
    volatility_all_22 = volatility_all_22.drop_duplicates().sort_values(by=['date'])

    volatility_all_05.to_csv(vol_5d_path, index=False)
    volatility_all_22.to_csv(vol_22d_path, index=False)
    print('Update make_volaitility_5_22 Done.')

def make_volatility_change_data():
    codeList = get_all_stock()
    codeList = [i[:-4] for i in codeList]
    origin = pd.read_csv(vol_change_path)
    origin_max_dt = max(origin.date)
    today_dt = datetime.date.today().strftime('%Y-%m-%d')
    
    if origin_max_dt == today_dt:
        print('volatility change data already up-to-date.')
        return 0

    print('Update make_volatility_change Start...')
    volatility_all_05 = pd.read_csv(vol_5d_path).set_index('date')
    volatility_all_22 = pd.read_csv(vol_22d_path).set_index('date')
    volatility_change_all = pd.DataFrame()

    # 1 波動放大 0 波動縮小
    for code in codeList:
        try:
            volatility_change = []
            for five, ttwo in zip(np.array(volatility_all_05[code]), np.array(volatility_all_22[code])):
                if np.isnan(five) == True:  # 判斷nan值
                    volatility_change.append('')
                else:
                    if five > ttwo:
                        volatility_change.append(1)
                    else:
                        volatility_change.append(0)
            df = pd.DataFrame({'volatility_change': volatility_change}, index=volatility_all_05.index)
            volatility_change_all[str(code)] = df['volatility_change']
        except:
            pass
    volatility_change_all.to_csv(vol_change_path)
    print('Update make_volaitility_change Done.')

def get_codelist():
    pool_hs300 = pd.read_csv(hs300_compose_path).code.tolist()
    pool_sh50 = pd.read_csv(sh50_compose_path).code.tolist()
    pool_list = sorted(list(set(pool_hs300 + pool_sh50)))
    pool_list = [i+'.csv' for i in pool_list if i not in ['000024.SZ', '600005.SH']] # 排除例外(提前退市股)
    return pool_list

def load_amount():
    amount = pd.read_csv(amount_path)
    rename_dict = {}
    for col in amount.columns[1:]:
        rename_dict[col] = col[:-3]
    amount = amount.rename(columns=rename_dict)
    return amount

def load_volatility():
    df = pd.read_csv(vol_change_path).set_index('date')
    return df 

def load_industry():
    df = pd.read_csv(all_industry)
    return df 

def _code_format_trans(df):
    arr = np.array([str(i).zfill(6) for i in df.code])
    return arr 

def _code_ind_mapping():
    industry = load_industry()
    code2ind = {}
    for i in range(len(industry)):
        code2ind[industry['code'][i]] = industry['ind'][i]  
    return code2ind

def create_stk_fluidity():
    mkt_amount, stk_fluidity0, stk_fluidity1 = [], [], []
    amount = load_amount()
    for date in amount['date']:
        row_volume = np.array(amount[amount['date'] == date].iloc[:, 1:]) # volumes
        include_suspension_row_volume = row_volume[0]
        row_volume = row_volume[~np.isnan(row_volume)]                    # 取不包含 nan 的 volume
        mkt_amount.append(np.sum(row_volume))
        stk_fluidity0.append(round(float(np.sum(np.sort(row_volume)[-100:]) / np.sum(row_volume)), 4)) # 當日前100大股的交易量/整體交易量比重
        stk_fluidity1.append(len(row_volume[row_volume <= 1000]) / len(include_suspension_row_volume)) # 當日交易量小於1000萬的個股數/整體股票數
    stk_fluidity_index = pd.DataFrame({'mkt_amount': mkt_amount,
                                       'stk_fluidity0': stk_fluidity0,
                                       'stk_fluidity1': stk_fluidity1}, index=amount['date'])
    
    print('create mkt_amount, stk_fluidity0, stk_fluidity1 done.')
    return stk_fluidity_index

def create_volatility():
    volatility_change_all = load_volatility()
    mkt_daycount = []
    for date in volatility_change_all.index:
        row_volatility = np.array(volatility_change_all.loc[date])
        row_volatility = row_volatility[~np.isnan(row_volatility)]
        mkt_daycount.append(np.sum(row_volatility) / len(row_volatility))
    volatility_index = pd.DataFrame({'mkt_volatility': mkt_daycount}, index=volatility_change_all.index)
    volatility_index = volatility_index.reset_index()
    print('create mkt_volatility done.')
    return volatility_index

def create_ind_volatility():
    volatility_change_all = load_volatility()
    industry = load_industry()

    ind_list = list(set(industry.ind))
    ind_dict = {}
    for ind in ind_list:
        ind_dict[ind] = list(industry[industry['ind'] == ind]['code']) # (ind, code) mapping

    ind_daycount_all = pd.DataFrame(columns=['date', 'ind_volatility', 'ind'])
    for ind in ind_list:
        ind_daycount = []
        target_code = ind_dict[ind]
        for day in volatility_change_all[target_code].index:
            row_volatility = np.array(volatility_change_all[target_code].loc[day])
            row_volatility = row_volatility[~np.isnan(row_volatility)]
            if len(row_volatility) ==0:
                ind_daycount.append(0)
            else:
                ind_daycount.append(np.sum(row_volatility) / len(row_volatility))

        ind_daycount_index = pd.DataFrame({'ind_volatility': ind_daycount}, index=volatility_change_all[target_code].index)
        ind_daycount_index = ind_daycount_index.reset_index()
        ind_daycount_index['ind'] = ind
        ind_daycount_all = pd.concat([ind_daycount_all, ind_daycount_index], axis=0)
    print('create industry volatility done.')
    return ind_daycount_all

def create_mood():
    mood = pd.read_csv(mood_path[0])
    mood['code'] = _code_format_trans(mood)
    return mood 

def fillna_no_trade(df, start_dt, end_dt):
    """
    :param start_dt: 新股(最早的日期), 退市/停牌皆已在起始判斷補值
    :param end_dt: 退市(最後的日期), 正常/新股/停牌以最新日期為主
    """
    index_stock = ts.get_k_data('sh', start= start_dt, end= end_dt)[['date']]
    df = pd.merge(index_stock, df, how='left', on='date')
    df['volume'] = df.volume.fillna(0)
    df['close'] = df.close.fillna(method='ffill')
    df = df.fillna(method='bfill', axis='columns') # 從右向左填充
    return df

def deliver_data():
    make_amount_data()
    make_volatility_5_22_data()
    make_volatility_change_data()

    codeList = get_codelist()
    mood = create_mood()
    stk_fluidity = create_stk_fluidity()
    volatility = create_volatility()
    ind_volatility = create_ind_volatility()
    code2ind = _code_ind_mapping()
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'mood', 'mkt_amount', 'stk_fluidity0', 'stk_fluidity1',
                'mkt_volatility', 'ind_volatility']
    
    existList, _ = get_list_info()
    error_code = []
    for code in codeList:
        try:
            df = pd.read_csv(os.path.join(basic_stock_path, code))[columns[:6]]
            df_start_dt = min(df.date)
            df_end_dt = ''
            # 退市股票
            if code[:-4] not in existList:
                df_end_dt = max(df.date)
            
            df = fillna_no_trade(df, df_start_dt, df_end_dt)           
            df = df[df['date'] >= '2014-01-02']
            df['ind'] = code2ind[code[:-4]]
            df['code'] = code[:-7]
            df = pd.merge(df, stk_fluidity, how='left', on='date')
            df = pd.merge(df, volatility, how='left', on='date')
            df = pd.merge(df, ind_volatility, how='left', on=['date', 'ind']).drop(['ind'], axis=1)
            df = pd.merge(df, mood, how='left', on=['date', 'code']).drop(['code'], axis=1)
            df = df.rename(columns={'grades':'mood'})
            df['mood'] = df['mood'].fillna(0)
            df = df[columns]
            df = df.drop_duplicates()
            df.to_csv(os.path.join(deliver_data_path, code), index=False)
            print('save {} done'.format(code))
        except:
            error_code.append(code)
    print(error_code)
    print('deliver done.')

if __name__ == '__main__':
    deliver_data()
