# -*_ coding: utf-8 -*-
"""
Created on Apr 30 00:13:10 2019
@author: Youngmi Huang
@email: cyeninesky3@gmail.com

选股模型的特徵构建
"""
import os
import yaml
import time
import datetime
import regex 
import pickle
import numpy as np 
import pandas as pd 
import tushare as ts 
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

path = './config.yml'
with open(path, encoding='utf-8') as f:
    config = yaml.load(f)

all_mkt_code          = config['base']['all_mkt_code']
all_industry          = config['base']['industry']
create_feature_status = config['raw_update']['batch_create_feature']
batch_start_dt        = config['raw_update']['batch_start_dt']

basic_stock_path      = config['path']['daily_stock_dir']
model_path            = config['path']['model_dir']
mood_path             = config['raw_update']['daily_ola']

hs300_compose_path    = os.path.join(model_path, config['base']['mkt_compose'][0])
sh50_compose_path     = os.path.join(model_path, config['base']['mkt_compose'][1])
basic_5_path          = os.path.join(model_path, config['dm_generate']['generate_dm_basic_5'])
feature_path          = os.path.join(model_path, config['dm_generate']['generate_total_feats'])

def load_dm():
    df = pd.read_csv(basic_5_path)
    df = df.replace('--', 'NaN')
    df['report_date'] = [(datetime.datetime.strptime(i, '%Y-%m-%d').date()).strftime('%Y-%m-%d') for i in df['report_date']]
    return df

def load_basic_data(code):
    df = pd.read_csv(os.path.join(basic_stock_path, '{}.csv'.format(code)))
    return df

def load_features():
    with open(feature_path, 'rb') as f:
        df = pickle.load(f)
    return df

def _create_momentum_return(data, month):
    try:
        result =[]
        window = month*22
        close = np.array(data.close)        
        for index in range(len(data) - window + 1):
            past = close[index]
            curr = close[index + window -1]
            result.append((curr-past)/past)
        
        df = pd.DataFrame()
        df['momentum_return_{}m'.format(month)] = result
        df['date'] = pd.DataFrame(np.array(data.date[(window-1): ]))
    except:
        df = pd.DataFrame(columns=['momentum_return_{}m'.format(month), 'date'])
    return df 

def _create_momentum_weightedreturn(data, month):
    try:
        result=[]
        window = month*22
        p_return = np.array([i for i in data.P_change])
        p_chg = np.array([i for i in data.turnoverratio])
        p_return_chg = p_return * p_chg
        for index in range(len(data) - window + 1):
            avg = np.mean(p_return_chg[index:(index+window)])
            result.append(avg)

        df = pd.DataFrame()
        df['momentum_reverse_{}m'.format(month)] = result
        df['date'] = pd.DataFrame(np.array(data.date[(window-1): ]))
    except:
        df = pd.DataFrame(columns=['momentum_reverse_{}m'.format(month), 'date'])
    return df

def _create_momentum_weightedreturn_exp(data, month):
    try:
        result=[]
        window = month*22
        p_return = np.array([i for i in data.P_change])
        p_turnover = np.array([i for i in data.turnoverratio])
        for index in range(len(data) - window + 1):
            rrate = p_return[index:(index+window)]
            chg = p_turnover[index:(index+window)]
            last = (len(rrate) - 1) * [rrate[-1]]           
            diff = list(map(lambda x: abs(x[0] - x[1]), zip(rrate[:-1], last)))
            diff_min_max = diff.index(np.min(diff))
            diff_days = window - diff_min_max - 1          # 距離第t天價差最小的交易日天數
            result.append(rrate[-1] * chg[-1] * np.exp(-diff_days/month/4))

        df = pd.DataFrame()
        df['momentum_return_exp{}m'.format(month)] = result
        df['date'] = pd.DataFrame(np.array(data.date[(window-1): ]))
    except:
        df = pd.DataFrame(columns=['momentum_return_exp{}m'.format(month), 'date'])
    return df

def _create_monthly_return(df, col):
    df = df.copy()
    window = 22 
    returns_compounded = []
    df['returns_daily'] = ((df[col] - df[col].shift(1))/ df[col].shift(1)).fillna(0)
    for i in range(len(df)- window+1):
        returns = 1
        units = df['returns_daily'][i:i+window].values
        for unit in units:
            returns = returns*(1+unit)
        returns_compounded.append(returns-1)
    returns_last_fillna = list(np.zeros(window-1))
    return np.array(returns_compounded + returns_last_fillna)

def _create_price_std_m(data, month):
    try:
        result = []
        window = month * 22
        rr = np.array([i for i in data.P_change])
        for index in range(len(data) - window + 1):
            rr_std = np.std(rr[index:(index+window)])
            result.append(rr_std)
        
        df = pd.DataFrame()
        df['rr_std_{}m'.format(month)] = result
        df['date'] = pd.DataFrame(np.array(data.date[(window-1): ]))
    except:
        df = pd.DataFrame(columns=['rr_std_{}m'.format(month), 'date'])
    return df 

def _create_turnoverratio_m(data, month):
    try:
        result = []
        window = month * 22
        p_turnover = np.array([i for i in data.turnoverratio])
        for index in range(len(data) - window + 1):
            avg = np.mean(p_turnover[index:(index+window)])
            result.append(avg)
        
        df = pd.DataFrame()
        df['avg_turnoverratio_{}m'.format(month)] = result
        df['date'] = pd.DataFrame(np.array(data.date[(window-1): ]))
    except:
        df = pd.DataFrame(columns=['avg_turnoverratio_{}m'.format(month), 'date'])
    return df

def _roa(df):
    df = df[['roe', 'sheqratio']].astype(float)
    df = df.dropna(how='any')
    return df.roe * df.sheqratio

def _code_foramt_trans(df):
    arr = np.array([str(i).zfill(6) for i in df.code])
    return arr 

def _daily_dropna(df):
    tmp = df[['code', 'date', 'ln_price']]
    df = df.drop(['code', 'date', 'ln_price'], axis=1).dropna(how='all')
    df = pd.concat([tmp, df], axis=1, sort=False)
    # df = df.dropna(thresh = round(len(df.columns)*0.75)).reset_index(drop=True)
    return df 

def _fillna_forward_back(df):
    df = df.fillna(method='ffill').fillna(0)
    return df 

def _merge_two(df1, df2):
    return pd.merge(df1, df2, on=['code', 'date'], how='outer')

# Technical index
def categ_leverage(data):
    df = pd.DataFrame()
    df['code'] = _code_foramt_trans(data)
    df['current_ratio'] = [i/100 for i in data['currentratio'].astype(float)]  # 流動比率
    df['cash_ratio'] = [i/100 for i in data['cashratio'].astype(float)]        # 現金比率
    df['date'] = data['report_date']
    df = df.replace('--', 'NaN')
    return df 

# Growth
def categ_growth(data):
    df = pd.DataFrame()
    df['code'] = _code_foramt_trans(data)
    df['sales_growth_q'] = data['mbrg']        # 主營業務收入增長率
    df['net_profit_growth_q'] = data['nprg']   # 淨利潤增長率
    df['roe_growth_q'] = data['seg']           # 股東權益增長率
    df['date'] = data['report_date']
    return df

# Financial quality
def categ_fin_quality(data):
    df = pd.DataFrame()
    df['code'] = _code_foramt_trans(data)
    df['gross_profit_ratio_q'] = data['gross_profit_rate']        # 毛利率
    df['net_profit_ratio_q'] = data['net_profit_ratio']           # 淨利率
    df['receivable_turnover_q'] = data['arturnover']              # 應收帳款週轉率
    df['inventory_turnover_q'] = data['inventory_turnover']       # 存貨週轉率
    df['currentasset_turnover_q'] = data['currentasset_turnover'] # 流動資產週轉率
    df['cf_nprofit_ratio_q'] = data['cf_nm']                      # 經營現金流量對淨利潤比率
    df['roe_q'] = data['roe']
    df['roa_q'] = _roa(data)                                      # 資產報酬率 = 股東權益比率(equity/asset) * roe 
    df['date'] = data['report_date']
    return df

# Momentum
def create_momentum_return(data, code):
    months = [3,6,12]
    df = _create_momentum_return(data, 1)       # initial month = first month
    for month in months:
        add = _create_momentum_return(data, month)
        df = pd.merge(df, add, on='date', how='left')
    df['code'] = code[:-3]
    return df

def create_momentum_weightedreturn(data, code):
    months = [3,6,12]
    df = _create_momentum_weightedreturn(data, 1)       # initial month = first month
    for month in months:
        add = _create_momentum_weightedreturn(data, month)

        df = pd.merge(df, add, on='date', how='left')
    df['code'] = code[:-3]
    return df

def create_momentum_weightedreturn_exp(data, code):
    months = [3,6,12]
    df = _create_momentum_weightedreturn_exp(data, 1)       # initial month = first month

    for month in months:
        add = _create_momentum_weightedreturn_exp(data, month)
        df = pd.merge(df, add, on='date', how='left')
    df['code'] = code[:-3]
    return df

def create_ln_price(data, code):
    df = pd.DataFrame()
    df['ln_price'] = np.log(data['close'])
    df['returns'] = [i/100 for i in data.P_change]
    df['close'] = data['close']
    df['open'] = data['open']
    df['high'] = data['high']
    df['low'] = data['low']
    df['avg_price'] = (data['open']+ data['close']+ data['high']+ data['low'])/4
    df['volume'] = data['volume']     # 成交量
    df['amount'] = data['amount']     # 成交金額
    df['returns_next1m_close'] = _create_monthly_return(df, col= 'close') # 未來一個月的報酬率
    df['returns_next1m_open'] = _create_monthly_return(df, col = 'open')
    df['code'] = code[:-3]
    df['date'] = data['date']
    return df

# Volatility
def create_price_std_m(data, code):
    df = _create_price_std_m(data, 1)
    months = [3, 6, 12]
    for month in months:
        add = _create_price_std_m(data, month)
        df = pd.merge(df, add, on = 'date', how='left')
    df['code'] = code[:-3]
    return df

def create_volatility_data(df, code):
    df = df.copy()
    return_cols = ['code', 'date', 'volatility_5', 'volatility_10']
    df = df.sort_values(by=['date'])
    df = df[['date', 'close']]
    df['log_ret']= np.log(df['close']/df['close'].shift(1))
    df = df.dropna()
    df['volatility_5'] = df['log_ret'].rolling(window=5,center=False).std()
    df['volatility_10']=df['log_ret'].rolling(window=10,center=False).std()
    df['code'] = code[:-3]
    return df[return_cols].fillna(0)

# Turnover ratio
def create_turnoverratio_m(data, code):
    df = _create_turnoverratio_m(data, 1)
    months = [3, 6, 12]
    for month in months:
        add = _create_turnoverratio_m(data, month)
        df = pd.merge(df, add, on = 'date', how='left')
    df['code'] = code[:-3]
    return df

def create_mood():
    df = pd.read_csv(mood_path[0])
    df['code'] = _code_foramt_trans(df)
    return df

def create_industry():
    df = pd.read_csv(all_industry).rename(columns={'ind': 'industry'})
    df['code'] = [i[:-3] for i in df['code']]
    return df

def create_features_quarterly(data):
    feat1 = categ_growth(data)
    feat2 = categ_fin_quality(data)
    feat3 = categ_leverage(data)
    combine = _merge_two(feat1, feat2)
    combine = _merge_two(combine, feat3)
    return combine

def create_features_daily(data, code):
    df = pd.DataFrame()
    feat1 = create_ln_price(data, code)
    feat2 = create_momentum_weightedreturn(data, code)
    feat3 = create_momentum_weightedreturn_exp(data, code)
    feat4 = create_momentum_return(data, code)
    feat5 = create_price_std_m(data, code)
    feat6 = create_turnoverratio_m(data, code)
    feat7 = create_volatility_data(data, code)

    combine = _merge_two(feat1, feat2)
    combine = _merge_two(combine, feat3)
    combine = _merge_two(combine, feat4)
    combine = _merge_two(combine, feat5)
    combine = _merge_two(combine, feat6)
    combine = _merge_two(combine, feat7)

    return combine 

def fillna_quarterly(df):
    tmp = df[['code', 'date']]
    df = df.drop(['code', 'date'], axis = 1)
    df = df.fillna(method='ffill').dropna(how='any')
    df = pd.concat([tmp, df], axis=1, sort=False).dropna(thresh = round(len(df.columns)*0.75)) # 刪除超過3/4欄位是空值的row數據
    df = df.reset_index(drop=True)
    return df 

def fillna_daily(df):
    df = _daily_dropna(df)
    df = _fillna_forward_back(df)
    return df

def quarterly_preprocess():
    df = load_dm()
    df = create_features_quarterly(df)
    df = fillna_quarterly(df)
    print('quarterly features preprocessing done.')
    return df

def daily_preprocess(df, code, max_dt, status):
    df_q = df[df['code'] == code[ :-3]]
    df_d = load_basic_data(code)
    
    if not status:
        df_d = df_d[df_d['date']> max_dt]   # 非 batch 更新的话，就会生成缩小的特徵表

    if len(df_d) == 0:
        return df_d
    else:
        df_d    = create_features_daily(df_d, code)
        df_d    = fillna_daily(df_d)
        dates   = list(df_d['date'])
        combine = pd.merge(df_d, df_q, how='outer', on=['code','date'])
        combine = combine.sort_values(by='date')
        combine = _fillna_forward_back(combine)
        combine = combine[combine['date'].isin(dates)]
        combine = combine.sort_values(by='date').reset_index(drop=True)
        return combine

# 为变动较大的股票打标签(当日涨幅前后30%的数据)
def create_label(df):
    label=[]
    total_samples = len(df)
    num = round(total_samples*0.3)
    for i in range(total_samples):
        if i <= num:
            label.append(1)
        elif i >= total_samples - num:
            label.append(-1)
        else:
            label.append(0)
    return label

def create_code_list():
    code1 = list(set(pd.read_csv(hs300_compose_path).code))
    code2 = list(set(pd.read_csv(sh50_compose_path).code))
    codeList = sorted(list(set(code1+code2)))
    codeList = [code.strip()[1:-1] for code in open(all_mkt_code).readlines() if code.strip()[1:-1] in codeList]
    return codeList

def get_total_features_batch(start_dt=batch_start_dt, status=create_feature_status):
    mood      = create_mood() 
    industry  = create_industry()
    df_q      = quarterly_preprocess()
    codeList  = create_code_list()
    pbar      = tqdm(codeList)

    add_feats = pd.DataFrame()
    for code in pbar:
        add = daily_preprocess(df_q, code, start_dt, status)
        # 跳过不产生该股票的特徵(该股票无新特徵资料)
        if len(add) == 0:
            continue
        add_feats = pd.concat([add_feats, add], axis=0, sort=False)  

    add_feats = pd.merge(add_feats, mood, how='left', on=['code', 'date']).fillna(0)
    add_feats = pd.merge(add_feats, industry, how='left', on=['code'])
 
    # 计算每一天根据市场涨跌排序前后30%的股票打标签，当做 training data
    print('caculating label start')
    start           = time.time()
    dates           = add_feats.date.unique()
    add_feats_label = pd.DataFrame()
    for date in dates:
        add_feats_daily          = add_feats[add_feats['date'] == date]
        add_feats_daily          = add_feats_daily.sort_values(by=['returns_next1m_close'], ascending=False)
        add_feats_daily['label'] = create_label(add_feats_daily)
        add_feats_label = pd.concat([add_feats_label, add_feats_daily], axis=0)
    add_feats_label   = add_feats_label.sort_values(by=['code', 'date']).reset_index(drop=True)
    total_feats_label = pd.concat([origin_feats, add_feats_label], axis=0, sort=False)

    # 回存
    with open('../csv/model/testing.pkl', 'wb') as f:
        pickle.dump(total_feats_label, f, protocol = pickle.HIGHEST_PROTOCOL)
    return total_feats_label

def get_total_features(status=False):
    origin_feats             = load_features()
    origin_feats_max_dt      = max(origin_feats.date)
    origin_feats_max_dt_prev = (datetime.datetime.strptime(origin_feats_max_dt, '%Y-%m-%d') + datetime.timedelta(days=-300)).strftime('%Y-%m-%d') # 因特徵产生最多需要参考近一年的数据
 
    mood      = create_mood() 
    industry  = create_industry()
    codeList  = create_code_list()
    df_q      = quarterly_preprocess()
    pbar      = tqdm(codeList)

    add_feats = pd.DataFrame()
    for code in pbar:
        add = daily_preprocess(df_q, code, origin_feats_max_dt_prev, status)
        # 跳过不产生该股票的特徵(该股票无新特徵资料)
        if len(add) == 0:
            continue
        add_feats = pd.concat([add_feats, add], axis=0, sort=False)  

    add_feats = pd.merge(add_feats, mood, how='left', on=['code', 'date']).fillna(0)
    add_feats = pd.merge(add_feats, industry, how='left', on=['code'])
    add_feats = add_feats[add_feats['date'] > origin_feats_max_dt]

    # 计算每一天根据市场涨跌排序前后30%的股票打标签，当做 training data
    print('caculating label start')
    start           = time.time()
    dates           = add_feats.date.unique()
    add_feats_label = pd.DataFrame()
    for date in dates:
        add_feats_daily          = add_feats[add_feats['date'] == date]
        add_feats_daily          = add_feats_daily.sort_values(by=['returns_next1m_close'], ascending=False)
        add_feats_daily['label'] = create_label(add_feats_daily)
        add_feats_label = pd.concat([add_feats_label, add_feats_daily], axis=0)
    add_feats_label   = add_feats_label.sort_values(by=['code', 'date']).reset_index(drop=True)
    total_feats_label = pd.concat([origin_feats, add_feats_label], axis=0, sort=False)

    # 回存
    with open(feature_path, 'wb') as f:
        pickle.dump(total_feats_label, f, protocol = pickle.HIGHEST_PROTOCOL)
    return total_feats_label
    
if __name__ == '__main__':
    # 如果要从零开始建立, 需至config档设定 batch_create_feature = True
    if create_feature_status:
        get_total_features_batch()
    else:
        get_total_features()

