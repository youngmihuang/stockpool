# -*_ coding: utf-8 -*-
"""
Created on Apr 28 00:13:10 2019
@author: Shao Xuan Huang
@email : HUANGSHOAXUAN801@pingan.com.cn

每日市场、行情数据更新
"""

__all__     = ['main']
__version__ = '0.1.0'

import os 
import csv
import time
import yaml
import datetime
import pandas as pd
import numpy as np
import tushare as ts
from tqdm import tqdm
from datetime import date
from os import listdir
from time import sleep

path = './config.yml'
with open(path, encoding='utf-8') as f:
    config = yaml.load(f)

basic_stock_path  = config['path']['daily_stock_dir']
new_stock_path    = config['path']['daily_new_dir']
index_data_path   = config['path']['daily_index_dir']
model_path        = config['path']['model_dir']

all_mkt_code      = config['base']['all_mkt_code']
token             = config['base']['ts_token']
mkt_index         = config['base']['mkt_index']
file_mkt          = config['raw_update']['daily_mkt']

dm_amount         = config['dm_generate']['cal_amount']
dm_vol_5d         = config['dm_generate']['cal_5d_vol']
dm_vol_22d        = config['dm_generate']['cal_22d_vol']
dm_vol_change     = config['dm_generate']['cal_vol_change']
dm_basic_5        = config['dm_generate']['generate_dm_basic_5']
dm_total_feats    = config['dm_generate']['generate_total_feats']
error_log         = config['dm_generate']['error_log']
zero_log          = config['dm_generate']['zero_log']

ts.set_token(token)
pro = ts.pro_api()

class db_daily(object):
    def __init__(self):
        self.get_all_stock()

    def get_all_stock(self):
        with open(all_mkt_code, 'r') as f:
            codeList = f.read().split('\n')
        self.codeList = [i[1:-1]+'.csv' for i in codeList]

    def get_mkt(self, code):
        origin_cols = ['trade_date', 'pct_chg', 'close']
        pro = ts.pro_api()
        df = pro.index_daily(ts_code=code, start_date= '20140102', end_date='')
        df = df[origin_cols]
        df[origin_cols[0]] = [(datetime.datetime.strptime(i , '%Y%m%d').date()).strftime('%Y-%m-%d') for i in df[origin_cols[0]]]
        df[origin_cols[1]] = [float(i)/100 for i in df[origin_cols[1]]]

        # 产出 for model 使用
        if code in mkt_index[1:]:
            columns_dict = {origin_cols[0]:'date', origin_cols[1]: 'returns_mkt', origin_cols[2]:'close_mkt'}
        # 产出 for RL 使用
        else:
            columns_dict = {origin_cols[0]:'date', origin_cols[1]: 'return', origin_cols[2]:'close'}
        
        df = df.rename(columns=columns_dict)
        df = df.sort_values(by=['date'])
        return df 

    def update_mkt_data(self):
        for i in range(len(mkt_index)):
            df = self.get_mkt(mkt_index[i])     
            if i == 0:
                df.to_csv(file_mkt[i], index=False)
                print('Update hs300 Done.')
            else:
                df.to_csv(os.path.join(model_path, file_mkt[i]), index=False)
                print('Update {} Done.'.format(file_mkt[i]))

    def update_new_stock_data(self, error_status=False):
        cols = ['date','P_change','Price_change','amount','close','high','low','open','turnoverratio','volume']
        cols_dict = {'ts_code': 'code', 'trade_date':'date', 'pct_chg':'P_change', 'change':'Price_change', 'vol':'volume', 'turnover_rate': 'turnoverratio'}
        if error_status:
            with open(os.path.join(new_stock_path, dm_error_log), 'r') as f:
                codeList = f.read().strip().split(',')
                print('error mode starting')
        error_code = []
        zero_code = []
        pbar = tqdm(self.codeList)
        print('Start update new stock..')
        for code in pbar:
            origin = pd.read_csv(os.path.join(basic_stock_path, code))
            origin_max_dt = max(origin.date)
            origin_max_dt_next = (datetime.datetime.strptime(origin_max_dt, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')       
            origin_max_dt_next = origin_max_dt_next.replace('-', '')
            today_dt = datetime.date.today().strftime('%Y-%m-%d')
            time.sleep(0.5)

            # 目前數據是最新的就不往下跑
            if origin_max_dt == today_dt:
                print('New {} is up-to-date'.format(code[:-4]))
                continue

            try:
                df = ts.pro_bar(pro_api=pro, ts_code=code[:-4], start_date=origin_max_dt_next, end_date='', factors='tor')
                # 停牌股票該交易日沒資料 or 隔日行情數據尚未更新就不回寫
                if len(df) == 0:
                    zero_code.append(code)                
                    continue
                
                df = df.rename(columns= cols_dict)
                df['date'] = [(datetime.datetime.strptime(i , '%Y%m%d').date()).strftime('%Y-%m-%d') for i in df.date]
                df['amount'] = [round(float(i)/10, 2) for i in df.amount] # pro版(單位:千元) => 原始數據 (單位:萬元)
                df['volume'] = [round(i) for i in df.volume]
                df = df.sort_values(by=['date'])
                df = df.drop(['pre_close', 'code'], axis=1)
                df = df[cols]
                df.to_csv(os.path.join(new_stock_path, code), index=False)
                
                if origin_max_dt < max(df.date):
                    origin = pd.concat([origin, df], axis=0, sort=False)
                    origin = origin.drop_duplicates()                     # 確保 raw data 唯一無重複
                    origin.to_csv(os.path.join(basic_stock_path, code), index=False)

                else:
                    print('{} is up-to-date.'.format(code[:-4]))
                    continue
            except Exception as e:
                print(e)
                error_code.append(code)

        print(error_code)
        print(zero_code)

        with open(os.path.join(new_stock_path, error_log), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(error_code)
        with open(os.path.join(new_stock_path, zero_log), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(zero_code)
        print('Update new stocks Done.')
        
        if len(error_code) == 0:
            return 0
        elif error_status:
            return 0
        else:
            self.update_new_stock_data(error_status=True)
        

if __name__ == '__main__':
    db_daily().update_mkt_data()


