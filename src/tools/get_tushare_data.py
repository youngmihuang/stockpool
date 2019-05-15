# -*_ coding: utf-8 -*-
"""
Created on Apr 11 10:50:12 2019
@author: Youngmi Huang
@email: cyeninesky3@gmail.com

get mkt data from tushare
"""
import tushare as ts 
import pandas as pd
import datetime

ts.set_token('2c7f8a125014cb1050c7d16ca13821c7e22edd2b5818c3e53059bb6d')
pro = ts.pro_api()

def get_mkt_index_weight(code):
     df = pro.index_weight(index_code=code)
     df = _preprocess(df)
     return df

def get_index_basic():
     df = pro.index_basic(market='SSE')
     return df 

def _preprocess(df):
     cols = {'con_code':'code', 'trade_date': 'date'}
     df = df.rename(columns=cols)
     df['date'] = [datetime.datetime.strptime(str(i), '%Y%m%d').date().strftime('%Y-%m-%d') for i in df.date]
     df = df.drop('index_code', axis=1)
     # df.to_csv('../../csv/model/sh50.csv', index=False)
     return df 

# def get_daily():
#      codeList = ['']

     
# 上證50
code = '000016.SH'
df = pd.read_csv('../../csv/model/hist_sh50.csv')
print(len(df))
print(len(list(set(df[df['date']< '2014-01-01'].code))))
# print(df.groupby(['date'])['code'].count())

