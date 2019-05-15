# -*_ coding: utf-8 -*-
"""
Created on Apr 28 00:13:10 2019
@author: Shao Xuan Huang
@email : HUANGSHOAXUAN801@pingan.com.cn

每月更新最新的财报数据(虽是季报但因公司发布日不一)
"""
__all__     = ['db_quarterly']
__version__ = '0.1.0'

import os
import yaml
import pickle
import datetime
import numpy as np 
import pandas as pd 
import tushare as ts 
from readconfig import ReadConfig

path = './config.yml'
with open(path, encoding='utf-8') as f:
    config = yaml.load(f)

model_folder   = config['path']['model_dir']
quarter_folder = config['path']['quarterly_dir']
report_dir     = os.path.join(model_folder, quarter_folder[0])
profit_dir     = os.path.join(model_folder, quarter_folder[1])
operation_dir  = os.path.join(model_folder, quarter_folder[2])
growth_dir     = os.path.join(model_folder, quarter_folder[3])
debtpay_dir    = os.path.join(model_folder, quarter_folder[4])
cashflow_dir   = os.path.join(model_folder, quarter_folder[5])
dm_basic_5     = os.path.join(model_folder, config['dm_generate']['generate_dm_basic_5'])

class db_quarterly(object):
    def __init__(self):
        self.current_time()

    def current_time(self):
        now = datetime.datetime.now()
        self.current_year = now.year
        self.current_month = now.month
        first_month_quarter = (now.month-1) - (now.month-1)%3 +1
        self.current_quarter, _ = divmod(first_month_quarter+3, 3)
    
    def _revise_year_quarter(self, year, quarter, month):
        if month in range(1,4):
            year -= 1
            quarter = 4
        else:
            quarter -= 1
        return year, quarter, month
    
    def get_report_basics(self):
        """
        :param year: year during statistical period for the report 
        :param quarter: quarter during statistical period for the report   

        需以業績主檔的 report_date 作為後面五張基本面表的唯一時間key值, report_date 只包含日期, 故在跑的時候須補當前年份;        
        基本面數據爬取需将公布时间转换为財報的統計時間
        e.g. current: 2019/3/30 => current_year= 2019, current_quarter = Q1 
             run    :           =>         year= 2018,         quarter = Q4    
        """

        year, quarter, _ = self._revise_year_quarter(self.current_year, self.current_quarter, self.current_month)
        filename = '{}Q{}.csv'.format(year, quarter)
        print('Now is Quarterly data {}Q{}: report updating..'.format(year, quarter))
        
        base = ts.get_report_data(year, quarter)
        if len(base) == 0:
            print('No {} Q{} data in current data source.'.format(year, quarter))
            return 0

        base['report_date'] = [str(self.current_year)+'-'+i for i in base.report_date]
        base = base.drop(['name'], axis=1)
        base.to_csv(os.path.join(report_dir, filename), index=False)

        # 以下五张的日期栏位依据主表的公布日为准
        base = base[['code', 'report_date']]
        print('Now is Quarterly data {}Q{}: other 5 table updating..'.format(year, quarter))              
        profit = ts.get_profit_data(year, quarter)
        profit = profit.drop(['name'], axis=1)
        profit = pd.merge(base, profit,  how='left', on='code')
        profit = self._fillna_basics_date(profit, year, quarter)
        profit.to_csv(os.path.join(profit_dir, filename), index=False)

        operation = ts.get_operation_data(year, quarter)
        operation = operation.drop(['name'], axis=1)
        operation = pd.merge(base, operation,  how='left', on='code')
        operation = self._fillna_basics_date(operation, year, quarter)
        operation.to_csv(os.path.join(operation_dir, filename), index=False)

        growth = ts.get_growth_data(year, quarter)
        growth = growth.drop(['name'], axis=1)
        growth = pd.merge(base, growth,  how='left', on='code')
        growth = self._fillna_basics_date(growth, year, quarter)                
        growth.to_csv(os.path.join(growth_dir, filename), index=False)

        debtpay = ts.get_debtpaying_data(year, quarter)
        debtpay = debtpay.drop(['name'], axis=1)
        debtpay = pd.merge(base, debtpay, how='left', on='code')
        debtpay = self._fillna_basics_date(debtpay, year, quarter)                
        debtpay.to_csv(os.path.join(debtpay_dir, filename), index=False)

        cashflow = ts.get_cashflow_data(year, quarter)
        cashflow = cashflow.drop(['name'], axis=1)
        cashflow = pd.merge(base, cashflow, how='left', on='code')
        cashflow = self._fillna_basics_date(cashflow, year, quarter)                
        cashflow.to_csv(os.path.join(cashflow_dir, filename), index=False)
        print('Update Quarterly data: {}Q{} Done.'.format(year, quarter))

    def _fillna_basics_date(self, df, year, quarter): 
        df = df.copy()   
        if quarter == 1:
            df['report_date'] = df['report_date'].fillna('{}-04-15'.format(year)) # 由於Q1會與年報的最終日相同，需避免同一天兩個不同數值(Q1財報公布日會早於年報)
        elif quarter == 2:
            df['report_date'] = df['report_date'].fillna('{}-08-31'.format(year))
        elif quarter == 3:
            df['report_date'] = df['report_date'].fillna('{}-10-31'.format(year))
        else:
            df['report_date'] = df['report_date'].fillna('{}-04-30'.format(year+1))  
        return df

    def process_duplicate_public_date(self, df):
        df = df.replace('--','NaN')
        combine = df[['code', 'report_date']].drop_duplicates()
        grp = df.groupby(['code','report_date'])
        for col in df.columns[1:-1]:
            if df[col].dtype == 'O':
                try:
                    df[col] = [float(i) for i in df[col]]
                except:
                    continue
            combine = pd.merge(combine, grp[col].mean().reset_index(), on=['code', 'report_date'], how='outer')
        return combine

    def main_combine_basic5(self):
        """
        同一种资料源需要一起处理，再将异质资料源合并，若有重复值则做简单的加权平均
        """
        self.get_report_basics()
        run_year, run_quarter, _ = self._revise_year_quarter(self.current_year, self.current_quarter, self.current_month)
        total_df = pd.DataFrame(columns=['code', 'report_date'])
        for i in range(1,len(quarter_folder)):
            df = pd.DataFrame()
            # 当年度以前的完整Q1-Q4
            for year in range(2014, run_year):
                for quarter in range(1,5):
                    data = pd.read_csv(os.path.join(model_folder, quarter_folder[i]) + '/{}Q{}.csv'.format(year, quarter))
                    df = pd.concat([df, data], axis=0, sort=False)
            # 当年度到当前季度
            for quarter in range(1, run_quarter+1):
                data = pd.read_csv(os.path.join(model_folder, quarter_folder[i]) + '/{}Q{}.csv'.format(run_year, quarter))
                df = pd.concat([df, data], axis=0, sort=False)                     
            df = self.process_duplicate_public_date(df)
            total_df = pd.merge(total_df, df , on=['code', 'report_date'], how='outer')
        
        total_df = total_df.sort_values(by=['report_date']).reset_index(drop=True)
        total_df = total_df.replace('--', 'NaN')
        total_df = total_df.drop_duplicates()
        total_df.to_csv('../csv/model/dm_basic_5.csv', index=False)
        print('batch combine done.')

if __name__ == '__main__':
    db_quarterly().main_combine_basic5()

# 重爬 2014-2018 Q4
    # root = '../csv/model/raw_data/report/'
    # for year in range(2014,2018):
    #     base = ts.get_report_data(year, 4)
    #     base['report_date'] = [str(year+1)+'-'+i for i in base.report_date]
    #     base = base.drop(['name'], axis=1)
    #     base.to_csv(root + '{}Q{}.csv'.format(year, 4), index=False)
