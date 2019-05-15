# -*_ coding: utf-8 -*-
"""
Created on Apr 27 00:13:10 2019
@author: Youngmi Huang
@email: cyeninesky3@gmail.com

根据当前股票池与每月选股模型产生的上涨概率，建立换股逻辑
"""
__all__     = ['batch_update', 'get_codelist_info']
__version__ = '0.1.0'


import os
import yaml
import pandas as pd
from pandas import  DataFrame
from datetime import datetime

path = './config.yml'
with open(path, encoding='utf-8') as f:
    config = yaml.load(f)

model_path          = config['path']['model_dir']
select_strategy     = config['select_stocks']['strategy_name']
performance         = config['select_stocks']['performance']
code_list_info      = config['select_stocks']['result_for_RL']

performance_path    = os.path.join(model_path, performance)
code_list_info_hist = os.path.join(model_path, config['select_stocks']['result_for_RL_backup'])

def update_remark(config_RL, target_code_list):
    """
    :param config_RL: 挑出指定的策略
    :param target_code_list: 每月更新的 performance table 

    """
    
    target_code_list = target_code_list[target_code_list.remarks == config_RL['remark']]
    target_code_list = target_code_list[['select_dt', 'start_dt', 'end_dt', 'select_code', 'up_prob', 'remarks']]
    target_code_list = target_code_list.reset_index(drop=True)

    indexing  = list(target_code_list.index)

    start_pool = []
    new_pool   = []
    change_all = []
    for ind in indexing:
        if ind == 0:
            start_pool = eval(target_code_list['select_code'].loc[ind]) # 最早的池子
            change_all.append(start_pool)
        else:
            new_target = eval(target_code_list['select_code'].loc[ind])
            up_prob    = eval(target_code_list['up_prob'].loc[ind])
            no_trade   = [i for i in start_pool if i not in list(up_prob.keys())]
        
            Notrade_pool = list(set(start_pool).intersection(set(no_trade)))            
            Intersection_pool = list(set(start_pool).intersection(set(new_target)))
            start_pool_minus_intersection =[ i for i in start_pool if i not in Intersection_pool ]

            # 補停牌股票在股票池概率(在候選名單內不能被選; 在池子內也不能剔除因為無法交易)
            Inf            = 1
            for x in Notrade_pool:
                up_prob[x] = Inf
            
            # 抓概率
            start_pool_prob = [ up_prob[x] for x in start_pool_minus_intersection ] 
            # 挑出三隻最小的位置
            min_prob_stock = []
            Inf            = 2        
            for i in range(1):
                min_prob_stock.append(start_pool_prob.index(min(start_pool_prob)))
                start_pool_prob[start_pool_prob.index(min(start_pool_prob))] = Inf
            minus_stock = [start_pool_minus_intersection[y] for y in min_prob_stock]
            print('delete 1: {}'.format(minus_stock))

            #扣掉三個上漲概率低的股票
            start_pool_minus_low_up_prob = [ i for i in start_pool if i not in minus_stock ]
            
            #---------------------------------------------------------------------------------------
            new_target_minus_intersection =[ i for i in new_target if i not in Intersection_pool ]
            max_prob_stock = []
            Inf            = 0
            # 抓概率
            new_target_prob = [  up_prob[x]  for x in new_target_minus_intersection]
            # 挑出三隻最大的位置
            for i in range(1):
                max_prob_stock.append(new_target_prob.index(max(new_target_prob)))
                new_target_prob[new_target_prob.index(max(new_target_prob))] = Inf
            add_stock = [new_target_minus_intersection[y] for y in max_prob_stock]
            #加上三個上漲概率高的股票
            start_pool = start_pool_minus_low_up_prob + add_stock
            change_all.append(start_pool)
    target_code_list['select_code'] = change_all    
    return target_code_list

def batch_update():
    """
    歷史回測: 一次性更新剔除新增股票池內的股票
    """
    today_dt = datetime.now().date().strftime('%Y-%m-%d')
    config_RL = {}
    target_code_list = pd.read_csv(performance_path)
    stockpool_total = pd.DataFrame()
    
    for i in range(len(select_strategy)):
        config_RL['remark'] = select_strategy[i]
        stockpool_by_remark = update_remark(config_RL, target_code_list)
        stockpool_total = pd.concat([stockpool_total, stockpool_by_remark], axis=0, sort=False)

    stockpool_total = stockpool_total.drop(['up_prob'], axis=1)
    stockpool_total.to_csv(code_list_info, index=False)
    stockpool_total.to_csv(code_list_info_hist, index=False)

def get_codelist_info():
    """
    模型每月更新: 設定起始股票池換股，以及新的up_prob, 讀取 performance_hist 最新和 codelist_info 最新的日期數據
    stockpool_by_remark: 共有6個策略的換股操作 (alpha01, RR1 ~ RR5)
    """

    cols = ['select_dt', 'start_dt', 'end_dt', 'select_code', 'up_prob', 'remarks']
    stockpool_total = pd.read_csv(code_list_info)               
    stockpool_current = pd.read_csv(performance_path)        # backup
    select_dt_total = max(stockpool_total.select_dt)
    select_dt_currrent = max(stockpool_current.select_dt)

    # 如果 codelist_info 已存在最新的股票池更新结果, 不更新
    if select_dt_total == select_dt_currrent:
        print('current select_dt: {} is already exists in codelist_info.'.format(select_dt_currrent))
        return 0

    # 从 code_list_info 取当前股票池
    initial = stockpool_total[stockpool_total['end_dt'] == select_dt_total]     # current select_dt = last end_dt
    stockpool_current = stockpool_current[stockpool_current['select_dt'] == select_dt_currrent][cols]
    stockpool_before_change = pd.concat([initial, stockpool_current], axis=0, sort=False)


    stockpool_after_change = pd.DataFrame()     
    config_RL = {}
    for i in range(len(select_strategy)):
        config_RL['remark'] = select_strategy[i]
        stockpool_by_remark = update_remark(config_RL, stockpool_before_change)
        stockpool_after_change = pd.concat([stockpool_after_change, stockpool_by_remark], axis=0, sort=False)
    stockpool_after_change = stockpool_after_change[stockpool_after_change['select_dt'] == select_dt_currrent].drop(['up_prob'], axis=1)
    stockpool_total = pd.concat([stockpool_total, stockpool_after_change], axis=0, sort=False)
    stockpool_total.to_csv(code_list_info, index=False)                 # save to RL Usage
    stockpool_total.to_csv(code_list_info_hist, index=False)  # backup
    return stockpool_total

if __name__ == '__main__':
    get_codelist_info()