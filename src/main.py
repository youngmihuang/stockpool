# -*_ coding: utf-8 -*-
"""
Created on Apr 28 00:13:10 2019
@author: Shao Xuan Huang
@email : HUANGSHOAXUAN801@pingan.com.cn

数据更新、选股模型更新
"""
__all__     = ['main']
__version__ = '0.1.0'

from tools.utils import get_select_dt, current_time
from daily_data import db_daily
from ola_connect import main_mood
from deliver_output import deliver_data
from quarterly_data import db_quarterly
from create_feature import get_total_features
from run_model import main_run_model

import pandas as pd
import time
from datetime import datetime


batch_start, batch_end = 201512, 201903

def main(topN, batch_mode=False):
    run_year, run_month, run_quarter = current_time()
    run_start, run_end = int(str(run_year) + str(run_month))
    select_dt        = get_select_dt(run_year, run_month)[0]
    today_dt         = datetime.now().date().strftime('%Y-%m-%d')
    
    print('Today is {}, now updating daily data.'.format(today_dt))
    print('='*80)
    daily_update     = db_daily()
    start = time.time()
    quarterly_update = db_quarterly()
    quarterly_update.main_combine_basic5()                         # 更新季报数据
    print('quarterly spending: {}'.format(time.time()-start))

    # daily 更新
    daily_update.update_mkt_data()                                 # 全市场
    daily_update.update_new_stock_data()                           # 更新行情
    main_mood()                                                    # 更新 ola 舆情
    deliver_data()                                                 # 输出 RL 所需的特徵

    if batch_mode:
        main_run_model(batch_start, batch_end, topN)
        return 0

    # quarterly 更新 (选股日)    
    if today_dt == select_dt:
        quarterly_update.main_combine_basic5()                     # 更新季报数据
        get_total_features()                                       # 模型的特徵构建
        codelist_info = main_run_model(run_start, run_end, topN)  # 更新当月选股结果
        print('Update select model {} {} Done.'.format(run_year, run_month))
        print('='*80)
    else:
        print('No update, select model this month run on {}.'.format(select_dt))
        print('='*80)

if __name__ == '__main__':
    main(topN=7)
