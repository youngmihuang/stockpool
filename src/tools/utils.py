# -*_ coding: utf-8 -*-
'''
Editor: Youngmi huang
Update: 2019/02/26

选股模型会用到的函数
'''

import numpy as np 
import pandas as pd 
import pickle
import calendar
import datetime
from datetime import date, timedelta
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import lightgbm as lgb


# 給到本月的 start_end + 下個月的 start and end
def get_start_end_dt(df, start_m, end_m):
    start_dt = []
    end_dt = []
    dates ={}
    select_dt = get_select_dt(start_m, end_m)
    today_dt = datetime.datetime.now().date().strftime('%Y-%m-%d')
    select_dt = [i for i in select_dt if i <= today_dt]
    num_select_dt = len(select_dt)
    if num_select_dt >1:
        for i in range(num_select_dt - 1):
            mask1 = df['date'] > select_dt[i]
            mask2 = df['date'] <= select_dt[i+1]
            dates_list = df[(mask1 & mask2)].date.tolist()
            start_dt.append(min(dates_list))
            end_dt.append(max(dates_list))

        for i in range(num_select_dt-1):
            dates[select_dt[i]] = (start_dt[i], end_dt[i])

    last_select_dt = datetime.datetime.strptime(select_dt[-1], '%Y-%m-%d')     # 當月最後一個 Fri +3天 = Monday
    next_start = (last_select_dt + timedelta(days=3)).strftime('%Y-%m-%d') 
    
    # 判断下个月的年月获得下个月最后一个星期五日期
    if last_select_dt.month < 12:
        year = last_select_dt.year
        month = last_select_dt.month +1

    else:
        year = last_select_dt.year +1
        month = 1    
        
    _, next_last_dt = getMonthFirstDayAndLastDay(year, month)
    next_end = get_last_byday('Friday', next_last_dt)
    
    # 若当月更新，则 select_dt 只有一笔日期数据
    if num_select_dt == 1:
        dates[select_dt[0]] = (next_start, next_end)
        return dates
    # 若batch更新，则设定 select_dt 最后一笔日期数据
    else:
        dates[select_dt[-1]] = (next_start, next_end)
        return dates

def get_select_dt(start_m, end_m, day='Friday'):
    """
    :param initial_year: select year
    :param initial_month: select month
    :return select_dt: dict, the last Fridays of the selected (year, month); if hist=False, the dict only contains 1 select_dt 
    """
    initial_year, initial_month = int(str(start_m)[:4]), int(str(start_m)[4:])
    _, initial_last_dt = getMonthFirstDayAndLastDay(initial_year, initial_month)
    initial_select_dt = get_last_byday(day, initial_last_dt)
    end_year, end_month = int(str(end_m)[:4]), int(str(end_m)[4:])
    current_year, current_month, current_quarter = current_time()
    select_dt = []
    run_dt = []

    while initial_year < end_year:
        for month in range(initial_month, 13):
            run_dt.append((initial_year, month))
        initial_year += 1
        initial_month = 1
    
    # 跑最後一年 initial_year = end_year
    for month in range(initial_month, end_month+1):
        run_dt.append((initial_year, month))

    for year, month in run_dt:
        _ , end_dt = getMonthFirstDayAndLastDay(year, month)
        select_dt.append(get_last_byday(day, end_dt))         # 最后一个星期五                          
    return select_dt

def get_select_dt_v0(initial_year=2015, initial_month=12, day='Friday'):
    """
    :param initial_year: select year
    :param initial_month: select month
    :return select_dt: dict, the last Fridays of the selected (year, month); if hist=False, the dict only contains 1 select_dt 
    """
    _, initial_last_dt = getMonthFirstDayAndLastDay(initial_year, initial_month)
    initial_select_dt = get_last_byday(day, initial_last_dt)
    current_year, current_month, current_quarter = current_time()
    select_dt = []
    run_dt = []

    while initial_year < current_year:
        for month in range(initial_month, 13):
            run_dt.append((initial_year, month))
        initial_year += 1
        initial_month = 1
    
    # 跑最後一年 initial_year = current_year
    for month in range(initial_month, current_month+1):
        run_dt.append((initial_year, month))

    for year, month in run_dt:
        _ , end_dt = getMonthFirstDayAndLastDay(year, month)
        select_dt.append(get_last_byday(day, end_dt))                                  
    return select_dt

def holding_period_returns(arr):
    """
    :param arr: daily returns list 
    :return: holding period returns = (1+r)^N - 1 , N = holding period
    """
    hpr = list(np.cumprod(arr+1))[-1]
    return hpr -1

def get_excess_returns(arr_p, arr_mkt):
    returns_p = holding_period_returns(arr_p)
    returns_mkt = holding_period_returns(arr_mkt)
    return returns_p - returns_mkt

def get_max_drawdown(arr):
    cum_max = (arr+1).cummax()
    cum_min = (arr+1).cummin()
    return max((cum_max - cum_min)/cum_max)

def get_info_ratio(arr_p, arr_mkt):
    excess_returns = get_excess_returns(arr_p, arr_mkt)
    diff = arr_p - arr_mkt       # daily    
    std = diff.std()*np.sqrt(22) # monthly
    return excess_returns/std

def get_calmr_ratio(arr_p):
    returns_p = holding_period_returns(arr_p)
    max_drawdown = get_max_drawdown(arr_p)
    return returns_p/max_drawdown

def get_sharpe_ratio(arr_p, arr_mkt):
    excess_returns = get_excess_returns(arr_p, arr_mkt)
    volatility = arr_p.std()*np.sqrt(22)
    return excess_returns/volatility

def current_time():
    now = datetime.datetime.now()
    current_year = now.year
    current_month = now.month
    first_month_quarter = (current_month-1) - (current_month-1)%3 +1
    current_quarter, _ = divmod(first_month_quarter+3, 3)
    return current_year, current_month, current_quarter

def get_last_byday(dayname, end_date=None):
    """
    :param end_date: last date of the month
    :return target_date: last friday date of the month (string)
    
    """
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday']

    if end_date is None:
        end_date = datetime.today()
        print(end_date)
    if type(end_date) == str:
        end_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')
        
    day_num = end_date.weekday()                   # 最後一天是星期幾
    day_num_target = weekdays.index(dayname)       # 目標是星期五
    days_ago = (7 + day_num - day_num_target) % 7  # 最後一天跟目標差幾天

    if days_ago == 0:
        return end_date.strftime('%Y-%m-%d')
    
    else:
        target_date = end_date - timedelta(days=days_ago)
        return target_date.strftime('%Y-%m-%d')

def getMonthFirstDayAndLastDay(year=None, month=None):
    """
    :param year: 年份，默认是本年，可传int或str类型
    :param month: 月份，默认是本月，可传int或str类型
    :return firstDay: 当月的第一天，datetime.date类型
            lastDay: 当月的最后一天，datetime.date类型
    """
    if year:
        year = int(year)
    else:
        year = datetime.date.today().year

    if month:
        month = int(month)
    else:
        month = datetime.date.today().month

    # 获取当月第一天的星期和当月的总天数
    firstDayWeekDay, monthRange = calendar.monthrange(year, month)

    # 获取当月的第一天
    firstDay = str(datetime.date(year=year, month=month, day=1))
    lastDay = str(datetime.date(year=year, month=month, day=monthRange))
    return firstDay, lastDay

def get_yearly_return(arr):
    return (1+get_monthly_return(arr))**12 -1

def auc_scores(y_true, y_pred):
    y_true= y_true.copy()
    y_true = (y_true>0).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    print('auc: %.4f' % (auc_score))
    return auc_score

def precision_scores(y_true, y_pred):
    precision = precision_score(y_true, (y_pred>0.5).astype(int))
    print('precision: %.4f' % (precision))
    return precision

def plot_cumulative_return(date, arr_p, arr_mkt, strategy='default'):
    """
    Plot the next month peroformance evaluation between stockpool and hs300 average index.
    """
    cumret_p = (arr_p+1).cumprod()
    cumret_mkt = (arr_mkt+1).cumprod()
    fig = plt.figure(figsize=(12,5))
    plt.plot(date, cumret_mkt, label= 'cum_return_mkt')
    plt.plot(date, cumret_p, label='cum_return_portfolio')
    fig.autofmt_xdate()
    plt.legend()
    # plt.savefig('../csv/result/{}_{}.png'.format(strategy, str(min(date)) + '_' + str(max(date))))
    plt.show()

def plot_feature_importance(model):
    """
    Plot the feature importance of lightgbm model.
    """
    importance = model.feature_importance(importance_type='split')
    feature_name = model.feature_name()
    feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance}).sort_values(by='importance', ascending=False)
    print(feature_importance)
    plt.figure(figsize=(20,10))
    lgb.plot_importance(model, max_num_features=20)
    plt.title("Featurer_imprtances")
    plt.show()


# if __name__ == '__main__':
#     df = load_hs300_average()
#     print(interval_setting(df))