# -*_ coding: utf-8 -*-
"""
Created on Apr 28 00:13:10 2019
@author: Shao Xuan Huang
@email : HUANGSHOAXUAN801@pingan.com.cn

每日更新 ola 舆情数据 (API获取)
"""
__all__     = ['main_mood']
__version__ = '0.1.0'

import os
import yaml
import requests
import numpy as np
import pandas as pd 
import json
import pickle
from datetime import datetime
import csv

path = './config.yml'
with open(path, encoding='utf-8') as f:
    config = yaml.load(f)

mood_dir            = config['path']['daily_ola_dir']
corp_name_path      = config['base']['corp_name']
model_path          = config['path']['model_dir']
mood_path           = config['raw_update']['daily_ola']
error_log           = config['dm_generate']['error_log']
hs300_compose_path  = os.path.join(model_path, config['base']['mkt_compose'][0])
sh50_compose_path   = os.path.join(model_path, config['base']['mkt_compose'][1])
mood_error_path     = os.path.join(mood_dir, error_log)

class APIKeyError(BaseException):
    pass

class ArchiveAPI(object):
    def __init__(self, key, start_dt, end_dt):
        """
        Initializes the ArchiveAPI class. Raises an exception if no API key is given.
        :param key: Ping An API Key
        """
        self.key = key
        self.root_event = 'https://api.pingan.com.cn/open/appsvr/public/api/getEventTimeline?'
        self.start_dt = start_dt
        self.end_dt = end_dt

    def query_event(self, name):
        """
        Calls the archive API and returns the results as a dictionary.
        :param key: Defaults to the API key used to initialize the ArchiveAPI class.
        """
        post_param = {'access_token': self.key,
                'request_id': '1516263062',
                'companyName': name,
                'startDate': self.start_dt,
                'endDate': self.end_dt}
        res = requests.get(url = self.root_event, params=post_param)
        res.encoding = 'utf-8'
        return res.content

    def is_query_valid(self, name):
        """Test the token valid or expired. Token if response not included obj.
        :param name: company name
        """
        res = self.query_event(name)
        if 'obj' not in json.loads(res).keys():
            access_token_page = 'https://api.pingan.com.cn/oauth/oauth2/access_token?client_id=P_GFACE_API&grant_type=client_credentials&client_secret=at15UCj6'
            exception_str = 'Warning: access token is expired. Please visit {}'
            raise APIKeyError(exception_str.format(access_token_page))
        else:
            return True

def get_codelist():
    pool_hs300 = pd.read_csv(hs300_compose_path).code.tolist()
    pool_sh50 = pd.read_csv(sh50_compose_path).code.tolist()
    pool_list = sorted(list(set(pool_hs300 + pool_sh50)))
    return pool_list

def names_info():
    df = pd.read_csv(corp_name_path)
    mask = df['ts_code'].isin(get_codelist())
    df = df[mask].reset_index(drop=True)

    names = df.fullname.tolist()
    names_dict = {}
    for i in range(len(df)):
        names_dict[df['fullname'][i]] = df['ts_code'][i][:-3]
    return names, names_dict

def query_token():
    token_url = 'https://api.pingan.com.cn/oauth/oauth2/access_token?client_id=P_GFACE_API&grant_type=client_credentials&client_secret=at15UCj6'
    res = requests.get(token_url).json()
    token = res['data']['access_token']
    return token

def get_error_code():
    with open(mood_error_path, 'r') as f:
        name_list = f.read().strip().split(',')
    return name_list

def update_mood_data(error_status=False):
    today_dt = datetime.now().date().strftime('%Y-%m-%d')
    origin = pd.read_csv(mood_path[0])
    origin_max_dt = max(origin.date)

    print('max date current: {}'.format(origin_max_dt))
    if origin_max_dt == today_dt:
        print('mood data already up-to-date. (single)')
        return 0    

    names, names_dict = names_info()
    token = query_token()
    api = ArchiveAPI(token, origin_max_dt,  today_dt)

    if error_status:
        print('error mode starting')
        name_list = get_error_code()
        names = [i for i in names if i in name_list]
        names_dict = {i[0]:i[1] for i in names_dict.items() if i[0] in names}
    
    miss_code=[]
    for name in names:
        try:
            res = api.query_event(name)
            dict_json = json.loads(res.decode('utf-8'))
            grades = [i['grade'] for i in dict_json['obj']]
            dates = [i['date'] for i in dict_json['obj']]
            mood = pd.DataFrame({'grades': grades, 'date': dates})
            mood_merge = mood['grades'].groupby(mood['date']).sum()
            mood_merge = pd.DataFrame(mood_merge).reset_index()
            mood_merge['code'] = names_dict[name]
            if len(mood_merge)>0:
                mood_merge.to_csv(os.path.join(mood_dir, '{}.csv'.format(names_dict[name])), index=False)
            print(name + 'done.')
        except:
            print(name + 'error.')
            miss_code.append(name)
            continue

    with open(mood_error_path, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(miss_code)
    
    if len(miss_code) == 0:
        print('correct, no error.')
        return 0
    # 若註解掉，error無法消除的話就會無限循環
    elif error_status:
        print('end')
        return 0
    else:
        update_mood_data(error_status=True)

def main_mood():
    update_mood_data()
    error_code = get_error_code()
    if len(error_code) >0:
        update_mood_data(error_status=True)

    today_dt = datetime.now().date().strftime('%Y-%m-%d')
    origin      = pd.read_csv(mood_path[0])
    origin_hist = pd.read_csv(mood_path[1])
    max_dt_origin      = max(origin.date)
    max_dt_origin_hist = max(origin_hist.date)

    if max_dt_origin_hist < max_dt_origin:
        origin.to_csv(mood_path[1], index=False)

    if max_dt_origin == today_dt:
        print('mood data already up-to-date. (append)')
        return 0
    
    files = os.listdir(mood_dir)
    files = [i for i in files if i not in ['.DS_Store', 'error_code.csv']]
    new = pd.DataFrame(columns=['date', 'grades', 'code'])

    # 只合并当天有更新的 mood data
    for f in files:
        filepath = os.path.join(mood_dir, f)
        last_edit_dt = datetime.fromtimestamp(os.path.getctime(filepath)).date()
        if last_edit_dt == datetime.now().date():
            df = pd.read_csv(filepath)
            new = pd.concat([new, df], axis=0, sort=False)
        
    new['grades'] = [round(i,4) for i in new.grades]
    new = pd.concat([origin, new], axis=0, sort=False)
    new = new.drop_duplicates().sort_values(by=['date'])
    new = new.groupby(['date', 'code'])['grades'].mean().reset_index()
    new.to_csv(mood_path[0], index=False)
    print('Update mood data Done.')

if __name__ == '__main__':
    main_mood()