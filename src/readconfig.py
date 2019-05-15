# -*_ coding: utf-8 -*-
"""
Created on May 5 15:36:10 2019
@author: Shao Xuan Huang
@email : HUANGSHOAXUAN801@pingan.com.cn

讀取 config 類提供其他 .py檔使用
"""

import os
import yaml

class ReadConfig(object):
    def __init__(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            self.config = yaml.load(f)
        self.basic_stock_path = self.config['path']['daily_stock_dir']
        self.all_stock_path   = self.config['base']['all_mkt_code']
        self.mkt_codelist     = self.config['base']['mkt_index']
        self.mkt_path         = self.config['raw_update']['daily_mkt']
        self.ola_mood_path    = self.config['raw_update']['daily_ola']

    def get_folder_path(self, params):
        value = self.config['path'][params]
        return value

    def get_values(self):
        model_folder = self.model_folder
        quarter_folder = self.quarter_folder
        print(model_folder)
        print(quarter_folder)

if __name__ == '__main__':
    path = './config_operation.yml'
    test = ReadConfig(path)
    print(test.basic_stock_path)