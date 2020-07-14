'''
@Author: Gao S
@Date: 2020-07-14 12:12:42
@LastEditTime: 2020-07-14 16:10:02
@Description: 
@FilePath: /HUAWEI_competition/data_analyse_util.py
'''
# -*- coding:utf-8 -*-
# @Time: 2020/7/14 10:43
# @Author: beilwu
# @Email: beilwu@seu.edu.cn
# @File: data_analyse_util.py

import pandas as pd
from config import config


class DataAnalyseUtil(object):

    # 包含各种用于分析处理数据的工具
    # def __init__(self):


    # 用于获取去重后还剩余的运单号
    def get_drop_duplicated_order_list(self, train_data):
        temp_train_data = train_data.drop_duplicates(subset=['vesselMMSI', 'timestamp'], keep='first', inplace=False)
        temp_train_data_order_list = temp_train_data.loadingOrder.unique().tolist()
        return temp_train_data_order_list

    # 根据剩余的运单号从原文件截取出新的数据集
    def get_drop_duplicated_data(self, train_data):
        temp_train_data_order_list = self.get_drop_duplicated_order_list(train_data)
        new_train_data = train_data.loc[train_data['loadingOrder'].isin(temp_train_data_order_list)]
        new_train_data = new_train_data.reset_index(drop=True)
        new_train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
        return new_train_data


dataAnalyseUtil = DataAnalyseUtil()

if __name__ == '__main__':
    train_data = pd.read_csv(config.train_gps_path, header=None)
    train_data.columns = config.train_data_columns
    
    train_data_dup = dataAnalyseUtil.get_drop_duplicated_data(train_data)
    
    train_data_dup.to_csv(config.train_data_dup, index=False)