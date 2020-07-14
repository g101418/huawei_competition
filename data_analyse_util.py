# -*- coding:utf-8 -*-
# @Time: 2020/7/14 10:43
# @Author: beilwu
# @Email: beilwu@seu.edu.cn
# @File: data_analyse_util.py

import pandas as pd


class DataAnalyseUtil(object):

    # 包含各种用于分析处理数据的工具
    # def __init__(self):


    # 用于获取去重后还剩余的运单号
    def get_drop_duplicated_order_list(self, file_path='train0711.csv', data_rows=0):
        if data_rows > 0:
            train_data = pd.read_csv(file_path, nrows=data_rows, header=None)
        else:
            train_data = pd.read_csv(file_path, header=None)

        # 需要给train_data加上列名
        train_data.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                              'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                              'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']

        temp_train_data = train_data.drop_duplicates(subset=['vesselMMSI', 'timestamp'], keep='first', inplace=False)
        temp_train_data_order_list = temp_train_data.loadingOrder.unique()
        return temp_train_data_order_list

    # 根据剩余的运单号从原文件截取出新的数据集
    def get_drop_duplicated_data(self, file_path):
        data_df = pd.read_csv(file_path, header=None)
        data_df.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                    'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                    'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
        temp_train_data_order_list = self.get_drop_duplicated_order_list(file_path)
        new_train_data = data_df.loc[data_df['loadingOrder'].isin(temp_train_data_order_list)]
        return new_train_data


dataAnalyseUtil = DataAnalyseUtil()

if __name__ == '__main__':

    # order_list = dataAnalyseUtil.get_drop_duplicated_order_list('train_data_50000.csv')
    data_frame = dataAnalyseUtil.get_drop_duplicated_data('train_data_50000.csv')
    print(data_frame)
    print(data_frame.shape[0])