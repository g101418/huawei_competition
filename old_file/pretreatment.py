'''
@Author: Gao S
@Date: 2020-07-14 10:13:17
@LastEditTime: 2020-07-15 16:45:50
@Description: 
@FilePath: /HUAWEI_competition/pretreatment.py
'''
from data_analyse_util import dataAnalyseUtil
from delete_drift import driftPoint
from find_all_ports_on_trace import findPorts

from config import config

import pandas as pd
from pandarallel import pandarallel

class Pretreatment(object):
    def __init__(self):
        super().__init__()
    
    # TODO PipeLine
    # TODO test集预处理：trace换名
    
    def pipeline(self, stage, train_data=None, test_data=None):
        
        pandarallel.initialize(nb_workers=config.nb_workers)
        
        # 暂时实现漂移处理和去重
        if train_data is None:
            train_data = pd.read_csv(config.train_gps_path, header=None)
            train_data.columns = config.train_data_columns
            
        if test_data is None:
            test_data = pd.read_csv(config.test_data_path)
        # 0. 手工调整港口信息：a) port_points.txt b) test集数据trace别名替换
        
        # 1. 对test集排序
        if 1 in stage:
            test_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
            test_data = test_data.reset_index(drop=True)
            
            # 写回
            test_data.to_csv(config.test_data_path, index=False)
        
        # 1.2 对test集去除漂移点
        
        # 2. 去重
        if 2 in stage:
            train_data = dataAnalyseUtil.get_drop_duplicated_data(train_data)
        
        # 2.2 删除方向为-1/小于0 或 大于36000的数据
        
        # 3. 删除漂移点
        if 3 in stage:
            train_data = driftPoint.delete_drift_point(train_data, 50)
        
            # 写回
            train_data.to_csv(config.train_data_drift, index=False)
        
        # 4. 找到途径港口
        if 4 in stage:
            orders_ports_dict = findPorts.find_ports(train_data)
            
            # 写回
            with open(config.tool_file_dir_path + 'orders_ports_dict.txt', 'w') as f:
                f.write(str(orders_ports_dict))
        
        # 5. 匹配轨迹并得到label
        if 5 in stage:
            pass