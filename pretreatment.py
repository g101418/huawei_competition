'''
@Author: Gao S
@Date: 2020-07-14 10:13:17
@LastEditTime: 2020-07-14 12:49:10
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
    
    def pipeline(self, stage, train_data=None):
        
        pandarallel.initialize(nb_workers=config.nb_workers)
        
        # 暂时实现漂移处理和去重
        if train_data is None:
            train_data = pd.read_csv(config.train_gps_path, header=None)
            train_data.columns = config.train_data_columns
        # 1. 手工调整港口信息：a) port_points.txt b) test集数据trace别名替换
        if 1 in stage:
            pass
        
        # 2. 去重
        if 2 in stage:
            train_data = dataAnalyseUtil.get_drop_duplicated_data(train_data)
        
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