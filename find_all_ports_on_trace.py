'''
@Author: Gao S
@Date: 2020-06-19 19:08:19
@LastEditTime: 2020-07-14 19:30:27
@Description: 找到航线经过的所有港口
@FilePath: /HUAWEI_competition/find_all_ports_on_trace.py
'''
from pandarallel import pandarallel
import geohash
import itertools

import pandas as pd
import numpy as np

from config import config
from utils import haversine, portsUtils

import warnings
warnings.filterwarnings('ignore')


class FindPorts(object):
    def __init__(self, train_data=None, distance_threshold=25, speed_threshold=2):
        super().__init__()
        self.train_data = train_data
        self.distance_threshold = distance_threshold
        self.speed_threshold = speed_threshold

    def find_all_ports_from_order(self, df):
        """
        用于发现轨迹中所有经过的港口
        ports包含了所有港口及其对应的下标(绝对下标/相对下标？)如(A:B)

        第二个坐标B为-1时，说明短暂停留
        """
        ports = []
        next_time = None
        next_lon = None
        next_lat = None

        last_in_port_state = False
        cur_in_port_state = False  # 跳变记录
        last_port_end_index = -1
        final_port_end_index = -1
        for i in range(df.index[0], df.index[-1]):

            if next_time is not None:
                cur_lon = next_lon
                cur_lat = next_lat
                cur_time = next_time
            else:
                cur_lon = df.loc[i]['longitude']
                cur_lat = df.loc[i]['latitude']
                cur_time = pd.to_datetime(df.loc[i, 'timestamp'])

            next_lon = df.loc[i+1]['longitude']
            next_lat = df.loc[i+1]['latitude']
            next_time = pd.to_datetime(df.loc[i+1, 'timestamp'])

            time_delta_hour = ((next_time-cur_time).total_seconds() / 3600)
            if time_delta_hour < 0.00027778:  # 1秒
                continue
            distance = haversine(next_lon, next_lat, cur_lon, cur_lat)
            speed = distance / time_delta_hour

            last_in_port_state = cur_in_port_state

            if speed < self.speed_threshold:
                port = portsUtils.get_port(
                    cur_lon, cur_lat, distance_threshold=self.distance_threshold)[0]
                if port is not None:
                    cur_in_port_state = True
                    if len(ports) == 0:
                        ports.append([port, [i, -1]])
                    elif ports[-1][0] == port:  # 还在该港口内
                        last_port_end_index = i
                    else:  # 别的港口
                        ports[-1][1][1] = - \
                            1 if last_port_end_index < ports[-1][1][0] else last_port_end_index
                        ports.append([port, [i, -1]])
                else:
                    cur_in_port_state = False
            else:
                cur_in_port_state = False

            if last_in_port_state == True and cur_in_port_state == False:
                final_port_end_index = i

        # ! 此处修改了，但不敢确保无事
        if len(ports) != 0 and ports[-1][1][1] == -1:
            if cur_in_port_state == True:
                ports[-1][1][1] = i + 1
            else:
                ports[-1][1][1] = final_port_end_index if final_port_end_index != -1 else i + 1

        # 不考虑最后刚刚到港情况

        return ports

    def insert_start_end_port(self, df):
        """find_all_ports_from_order函数不能很好地处理
        头结点第二个坐标-2，尾节点第一个坐标-2

        Args:
            df ([type]): [description]
        """
        order = df.loc[df.index[0]]['loadingOrder']

        start_index, end_index = df.index[0], df.index[-1]

        start_lon = df.loc[start_index]['longitude']
        start_lat = df.loc[start_index]['latitude']

        end_lon = df.loc[end_index]['longitude']
        end_lat = df.loc[end_index]['latitude']

        start_port = portsUtils.get_port(
            start_lon, start_lat, distance_threshold=self.distance_threshold)[0]
        end_port = portsUtils.get_port(
            end_lon, end_lat, distance_threshold=self.distance_threshold)[0]

        result = []
        if start_port is not None:
            if len(self.orders_ports_dict[order]) != 0:
                if start_port != self.orders_ports_dict[order][0][0]:
                    result.append([start_port, [start_index, -2]])
            else:
                result.append([start_port, [start_index, -2]])

        if end_port is not None:
            if len(self.orders_ports_dict[order]) != 0:
                if end_port != self.orders_ports_dict[order][-1][0]:
                    result.append([end_port, [-2, end_index]])
            else:
                result.append([end_port, [-2, end_index]])

        return result

    def integration_port(self):
        for key in self.orders_ports_dict.keys():
            if len(self.start_end_ports_dict[key]) == 0:
                continue
            if len(self.start_end_ports_dict[key]) == 1:
                port, index = self.start_end_ports_dict[key][0]
                if index[0] == -2:
                    self.orders_ports_dict[key].append(
                        self.start_end_ports_dict[key][0])
                else:
                    self.orders_ports_dict[key].insert(
                        0, self.start_end_ports_dict[key][0])
            if len(self.start_end_ports_dict[key]) == 2:
                port0, index0 = self.start_end_ports_dict[key][0]
                port1, index1 = self.start_end_ports_dict[key][1]

                if index0[1] == -2:
                    self.orders_ports_dict[key].insert(
                        0, self.start_end_ports_dict[key][0])
                    self.orders_ports_dict[key].append(
                        self.start_end_ports_dict[key][1])
                else:
                    self.orders_ports_dict[key].insert(
                        0, self.start_end_ports_dict[key][1])
                    self.orders_ports_dict[key].append(
                        self.start_end_ports_dict[key][0])
    
    def delete_not_0_speed(self):
        
        orders_ports_dict_ = dict()
        
        for key, value in self.orders_ports_dict.items():
            ports = [item[0] for item in value]
            
            if len(value) == 0:
                orders_ports_dict_[key] = []
                continue
                
            value_ = []
            for port in value:
                port_name = port[0]
                start_index = port[1][0]
                end_index = port[1][1]
                
                if start_index < 0 or end_index < 0:
                    value_.append([port_name, [start_index, end_index]])
                    continue
                
                port_df = train_data.loc[start_index: end_index]
                
                port_df_speed_not_0 = port_df[port_df['speed'] <= 1]
                
                
                if len(port_df_speed_not_0) != 0:
                    value_.append([port_name, [port_df_speed_not_0.index[0], port_df_speed_not_0.index[-1]]])
            
            orders_ports_dict_[key] = value_  
              
        self.orders_ports_dict = orders_ports_dict_

    def find_ports(self, train_data=None, ordername=None):
        if train_data is None:
            train_data = self.train_data

        if ordername is not None:
            train_data = train_data[train_data['loadingOrder'] == ordername]

        orders_ports = train_data[:].groupby('loadingOrder')[[
            'timestamp', 'longitude', 'latitude']].parallel_apply(self.find_all_ports_from_order)
        self.orders_ports_dict = orders_ports.to_dict()

        start_end_ports = train_data[:].groupby('loadingOrder')[[
            'loadingOrder', 'timestamp', 'longitude', 'latitude']].parallel_apply(
                self.insert_start_end_port)
        self.start_end_ports_dict = start_end_ports.to_dict()

        self.integration_port()
        
        self.delete_not_0_speed()

        return self.orders_ports_dict
    
findPorts = FindPorts()

if __name__ == '__main__':
    train_data = pd.read_csv(config.train_data_drift_dup)

    # progress_bar=True
    pandarallel.initialize(nb_workers=config.nb_workers)

    
    orders_ports_dict = findPorts.find_ports(train_data)
    
    with open(config.tool_file_dir_path + 'orders_ports_dict_0714.txt', 'w') as f:
        f.write(str(orders_ports_dict))
