'''
@Author: Gao S
@Date: 2020-06-20 13:35:36
@LastEditTime: 2020-06-23 00:50:54
@Description: 切割轨迹
@FilePath: /HUAWEI_competition/cut_trace.py
'''
from utils import haversine
from utils import portsUtils
from config import config

import pandas as pd

class CutTrace(object):
    """用于切割轨迹

    Args:

    """

    def __init__(self,
                 orders_ports_dict_filename=config.orders_ports_dict_filename,
                 orders_ports_dict=None):
        super().__init__()
        if orders_ports_dict is None:
            with open(orders_ports_dict_filename, 'r') as f:
                self.orders_ports_dict = eval(f.read())
        else:
            self.orders_ports_dict = orders_ports_dict

        # 用于快速匹配，只保留名字，正向反向分别保存
        self.__orders_ports_name_dict = {}
        for key in self.orders_ports_dict.keys():
            if len(self.orders_ports_dict[key]) == 0:
                self.__orders_ports_name_dict[key] = []
            else:
                self.__orders_ports_name_dict[key] = [
                    k[0] for k in self.orders_ports_dict[key]]  # 港口名list

    def get_use_indexs(self, start_port, end_port, line=True):
        """获取与start_port、end_port相关的轨迹的所有下标
        start_port优先匹配最靠前港，end_port优先匹配最靠后港

        Args:
            start_port (str): 起始港，映射后名称
            end_port (str): 终点港，映射后名称
            line (Bool): 是否直接返回平铺的全部索引值，如果False，则会返回对应的订单及子下标: [[xx, xx], [xx, xx], ...]
        """

        # TODO list为空

        use_indexs = []

        for key in self.orders_ports_dict.keys():
            if len(self.__orders_ports_name_dict[key]) < 2:
                continue
            if start_port in self.__orders_ports_name_dict[key] and end_port in self.__orders_ports_name_dict[key]:
                # 起止港均在内
                start_indexs = [i for i in range(len(
                    self.__orders_ports_name_dict[key])) if self.__orders_ports_name_dict[key][i] == start_port]
                end_indexs = [i for i in range(len(
                    self.__orders_ports_name_dict[key])) if self.__orders_ports_name_dict[key][i] == end_port]

                if start_indexs[0] >= end_indexs[-1]:
                    continue

                start_index, end_index = -1, -1
                i, j = 0, len(end_indexs)-1
                while i < len(start_indexs) and j >= 0 and start_indexs[i] < end_indexs[j]:
                    start_first, _ = self.orders_ports_dict[key][start_indexs[i]][1]
                    if start_first < 0:
                        i += 1
                        continue
                    end_first, end_second = self.orders_ports_dict[key][end_indexs[i]][1]
                    if end_second < 0:
                        j -= 1
                        continue
                    if start_first >= end_second:
                        break
                    
                    if end_first > 0:
                        start_index, end_index = start_first, end_first
                    else:
                        start_index, end_index = start_first, end_second

                    start_index, end_index = start_first, end_second
                    break

                if start_index == -1 or end_index == -1:
                    continue
                
                if line == True:
                    use_indexs.append([start_index, end_index])
                else:
                    use_indexs.append([key, start_index, end_index])
            else:
                continue
            
        if line == True:
            use_indexs_ = []
            for row in use_indexs:
                if len(row) != 0:
                    use_indexs_ += list(range(row[0], row[1]+1))
            return use_indexs_
        else:
            return use_indexs
        
    def get_use_indexs_len(self, start_port, end_port):
        result =  self.get_use_indexs(start_port, end_port, line=False)
        return len(result)
    
    def __get_start_end_index_cut_for_test(self, df, test_lonlat, distance_threshold):
        # 用于apply处理
        # 先处理从头开始的
        test_start_lon, test_start_lat  = test_lonlat[0], test_lonlat[1]
        test_end_lon, test_end_lat = test_lonlat[2], test_lonlat[3]
        
        start_index = -1
        for i in range(df.index[0], df.index[-1]):
            lon, lat = df.loc[i][['longitude', 'latitude']].tolist()
            distance = haversine(lon, lat, test_start_lon, test_start_lat)
            if distance <= distance_threshold:
                start_index = i
                break
        if start_index != df.index[-1] - 1 or start_index != -1:
            pass
        else:
            start_index = -1
            # ! 结束
        
        end_index = -1
        if start_index != -1:
            for i in range(df.index[-1], start_index, -1):
                lon, lat = df.loc[i][['longitude', 'latitude']].tolist()
                distance = haversine(lon, lat, test_end_lon, test_end_lat)
                if distance <= distance_threshold:
                    end_index = i
                    break
            if end_index != df.index[0]+1 or end_index != -1 or end_index != start_index +1:
                pass
            else:
                end_index = -1
            # ! 结束
        
        if start_index != -1 and end_index != -1:
            return [start_index, end_index]
        else:
            return []
    
    def cut_trace_for_test(self, test_df, match_df, distance_threshold=80):
        """根据test的df(包含lon、lat数据，即轨迹)，对match_df进行切割
        该函数针对的是df数据，而不是轨迹数据(np.array)
        思路是将train中轨迹，从头从尾分别开始，计算每个点到test头尾点的距离，符合某个阈值时停止
        Args:
            test_df ([type]): test的df，该df应该只包含一个订单，且已经排序，和match_df
            match_df ([type]): 针对test匹配到的train的df数据
            distance_threshold (int, optional): 到首尾节点的距离阈值. Defaults to 80.

        Returns:
            match_df: 切割后的df数据，index已经重设
        """
        # 得到test的首末点坐标
        test_start_lon, test_start_lat = test_df.loc[test_df.index[0]][['longitude', 'latitude']].tolist()
        test_end_lon, test_end_lat = test_df.loc[test_df.index[0]][['longitude', 'latitude']].tolist()
        
        
            
        use_indexs = match_df.groupby('loadingOrder')['longitude', 'latitude'].parallel_apply(
            lambda x: self.__get_start_end_index_cut_for_test(
                x, [test_start_lon, test_start_lat, test_end_lon, test_end_lat], distance_threshold))
        use_indexs = use_indexs.tolist()
        
        use_indexs_ = []
        for row in use_indexs:
            if len(row) != 0:
                use_indexs_ += list(range(row[0], row[1]))
        
        match_df = match_df.loc[use_indexs_]
        match_df = match_df.reset_index(drop=True)

        return match_df
    
    def cut_traj_for_test(self, test_traj, match_traj, distance_threshold=80):
        """对cut_trace_for_test函数的包装，用于处理traj数据(np.array格式)

        Args:
            test_traj (np.array): [description]
            match_traj (list(np.array)): [description]
            distance_threshold (int, optional): [description]. Defaults to 80.
            
        Returns:
            match_df: 切割后的df数据，index已经重设
        """
        # 对cut_trace_for_test函数的包装，用于处理traj数据(np.array格式)
        # df需要三列：loadingOrder、longitude、latitude
        # 先构造test_df
        test_df = pd.DataFrame(test_traj)
        test_df.columns=['longitude', 'latitude']
        test_df['loadingOrder'] = 'XXXXX'
        
        # 构造match_df
        match_df = pd.DataFrame()
        for i in range(len(match_traj)):
            traj = match_traj[i]
            
            traj_df = pd.DataFrame(traj)
            traj_df['loadingOrder'] = 'XX_' + str(i)
            
            match_df = match_df.append(traj_df, ignore_index=True)
        match_df.columns=['longitude', 'latitude', 'loadingOrder']
        
        cutted_df = self.cut_trace_for_test(test_df, match_df, distance_threshold)
        
        cutted_traj = cutted_df.groupby('loadingOrder')[['longitude', 'latitude']].apply(lambda x: [x.values])
        cutted_traj = list(map(lambda x: x[0], cutted_traj.tolist()))

        return cutted_traj
        
cutTrace = CutTrace()


if __name__ == '__main__':
    # CNYTN-MXZLO
    start_port, _ = portsUtils.get_mapped_port_name('CNYTN')
    end_port, _ = portsUtils.get_mapped_port_name('MXZLO')
    
    related_indexs = cutTrace.get_use_indexs(start_port, end_port, line=True)
    
    # print(related_indexs[:10])