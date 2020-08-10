'''
@Author: Gao S
@Date: 2020-06-20 13:35:36
@Description: 切割轨迹
@FilePath: /HUAWEI_competition/cut_trace.py
'''
from utils import haversine
from utils import portsUtils
from config import config

import pandas as pd
import numpy as np

class CutTrace(object):
    """用于切割轨迹

    """

    def __init__(self,
                 orders_ports_dict_filename=config.orders_ports_dict_filename,
                 orders_ports_dict=None):
        """初始化

        Args:
            orders_ports_dict_filename (str, optional): 港口路由表. Defaults to config.orders_ports_dict_filename.
            orders_ports_dict (dict, optional): 港口录音表字典. Defaults to None.
        """
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

    def get_use_indexs(self, start_port, end_port, match_start_end_port=False, line=True):
        """获取与start_port、end_port相关的轨迹的所有下标
        start_port优先匹配最靠前港，end_port优先匹配最靠后港

        Args:
            start_port (str): 起始港，映射后名称
            end_port (str): 终点港，映射后名称
            line (Bool): 是否直接返回平铺的全部索引值，如果False，则会返回对应的订单及子下标: [[xx, xx], [xx, xx], ...]
        Returns:
            (list): 两种，当line=True时，返回一列数据，元素为匹配到的绝对下标；当line=False时，返回n行2列，
                每行是匹配到的train集头尾坐标，第一列是头坐标，第二列是尾坐标
        """
        
        # TODO list为空

        use_indexs = []

        for key in self.orders_ports_dict.keys():
            if len(self.__orders_ports_name_dict[key]) < 2:
                continue
            if start_port in self.__orders_ports_name_dict[key] and end_port in self.__orders_ports_name_dict[key]:
                if match_start_end_port:
                    if (start_port != self.__orders_ports_name_dict[key][0] or 
                        end_port != self.__orders_ports_name_dict[key][-1]):
                        continue
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
                    start_first, start_second = self.orders_ports_dict[key][start_indexs[i]][1]
                    if start_first < 0:
                        i += 1
                        continue
                    end_first, end_second = self.orders_ports_dict[key][end_indexs[j]][1]
                    if end_second < 0 or end_first < 0:
                        j -= 1
                        continue
                    
                    if start_first >= end_second:
                        break
                    if end_first > 0 and start_first >= end_first:
                        break
                    if start_second > 0 and start_second >= end_second:
                        break
                    if end_first > 0 and start_second > 0 and start_second >= end_first:
                        break
                    
                    if match_start_end_port:
                        if start_first > 0 and end_first > 0:
                            start_index, end_index = start_first, end_first
                        elif start_first > 0 and end_second > 0:
                            start_index, end_index = start_first, end_second
                        elif start_second > 0 and end_first > 0:
                            start_index, end_index = start_second, end_first
                        elif start_second > 0 and end_second > 0:
                            start_index, end_index = start_second, end_second
                    else:
                        if start_second > 0 and end_first > 0:
                            start_index, end_index = start_second, end_first
                        elif start_second > 0 and end_second > 0:
                            start_index, end_index = start_second, end_second
                        elif start_first > 0 and end_first > 0:
                            start_index, end_index = start_first, end_first
                        elif start_first > 0 and end_second > 0:
                            start_index, end_index = start_first, end_second
                        

                    # start_index, end_index = start_first, end_second
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
        """得到匹配到的train集轨迹数，即loadingOrder数

        Args:
            start_port (str): 起始港，映射后名称
            end_port (str): 终点港，映射后名称

        Returns:
            (int): 长度，匹配到的train集轨迹数，即loadingOrder数
        """
        result =  self.get_use_indexs(start_port, end_port, line=False)
        return len(result)
    
    # ! 添加处理轨迹和test之trace数据
    def cut_trace_for_test(self, test_df, match_df, distance_threshold=80, for_start=False, for_parallel=True):
        """根据test的df(包含lon、lat数据，即轨迹)，对match_df进行切割
        该函数针对的是df数据，而不是轨迹数据(np.array)
        思路是将train中轨迹，从头从尾分别开始，计算每个点到test头尾点的距离，符合某个阈值时停止
        Args:
            test_df (pd.DataFrame): test的df，该df应该只包含一个订单，且已经排序，和match_df
            match_df (pd.DataFrame): 针对test匹配到的train的df数据
            distance_threshold (int, optional): 到首尾节点的距离阈值. Defaults to 80.
            for_start (Bool, optional): 切割起点港到轨迹头的路线. Defaults to False.
            for_parallel (Bool): 为True时使用多线程，为False时不使用多线程
        Returns:
            (pd.DataFrame): 切割后的df数据，index已经重设，可能为空，最后一条用切割前最后一条代替以求
                Label，倒数第二条为切割后train最后一个时间戳
        """
        # 得到test的首末点坐标
        test_start_lon, test_start_lat = test_df.loc[test_df.index[0]][['longitude', 'latitude']].tolist()
        test_end_lon, test_end_lat = test_df.loc[test_df.index[-1]][['longitude', 'latitude']].tolist()
        
        def while_for_cut_multi(df):
            # 用于循环迭代，依此增加距离阈值以匹配
            nonlocal distance_threshold
            distance_threshold_ = distance_threshold
            
            cutted_df = get_start_end_index_cut_for_test(df, threshold=distance_threshold_)[0]
            
            cut_multi = 2.0
            while len(cutted_df) == 0:
                if distance_threshold_ < 5:
                    if distance_threshold_ * cut_multi < 5:
                        cutted_df = get_start_end_index_cut_for_test(
                            df, threshold=distance_threshold_ * cut_multi)[0]
                        cut_multi += 1.0
                    else:
                        distance_threshold_ = 30
                        cut_multi = 1.0
                else:
                    if distance_threshold_ * cut_multi < 200:
                        cutted_df = get_start_end_index_cut_for_test(
                            df, threshold=distance_threshold_ * cut_multi)[0]
                        cut_multi += 1.0
                    else:
                        break
                    
            if len(cutted_df) != 0:
                return [cutted_df]
            else:
                return [pd.DataFrame(columns=df.columns)]
        
        def get_start_end_index_cut_for_test(df, threshold):
            # df : 训练集轨迹对应的df
            # 用于apply处理
            # 先处理从头开始的
            def limit_try(up_limit, try_i, start=False, end=False):
                nonlocal lon, lat, distance, i
                def try_dist(lon, lat, try_i):
                    try_lon, try_lat = df.loc[try_i][['longitude', 'latitude']].tolist()
                    return haversine(lon, lat, try_lon, try_lat)
                if distance > up_limit:
                    try:
                        if start == True:
                            if i + try_i < df.index[-1]:
                                if try_dist(lon, lat, i + try_i) < up_limit:
                                    return True
                        else:
                            if i - try_i > start_index:
                                if try_dist(lon, lat, i - try_i) < up_limit:
                                    return True
                    except:
                        return False
                return False

            start_index = -1
            i = df.index[0]
            while i < df.index[-1]:
            # for i in range(df.index[0], df.index[-1]):
                lon, lat = df.loc[i][['longitude', 'latitude']].tolist()
                distance = haversine(lon, lat, test_start_lon, test_start_lat)
                
                if distance <= threshold:
                    start_index = i
                    break
                # 用于加速

                if limit_try(3000,1000,start=True): i += 1000; continue;
                if limit_try(2000,700,start=True): i += 700; continue;
                if limit_try(2000,400,start=True): i += 400; continue;
                if limit_try(1000,200,start=True): i += 200; continue;
                if limit_try(1000,100,start=True): i += 100; continue;
                if limit_try(1000,50,start=True): i += 50; continue;
                if limit_try(1000,30,start=True): i += 30; continue;
                if limit_try(1000,20,start=True): i += 20; continue;
                if limit_try(200,10,start=True): i += 10; continue;
                if limit_try(200,5,start=True): i += 5; continue;
                if limit_try(200,2,start=True): i += 2; continue;

                if limit_try(threshold,20,start=True): i += 20; continue;
                if limit_try(threshold,10,start=True): i += 10; continue;
                if limit_try(threshold,5,start=True): i += 5; continue;
                if limit_try(threshold,2,start=True): i += 2; continue;
                i += 1
            if start_index < df.index[-1] - 1 or start_index != -1:
                pass
            else:
                start_index = -1
                # ! 结束
            end_index = -1
            if start_index != -1 and for_start == False:
                i = df.index[-1]
                while i > start_index:
                # for i in range(df.index[-1], start_index, -1):
                    lon, lat = df.loc[i][['longitude', 'latitude']].tolist()
                    distance = haversine(lon, lat, test_end_lon, test_end_lat)
                    if distance <= threshold:
                        end_index = i
                        break
                    
                    # 用于加速

                    if limit_try(3000,1000,end=True): i -= 1000; continue;
                    if limit_try(2000,700,end=True): i -= 700; continue;
                    if limit_try(2000,400,end=True): i -= 400; continue;
                    if limit_try(1000,200,end=True): i -= 200; continue;
                    if limit_try(1000,100,end=True): i -= 100; continue;
                    if limit_try(1000,50,end=True): i -= 50; continue;
                    if limit_try(1000,30,end=True): i -= 30; continue;
                    if limit_try(1000,20,end=True): i -= 20; continue;
                    if limit_try(200,10,end=True): i -= 10; continue;
                    if limit_try(200,5,end=True): i -= 5; continue;
                    if limit_try(200,2,end=True): i -= 2; continue;

                    if limit_try(threshold,20,end=True): i -= 20; continue;
                    if limit_try(threshold,10,end=True): i -= 10; continue;
                    if limit_try(threshold,5,end=True): i -= 5; continue;
                    if limit_try(threshold,2,end=True): i -= 2; continue;
                    i -= 1
                if end_index > df.index[0]+1 or end_index != -1 or end_index > start_index +1:
                    pass
                else:
                    end_index = -1
                # ! 结束
            
            # ! 打标问题
            if for_start == False:
                if start_index != -1 and end_index != -1:
                    use_df_label = df.loc[start_index:end_index]
                    # 最后一行数据的时间戳为对应train轨迹的到港时间戳
                    use_df_label.loc[end_index, 'timestamp'] = df.loc[df.index[-1], 'timestamp']
                    return [use_df_label]
                else:
                    return [pd.DataFrame(columns=df.columns)]
            else:
                if start_index != -1:
                    use_df_label = df.loc[df.index[0]: start_index]
                    # 最后一行数据的时间戳为对应train轨迹的到港时间戳
                    return [use_df_label]
                else:
                    return [pd.DataFrame(columns=df.columns)]
        
        if len(match_df) == 0:
            print('match_df为空')
            return pd.DataFrame(columns=match_df.columns)
        
        if for_parallel == False:
            use_df = match_df.groupby('loadingOrder').apply(
                lambda x:while_for_cut_multi(x)).tolist()
        else:
            use_df = match_df.groupby('loadingOrder').parallel_apply(
                lambda x:while_for_cut_multi(x)).tolist()
        
        use_df = list(map(lambda x: x[0], use_df))
        
        use_df_ = pd.DataFrame()
        use_df_ = use_df_.append(use_df, ignore_index=True)
        
        use_df_ = use_df_.reset_index(drop=True)

        return use_df_
    
    # ! 该部分有大量错误需要修改！
    # def cut_traj_for_test(self, test_traj, match_traj, distance_threshold=80, for_traj=True):
    #     """对cut_trace_for_test函数的包装，用于处理traj数据(np.array格式)

    #     Args:
    #         test_traj (np.array): [description]
    #         match_traj (list(np.array)): [description]
    #         distance_threshold (int, optional): [description]. Defaults to 80.
            
    #     Returns:
    #         match_df: 切割后的df数据，index已经重设
    #     """
    #     # 对cut_trace_for_test函数的包装，用于处理traj数据(np.array格式)
    #     # df需要三列：loadingOrder、longitude、latitude
    #     # 先构造test_df
    #     test_df = pd.DataFrame(test_traj)
    #     test_df.columns=['longitude', 'latitude']
    #     test_df['loadingOrder'] = 'XXXXX'
        
    #     # 构造match_df
    #     match_df = pd.DataFrame()
    #     for i in range(len(match_traj)):
    #         traj = match_traj[i]
            
    #         traj_df = pd.DataFrame(traj)
    #         traj_df['loadingOrder'] = 'XX_' + str(i)
    #         traj_df['timestamp'] = 'XXXXX'
            
    #         match_df = match_df.append(traj_df, ignore_index=True)
    #     match_df.columns=['longitude', 'latitude', 'loadingOrder', 'timestamp']
        
    #     cutted_df = self.cut_trace_for_test(test_df, match_df, distance_threshold, for_traj=True)
    #     # ! 遇到空DataFrame问题
    #     if len(cutted_df) != 0:
    #         cutted_traj = cutted_df.groupby('loadingOrder')[['longitude', 'latitude']].apply(lambda x: [x.values])
    #         cutted_traj = list(map(lambda x: x[0], cutted_traj.tolist()))
    #     else:
    #         cutted_traj = []

    #     return cutted_traj
        
cutTrace = CutTrace()


if __name__ == '__main__':
    # CNYTN-MXZLO
    start_port, _ = portsUtils.get_mapped_port_name('CNYTN')
    end_port, _ = portsUtils.get_mapped_port_name('MXZLO')
    
    related_indexs = cutTrace.get_use_indexs(start_port, end_port, line=True)
    
    # print(related_indexs[:10])