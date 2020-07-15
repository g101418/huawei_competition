'''
@Author: Gao S
@Date: 2020-06-20 18:09:10
@LastEditTime: 2020-07-15 10:25:02
@Description: 
@FilePath: /HUAWEI_competition/trajectory_matching.py
'''
from cut_trace import CutTrace
from utils import portsUtils
from config import config

import numpy as np
import pandas as pd
import traj_dist.distance as tdist
import geohash
import itertools

from pandarallel import pandarallel

import heapq

import warnings
warnings.filterwarnings('ignore')


class TrajectoryMatching(object):
    """轨迹匹配，用于得到匹配后轨迹相关label等

    """
    def __init__(self, 
                 train_data, 
                 geohash_precision=4, 
                 metric="sspd",
                 cut_distance_threshold=-1,
                 mean_label_num=1,
                 vessel_name=''):
        """初始化

        Args:
            train_data (pd.DataFrame): 训练集，完整的
            geohash_precision (int, optional): 利用geohash压缩轨迹缓解GPS点密度不同问题. Defaults to 4.
            metric (str, optional): 轨迹相似度(距离)算法. Defaults to "sspd".
            cut_distance_threshold (int, optional): 轨迹前后收缩切割的距离阈值. Defaults to -1.
            mean_label_num (int, optional): 匹配轨迹后求平均的轨迹数. Defaults to 1.
            vessel_name (str, optional): 只有两种：'carrierName'或'vesselMMSI'. Defaults to ''.
        """
        super().__init__()
        self.train_data = train_data
        self.__cutTrace = CutTrace()
        self.match_traj_dict = {}
        self.match_df_dict = {}
        self.__geohash_precision = geohash_precision
        self.__metric = metric
        self.cut_distance_threshold = cut_distance_threshold
        self.__mean_label_num = mean_label_num
        if vessel_name in ['carrierName', 'vesselMMSI']:
            self.__vessel_name = vessel_name
        else:
            self.__vessel_name = ''

    def __get_traj_order_label(self, start_port, end_port):
        """按照起止港得到相关训练集
        确保数据集已经排序。如果没有起止港相关数据，则返回 None, None, None
        Args:
            start_port (str): 起始港名称
            end_port (str): 终止港名称

        Returns:
            order_list, label_list, traj_list (list, list, list(np.array)): 
                返回与起止港有关的(可能切断)航线的order、label(时间段)、轨迹
        """
        # 得到相关训练集
        match_df = self.__get_match_df(start_port, end_port, reset_index=False)
        if match_df is None:
            return None, None, None

        order_list = match_df.loadingOrder.unique().tolist()

        traj_list_label_series = match_df.groupby('loadingOrder')[[
            'timestamp', 'longitude', 'latitude']].apply(lambda x: self.__get_traj_list_label(x))
        traj_list_label_series = np.array(traj_list_label_series.tolist())

        label_list = list(traj_list_label_series[:, 1])

        traj_list = list(traj_list_label_series[:, 0])
        traj_list = list(map(lambda x: np.array(x), traj_list))

        return order_list, label_list, traj_list

    def __get_match_df(self, start_port, end_port, reset_index=True):
        """得到与trace相关的训练集，训练集可选是否排序
        如果没有相关df，则返回None
        Args:
            start_port (str): 起始港
            end_port (str): 终点港
            reset_index (Bool): 选择是否返回index重排的df，默认重排序

        Returns:
            match_df (pd.DataFrame): 与trace相关的训练集，可选择是否排序
        """
        
        # TODO 此处考虑增加相近港口
        # TODO 初步：考虑结果为空者
        # TODO 中级：考虑将不同起止点进行融合，考虑order重合现象
        result = self.__cutTrace.get_use_indexs(start_port, end_port)
        
        if len(result) == 0:
            start_port_near_names = portsUtils.get_near_name(start_port)
            end_port_near_names = portsUtils.get_near_name(end_port)
            
            if len(start_port_near_names) == 1 and len(end_port_near_names) == 1:
                return None
            
            near_name_pairs = [(i,j) for i in start_port_near_names for j in end_port_near_names]
            
            if len(near_name_pairs) == 0:
                return None
            
            results = []
            for item in near_name_pairs:
                result = self.__cutTrace.get_use_indexs(item[0], item[1])
                results.append((result, len(result)))
            results.sort(key=lambda x: x[1], reverse=True)
            result = results[0][0]

        if len(result) == 0:
            return None

        # 得到相关训练集
        match_df = self.train_data.loc[result]
        if reset_index == True:
            match_df = match_df.reset_index(drop=True)

        return match_df

    def __get_label(self, df, for_traj=True):
        """得到标签列，即时间差

        Args:
            df (pd.DataFrame): 按group划分后的数据集
            for_traj (Bool): 如果是True，则label为train全程时间，如
                果是False，则label为剪切后最后的时间戳到到港时间
        Returns:
            label (): 时间差
        """
        if for_traj == True:
            first_time = pd.to_datetime(df.loc[df.index[0], 'timestamp'])
        else:
            first_time = pd.to_datetime(df.loc[df.index[-2], 'timestamp'])
            
        final_time = pd.to_datetime(df.loc[df.index[-1], 'timestamp'])
        label = final_time - first_time

        return label

    def __get_traj_list(self, df):
        """得到轨迹list

        Args:
            df (pd.DataFrame): 按group划分后的数据集

        Returns:
            traj_list (np.array): n行2列的GPS轨迹列表
        """
        lon = df['longitude'].tolist()
        lat = df['latitude'].tolist()
        traj_list = list(map(list, zip(lon, lat)))

        traj_list = list(map(lambda x: geohash.encode(
            x[1], x[0], precision=self.__geohash_precision), traj_list))
        traj_list = [k for k, g in itertools.groupby(traj_list)]
        traj_list = list(
            map(lambda x: [geohash.decode(x)[1], geohash.decode(x)[0]], traj_list))

        return traj_list

    def __get_traj_list_label(self, df, for_traj=True):
        """得到轨迹列表和标签

        Args:
            df (pd.DataFrame): 按group划分后的数据集
            for_traj (Bool): 如果是True，则label为train全程时间，如
                果是False，则label为剪切后最后的时间戳到到港时间
        Returns:
            traj_list, label (np.array, ): 轨迹列表(n行2列的GPS轨迹列表)，标签
        """
        if for_traj == True:
            label = self.__get_label(df)
        else:
            label = self.__get_label(df, for_traj=False)
        
        traj_list = self.__get_traj_list(df)

        return [traj_list, label]

    def __get_trace(self, data_series):
        """得到trace

        Args:
            data_series (pd.Series): 按group划分后的trace列

        Returns:
            trace (list): 1行2列的trace列表，第一列是开始港、第二列是结束港
        """
        trace = data_series.unique()[0]
        trace = trace.replace(' ', '')
        trace = trace.split('-')
        
        trace_ = [None, None]
        
        trace_[0] = portsUtils.get_alias_name(trace[0])
        trace_[1] = portsUtils.get_alias_name(trace[-1])

        return trace_

    def get_test_trace(self, test_data):
        """对测试集进行处理
        将测试集分为三列，分别是订单、trace、轨迹
        Args:
            test_data (pd.DataFrame): 测试集df

        Returns:
            order_list, trace_list, traj_list (list, list, list(np.array)): 
                返回订单、trace、轨迹
        """
        order_list = test_data.loadingOrder.unique().tolist()

        traj_list = test_data.groupby('loadingOrder')[[
            'longitude', 'latitude']].apply(self.__get_traj_list)
        traj_list = traj_list.tolist()
        traj_list = list(map(lambda x: np.array(x), traj_list))

        trace_list = test_data.groupby('loadingOrder')[
            'TRANSPORT_TRACE'].apply(self.__get_trace).tolist()

        return order_list, trace_list, traj_list

    def get_related_traj_len(self, start_port, end_port):
        """得到trace相关轨迹的数量

        Args:
            start_port (str): 起始港名称
            end_port (str): 终止港名称

        Returns:
            [int]: 相关轨迹的数量
        """
        start_port = portsUtils.get_alias_name(start_port)
        end_port = portsUtils.get_alias_name(end_port)
        
        result = self.get_related_traj(start_port, end_port)[0]
        if result is None:
            return 0
        else:
            return len(result)
        
    def get_related_df_len(self, start_port, end_port):
        """得到trace相关df中订单的数量

        Args:
            start_port (str): 起始港名称
            end_port (str): 终止港名称

        Returns:
            [int]: 相关df中订单的数量
        """
        
        result = self.get_related_df(start_port, end_port)
        if result is None:
            return 0
        return result.loadingOrder.nunique()

    def get_final_label(self, order, trace, traj, test_data, for_parallel=False):
        """输入某一订单的订单名称、trace、轨迹，得到最相似轨迹的label并返回
        在调用此函数前，应该确认该trace存在相关轨迹！
        Args:
            order (str): 订单名
            trace (list): 1行2列list，分别是开始港、结束港
            traj (np.array): test的轨迹
            test_data (pd.DataFrame): test集df，单个订单
            for_parallel (bool, optional): 用于处理单条航线时并行化. Defaults to False.

        Returns:
            order, label (str, pd.Timedelta) :订单名称、时间差
        """
        try:
            if self.cut_distance_threshold < 0:
                # TODO 是否为作废代码
                train_order_list, train_label_list, train_traj_list = self.get_related_traj(
                    trace[0], trace[1])

                if train_label_list is None:
                    return None, None
            else:
                train_order_list, train_label_list, train_traj_list = self.modify_traj_label(test_data, for_parallel=for_parallel)
                if train_label_list is None or len(train_label_list) == 0:
                    return None, None
        except:
            print('error:', order, 'modify_traj_label')
            
        try:
            cdist = list(tdist.cdist(
                [traj], train_traj_list, metric=self.__metric)[0])
            min_traj_index = cdist.index(min(cdist))
        except:
            print('error:', order, 'tdist.cdist')
            return None,None
        
        try:
            min_traj_index_3 = list(map(lambda x:cdist.index(x), 
                heapq.nsmallest(min(len(train_label_list), self.__mean_label_num), cdist)))
            
            mean_label_seconds = np.mean(list(map(lambda x: x.total_seconds(), 
                list(np.array(train_label_list)[min_traj_index_3]))))

            mean_label = pd.Timedelta(seconds=mean_label_seconds)
        except:
            print('求取平均值错误:', order)
            
        return train_order_list[min_traj_index], mean_label

    def parallel_get_label(self, df, test_data, for_parallel=False):
        """用于并行化处理，得到order及label

        Args:
            df (pd.DataFrame): test集中相关数据构造的数据集的某一行
            for_parallel (bool, optional): 用于处理单条航线时并行化. Defaults to False.

        Returns:
            [result_order, result_label] ([str, ]): 返回order及对应的label列表
        """
        order = df.loc[df.index[0], 'loadingOrder']
        trace = df.loc[df.index[0], 'trace']
        traj = df.loc[df.index[0], 'traj']

        test_data_ = test_data[test_data['loadingOrder']==order]
        
        result_order, result_label = self.get_final_label(order, trace, traj, test_data_, for_parallel=for_parallel)
        return [order, result_order, result_label]

    def get_related_traj(self, start_port, end_port):
        """引入字典，根据trace得到相关数据
        输入参数应该已经通过map映射，即使用get_test_trace()函数得到的数据
        Args:
            start_port (str): 起始港名称
            end_port (str): 终止港名称

        Returns:
            [list]: 1行3列，3列分别是订单列、标签列、轨迹列
        """
        # 准备相关航线，写入self的字典
        trace_str = start_port+'-'+end_port

        if trace_str in self.match_traj_dict:
            return self.match_traj_dict[trace_str]
        else:
            order_list, label_list, traj_list = self.__get_traj_order_label(
                start_port, end_port)

            if order_list is not None or label_list is not None or traj_list is not None:
                self.match_traj_dict[trace_str] = [
                    order_list, label_list, traj_list]
                return [order_list, label_list, traj_list]
            else:
                return [order_list, label_list, traj_list]

    def get_related_df(self, start_port, end_port):
        """引入字典，根据trace得到相关DataFrame
        输入参数应该已经通过map映射，即使用get_test_trace()函数得到的数据。
        如果没有相关df，则返回None
        Args:
            start_port (str): 起始港名称
            end_port (str): 终止港名称

        Returns:
            (pd.DataFrame): 根据起止港路由得到的匹配df(match_df)
        """
        # 准备相关航线，写入self的字典
        trace_str = start_port+'-'+end_port

        if trace_str in self.match_df_dict:
            return self.match_df_dict[trace_str]
        else:
            match_df = self.__get_match_df(
                start_port, end_port, reset_index=True)

            if match_df is not None:
                self.match_df_dict[trace_str] = match_df.copy()
                return match_df
            else:
                return match_df

            
    def modify_traj_label(self, df, for_parallel=False):
        """切割匹配到的训练集，得到相关轨迹和label
        
        Args:
            df (pd.DataFrame): test集df，只有单个loadingOrder
            for_parallel (bool, optional): 用于处理单条航线时并行化. Defaults to False.
        Returns:
            [list, list, list]: 1行3列，3列分别是订单列、标签列、轨迹列
        """
        # df为test的中的部分，只有单个订单
        # 该函数用于并行化处理
        order = df.loc[df.index[0],'loadingOrder']
        
        trace = df.loc[df.index[0],'TRANSPORT_TRACE'].split('-')
        
        strat_port = portsUtils.get_alias_name(trace[0])
        end_port = portsUtils.get_alias_name(trace[-1])
        
        trace_str = strat_port+'-'+end_port
        
        if trace_str not in self.match_df_dict:
            return [None, None, None]
        
        match_df = self.match_df_dict[trace_str]
        
        if len(match_df) == 0:
            return [None, None, None]
        
        try:
            if len(self.__vessel_name) > 0:
                match_df_ = match_df[match_df[self.__vessel_name]==test_data[self.__vessel_name].unique().tolist()[0]]
                
                if len(match_df_) != 0:
                    match_df_ = match_df_.reset_index(drop=True)
                    cutted_df = self.__cutTrace.cut_trace_for_test(
                        df, match_df_, self.cut_distance_threshold, for_parallel=for_parallel)
                    if len(cutted_df) == 0:
                        cutted_df = self.__cutTrace.cut_trace_for_test(
                            df, match_df, self.cut_distance_threshold, for_parallel=for_parallel)
                else:
                    cutted_df = self.__cutTrace.cut_trace_for_test(
                        df, match_df, self.cut_distance_threshold, for_parallel=for_parallel)
            else:
                cutted_df = self.__cutTrace.cut_trace_for_test(
                    df, match_df, self.cut_distance_threshold, for_parallel=for_parallel)
        except:
            print('船号匹配处错误，test_order:',order)
        
        if len(cutted_df) == 0:
            return [None, None, None]
        
        traj_list_label_series = cutted_df.groupby('loadingOrder')[[
            'timestamp', 'longitude', 'latitude']].apply(lambda x: self.__get_traj_list_label(x, for_traj=False))
        traj_list_label_series = np.array(traj_list_label_series.tolist())
        
        order_list = cutted_df.loadingOrder.unique().tolist()
        
        label_list = list(traj_list_label_series[:, 1])

        traj_list = list(traj_list_label_series[:, 0])
        traj_list = list(map(lambda x: np.array(x), traj_list))
        
        return [order_list, label_list, traj_list]


if __name__ == "__main__":
    train_data = pd.read_csv(config.train_data_drift_dup)

    test_data = pd.read_csv(config.test_data_path)

    pandarallel.initialize(nb_workers=config.nb_workers)

    trajectoryMatching = TrajectoryMatching(
        train_data, 
        geohash_precision=5, 
        cut_distance_threshold=1.3, 
        metric='sspd', 
        mean_label_num=10, 
        vessel_name='vesselMMSI')
    
    order_list, trace_list, traj_list = trajectoryMatching.get_test_trace(test_data)
    
    # 匹配到的订单下标
    matched_index_list = []
    for i in range(len(order_list)):
        length = trajectoryMatching.get_related_df_len(
            trace_list[i][0], trace_list[i][1])
        if length != 0:
            matched_index_list.append(i)
    
    # 匹配到的订单df，！添加了order下标
    matched_df_list = []
    for i in matched_index_list[:]:
        match_df = trajectoryMatching.get_related_df(
            trace_list[i][0], trace_list[i][1])
        matched_df_list.append([i, match_df])
        
    matched_order_list, matched_trace_list, matched_traj_list = [], [], []
    for i in matched_index_list:
        matched_order_list.append(order_list[i])
        matched_trace_list.append(trace_list[i])
        matched_traj_list.append(traj_list[i])
        
    # 路由中没有匹配到的轨迹
    unmatched_index_list = [k for k in range(len(order_list)) if k not in matched_index_list]
    unmatched_order_list = [order_list[i] for i in unmatched_index_list]
    # # 修改内存在对象中的traj和label
    # for data in matched_df_list[25:30]:
    #     order = order_list[data[0]]
        
    #     train_df = data[1]
    #     test_df = test_data[test_data['loadingOrder']==order]
        
    #     cutted_df = cutTrace.cut_trace_for_test(test_df, train_df, 80)
    # print('开始剪切')
    # test_data.groupby('loadingOrder').apply(lambda x: trajectoryMatching.modify_traj_label(x))
    # print('剪切完毕')

    matched_test_data = pd.DataFrame(
        {'loadingOrder': matched_order_list, 'trace': matched_trace_list, 'traj': matched_traj_list})

    final_order_label = matched_test_data.groupby('loadingOrder').parallel_apply(
        lambda x: trajectoryMatching.parallel_get_label(x, test_data))
    final_order_label = final_order_label.tolist()
    
    
    for order in unmatched_order_list:
        final_order_label.append([order, None, None])
    
    with open(config.txt_file_dir_path + 'final_order_label_0714.txt', 'w')as f:
        f.write(str(final_order_label))

    final_order_label_dict = {}
    for i in range(len(final_order_label)):
        final_order_label_dict[final_order_label[i][0]] = final_order_label[i][2]

    for order in unmatched_order_list:
        final_order_label_dict[order] = None
        
    with open(config.txt_file_dir_path + 'final_order_label_dict_0714.txt', 'w')as f:
        f.write(str(final_order_label_dict))
        
    
# TODO 别名处理
# TODO 无用代码删除