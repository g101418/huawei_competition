'''
@Author: Gao S
@Date: 2020-06-20 18:09:10
@LastEditTime: 2020-07-16 20:26:20
@Description: 
@FilePath: /HUAWEI_competition/trajectory_matching.py
'''
from cut_trace import CutTrace
from utils import portsUtils, send_mail
from config import config

import numpy as np
import pandas as pd
import traj_dist.distance as tdist
import geohash
import itertools
from datetime import datetime
import time
import heapq
import traceback

from pandarallel import pandarallel

import warnings
warnings.filterwarnings('ignore')


class TrajectoryMatching(object):
    """轨迹匹配，用于得到匹配后轨迹相关label等

    """
    def __init__(self, 
                 train_data, 
                 geohash_precision=4, 
                 metric="sspd",
                 cut_distance_threshold=1.3,
                 use_near=True,
                 mean_label_num=1,
                 top_N_for_parallel=10,
                 cut_level=1000,
                 matching_down=True,
                 cut_num=400,
                 after_cut_mean_num=-1,
                 get_label_way='mean',
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
        self.__top_N_for_parallel = max(0, top_N_for_parallel)
        self.__cut_level = cut_level
        self.__cut_num = cut_num
        self.__after_cut_mean_num = after_cut_mean_num
        self.__matching_down = matching_down
        self.__use_near = use_near
        
        self.__get_label_way = get_label_way if get_label_way in ['mean', 'min', 'median'] else 'mean'
        self.__vessel_name = vessel_name if vessel_name in ['carrierName', 'vesselMMSI'] else ''
        

    def __get_match_df(self, start_port, end_port, reset_index=True, use_near=True):
        """得到与trace相关的训练集，训练集可选是否排序
        如果没有相关df，则返回None
        Args:
            start_port (str): 起始港
            end_port (str): 终点港
            reset_index (Bool): 选择是否返回index重排的df，默认重排序
            use_near (Bool): 选择是否使用附近港

        Returns:
            match_df (pd.DataFrame): 与trace相关的训练集，可选择是否排序
        """
        
        # TODO 此处考虑增加相近港口
        # TODO 初步：考虑结果为空者
        # TODO 中级：考虑将不同起止点进行融合，考虑order重合现象
        
        try:
            start_port_near_names = portsUtils.get_near_name(start_port)
            end_port_near_names = portsUtils.get_near_name(end_port)
            
            if use_near:
                near_name_pairs = [(i,j) for i in start_port_near_names for j in end_port_near_names]
            else:
                near_name_pairs = [(start_port, end_port)]
                
            results = []
            for item in near_name_pairs:
                result = self.__cutTrace.get_use_indexs(item[0], item[1])
                results += result
            results = list(set(results))
            results.sort()
        except:
            traceback.print_exc()
            print('处理near港错误')

        if len(results) == 0:
            return None

        # 得到相关训练集
        match_df = self.train_data.loc[results]
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
        
    def get_related_df_len(self, start_port, end_port, use_near=True):
        """得到trace相关df中订单的数量

        Args:
            start_port (str): 起始港名称
            end_port (str): 终止港名称

        Returns:
            [int]: 相关df中订单的数量
        """
        
        result = self.get_related_df(start_port, end_port, use_near=use_near)
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
            result, is_after_cut = self.modify_traj_label(test_data, for_parallel=for_parallel)
            train_order_list, train_label_list, train_traj_list = result
            if train_label_list is None or len(train_label_list) == 0:
                print('get_final_label，切割后为空', order)
                return None, None
        except:
            traceback.print_exc()
            print('error:', order, 'modify_traj_label')
            return None, None
            
        try:
            cdist = list(tdist.cdist(
                [traj], train_traj_list, metric=self.__metric)[0])
            min_traj_index = cdist.index(min(cdist))
        except:
            traceback.print_exc()
            print('error:', order, 'tdist.cdist')
            return None,None
        
        try:
            mean_label_num = self.__mean_label_num
            
            if self.__after_cut_mean_num > 0 and is_after_cut:
                mean_label_num = self.__after_cut_mean_num
            
            min_traj_index_3 = list(map(lambda x:cdist.index(x), 
                heapq.nsmallest(min(len(train_label_list), mean_label_num), cdist)))
            
            temp_list = list(map(lambda x: x.total_seconds(), 
                          list(np.array(train_label_list)[min_traj_index_3])))
            if self.__get_label_way == 'mean':
                mean_label_seconds = np.mean(temp_list)
            elif self.__get_label_way == 'min':
                mean_label_seconds = np.min(temp_list)
            elif self.__get_label_way == 'median':
                mean_label_seconds = np.median(temp_list)

            mean_label = pd.Timedelta(seconds=mean_label_seconds)
        except:
            traceback.print_exc()
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

    def get_related_df(self, start_port, end_port, use_near=True):
        """引入字典，根据trace得到相关DataFrame
        输入参数应该已经通过map映射，即使用get_test_trace()函数得到的数据。
        如果没有相关df，则返回None
        Args:
            start_port (str): 起始港名称
            end_port (str): 终止港名称
            use_near (Bool): 选择是否使用附近港

        Returns:
            (pd.DataFrame): 根据起止港路由得到的匹配df(match_df)
        """
        # 准备相关航线，写入self的字典
        trace_str = start_port+'-'+end_port

        if trace_str in self.match_df_dict:
            return self.match_df_dict[trace_str]
        else:
            match_df = self.__get_match_df(
                start_port, end_port, reset_index=True, use_near=use_near)

            if match_df is not None:
                self.match_df_dict[trace_str] = match_df.copy()
                return self.match_df_dict[trace_str]
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
        
        trace_ = df.loc[df.index[0],'TRANSPORT_TRACE']
        trace = df.loc[df.index[0],'TRANSPORT_TRACE'].split('-')
        
        strat_port = portsUtils.get_alias_name(trace[0])
        end_port = portsUtils.get_alias_name(trace[-1])
        
        trace_str = strat_port+'-'+end_port
        
        if trace_str not in self.match_df_dict:
            print('modify_traj_label，trace_str不在字典内', order)
            return [None, None, None], None
        
        match_df = self.match_df_dict[trace_str]
        
        if len(match_df) == 0:
            print('modify_traj_label，match_df为空', order)
            return [None, None, None], None
        
        for i in range(10):
            port_match_orders, is_single_level = portsUtils.get_max_match_ports(trace_, cut_level=self.__cut_level+i, cut_num=self.__cut_num, matching_down=self.__matching_down)
            match_df_temp = match_df[match_df['loadingOrder'].isin(port_match_orders)].reset_index(drop=True)
            
            if len(match_df_temp) != 0:
                break
            
        if len(match_df_temp) == 0:
            print('modify_traj_label，match_df_temp为空', order)
            return [None, None, None], None
        
        match_df = match_df_temp
        
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
            
            if len(cutted_df) == 0:
                print('modify_traj_label，cutted_df为空', order)
                return [None, None, None], None
        except:
            traceback.print_exc()
            print('船号匹配处错误，test_order:',order)
            return [None, None, None], None
            
        
        traj_list_label_series = cutted_df.groupby('loadingOrder')[[
            'timestamp', 'longitude', 'latitude']].apply(lambda x: self.__get_traj_list_label(x, for_traj=False))
        traj_list_label_series = np.array(traj_list_label_series.tolist())
        
        order_list = cutted_df.loadingOrder.unique().tolist()
        
        label_list = list(traj_list_label_series[:, 1])

        traj_list = list(traj_list_label_series[:, 0])
        traj_list = list(map(lambda x: np.array(x), traj_list))
        
        return [order_list, label_list, traj_list], not is_single_level

    def process(self, test_data, order=None):
        print('开始运行process函数')
        print('当前时间：', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        order_list, trace_list, traj_list = self.get_test_trace(test_data)
        
        # 匹配到的订单下标
        matched_index_list_all = []
        length_list = []
        if order is None:
            for i in range(len(order_list)):
                length = self.get_related_df_len(trace_list[i][0], trace_list[i][1], use_near=self.__use_near)
                if not self.__use_near and length == 0:
                    length = self.get_related_df_len(trace_list[i][0], trace_list[i][1], use_near=True)
                    
                length_list.append(length)
                if length != 0:
                    matched_index_list_all.append(i)
        else:
            order_index = order_list.index(order)
            length = self.get_related_df_len(trace_list[order_index][0], trace_list[order_index][1], use_near=self.__use_near)
            if not self.__use_near and length == 0:
                length = self.get_related_df_len(trace_list[order_index][0], trace_list[order_index][1], use_near=True)
                    
            length_list = [-1] * order_index
            length_list.append(length)
            if length != 0:
                matched_index_list_all.append(order_index)
            else:
                print('process，匹配轨迹长度为0', order)
                return [[order, None, None]]
            
            
        max_length = max(length_list)
        top_N_length_index = [i for i,x in enumerate(length_list) if x == max_length ]
        top_N_length_index = top_N_length_index[:min(len(top_N_length_index), self.__top_N_for_parallel)]

        matched_index_list = [k for k in matched_index_list_all if k not in top_N_length_index]
        
        
        matched_order_list, matched_trace_list, matched_traj_list = [], [], []
        for i in matched_index_list:
            matched_order_list.append(order_list[i])
            matched_trace_list.append(trace_list[i])
            matched_traj_list.append(traj_list[i])
        
        top_N_matched_order_list, top_N_matched_trace_list, top_N_matched_traj_list = [], [], []
        for i in top_N_length_index:
            top_N_matched_order_list.append(order_list[i])
            top_N_matched_trace_list.append(trace_list[i])
            top_N_matched_traj_list.append(traj_list[i])
        
        
        
        # 路由中没有匹配到的轨迹
        unmatched_index_list = [k for k in range(len(order_list)) if k not in matched_index_list_all]
        unmatched_order_list = [order_list[i] for i in unmatched_index_list]
    
        print('切割前处理完毕')
        print('当前时间：', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        # 
        final_order_label = []
        
        matched_test_data = pd.DataFrame(
            {'loadingOrder': matched_order_list, 'trace': matched_trace_list, 'traj': matched_traj_list})
        # TODO 疑问？此处用self无法并行化
        if len(matched_test_data) > 0:
            final_order_label = matched_test_data.groupby('loadingOrder').parallel_apply(
                lambda x: trajectoryMatching.parallel_get_label(x, test_data))
            final_order_label = final_order_label.tolist()
        
        # 
        top_N_final_order_label = []
        
        print('开始单项并行')
        print('当前时间：', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        top_N_matched_test_data = pd.DataFrame(
            {'loadingOrder': top_N_matched_order_list, 'trace': top_N_matched_trace_list, 'traj': top_N_matched_traj_list})
        # TODO 疑问？此处用self可以并行化
        if len(top_N_matched_test_data) > 0:
            top_N_final_order_label = top_N_matched_test_data.groupby('loadingOrder').apply(
                lambda x: trajectoryMatching.parallel_get_label(x, test_data, for_parallel=True))
            top_N_final_order_label = top_N_final_order_label.tolist()
        
        
        final_order_label += top_N_final_order_label
        if order is None:
            for order in unmatched_order_list:
                print('related_df为空', order)
                final_order_label.append([order, None, None])
        
        print('全部处理完毕')
        print('当前时间：', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        
        return final_order_label

if __name__ == "__main__":
    train_data_path = config.train_data_dup_direc_drift
    test_data_path = config.test_data_drift
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    pandarallel.initialize(nb_workers=config.nb_workers)

    dict_name = '0805'
    kwargs = {
        'geohash_precision': 5, 
        'cut_distance_threshold': 1.3, 
        'metric': 'sspd', 
        'use_near': True,
        'mean_label_num': 15, 
        'top_N_for_parallel': 55,
        'cut_level': 2,
        'matching_down': True,
        'cut_num': 1500,
        'after_cut_mean_num': 3,
        'get_label_way': 'mean',
        'vessel_name': '__'
        }
    contents = [key+'='+str(value) for key,value in kwargs.items()]
    contents += ['train_data_path'+'='+train_data_path,
                  'test_data_path'+'='+test_data_path,
                  'order_ports_dict'+'='+config.orders_ports_dict_filename]
    
    try:
        trajectoryMatching = TrajectoryMatching(train_data, **kwargs)
        
        final_order_label = trajectoryMatching.process(test_data)
        
        with open(config.txt_file_dir_path + 'final_order_label_'+dict_name+'.txt', 'w')as f:
            f.write(str(final_order_label))

        final_order_label_dict = {}
        for i in range(len(final_order_label)):
            final_order_label_dict[final_order_label[i][0]] = final_order_label[i][2]

        with open(config.txt_file_dir_path + 'final_order_label_dict_'+dict_name+'.txt', 'w')as f:
            f.write(str(final_order_label_dict))
            
    except:
        send_mail(subject='traj_match '+dict_name+' 出错', contents=contents)
        traceback.print_exc()
    else:
        send_mail(subject='traj_match '+dict_name+' 运行完毕', contents=contents)
    
# TODO 别名处理
# TODO 无用代码删除
# TODO logging