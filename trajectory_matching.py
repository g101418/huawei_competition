'''
@Author: Gao S
@Date: 2020-06-20 18:09:10
@LastEditTime: 2020-06-29 17:22:26
@Description: 
@FilePath: /HUAWEI_competition/trajectory_matching.py
'''
from cut_trace import CutTrace
from utils import portsUtils

import numpy as np
import pandas as pd
import traj_dist.distance as tdist
import geohash
import itertools

from pandarallel import pandarallel

import warnings
warnings.filterwarnings('ignore')


class TrajectoryMatching(object):
    """用于匹配轨迹

    Args:

    """

    def __init__(self, 
                 train_data, 
                 geohash_precision=4, 
                 metric="sspd",
                 cut_distance_threshold=-1):
        super().__init__()
        self.train_data = train_data
        self.__cutTrace = CutTrace()
        self.match_traj_dict = {}
        self.match_df_dict = {}
        self.__geohash_precision = geohash_precision
        self.__cutting_proportion = -1 # 按比例切割暂时不用
        self.__metric = metric
        self.cut_distance_threshold = cut_distance_threshold

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

    # TODO 函数：返回相关训练集，并重新排序，写入字典
    def __get_match_df(self, start_port, end_port, reset_index=True, for_df=False):
        """得到与trace相关的训练集，训练集可选是否排序
        如果没有相关df，则返回None
        Args:
            start_port (str): 起始港
            end_port (str): 终点港
            reset_index (Bool): 选择是否返回index重排的df，默认重排序

        Returns:
            match_df (pd.DataFrame): 与trace相关的训练集，可选择是否排序
        """

        start_port = portsUtils.get_mapped_port_name(start_port)[0]
        end_port = portsUtils.get_mapped_port_name(end_port)[0]
        # TODO 删除__cutting_proportion相关
        if for_df == True and self.__cutting_proportion > 0:
            result = self.__cutTrace.get_use_indexs(
                start_port, end_port, line=False)
            result_ = []
            for row in result:
                if len(row) != 0:
                    result_ += list(range(row[1], int((row[2]+1-row[1]) * self.__cutting_proportion)+row[1]))
            result = result_
        else:
            result = self.__cutTrace.get_use_indexs(start_port, end_port)

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

        if self.__cutting_proportion > 0:
            traj_list = traj_list[:int(len(traj_list)*self.__cutting_proportion)]

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

        trace[0] = portsUtils.get_mapped_port_name(trace[0])[0]
        trace[1] = portsUtils.get_mapped_port_name(trace[1])[0]

        return trace

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
        start_port = portsUtils.get_mapped_port_name(start_port)[0]
        end_port = portsUtils.get_mapped_port_name(end_port)[0]
        
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
        start_port = portsUtils.get_mapped_port_name(start_port)[0]
        end_port = portsUtils.get_mapped_port_name(end_port)[0]
        
        result = self.get_related_df(start_port, end_port)
        if result is None:
            return 0
        return result.loadingOrder.nunique()

    def get_final_label(self, order, trace, traj, test_data):
        """输入某一订单的订单名称、trace、轨迹，得到最相似轨迹的label并返回
        在调用此函数前，应该确认该trace存在相关轨迹！
        Args:
            order (str): 订单名
            trace (list): 1行2列list，分别是开始港、结束港
            traj (np.array): test的轨迹

        Returns:
            order, label (str, pd.Timedelta) :订单名称、时间差
        """
        try:
            if self.cut_distance_threshold < 0:
                train_order_list, train_label_list, train_traj_list = self.get_related_traj(
                    trace[0], trace[1])

                if train_label_list is None:
                    return None, None
            else:
                train_order_list, train_label_list, train_traj_list = self.modify_traj_label(test_data)
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

        return train_order_list[min_traj_index], train_label_list[min_traj_index]

    def parallel_get_label(self, df, test_data):
        """用于并行化处理，得到order及label

        Args:
            df (pd.DataFrame): test集中相关数据构造的数据集的某一行

        Returns:
            [result_order, result_label] ([str, ]): 返回order及对应的label列表
        """
        order = df.loc[df.index[0], 'loadingOrder']
        trace = df.loc[df.index[0], 'trace']
        traj = df.loc[df.index[0], 'traj']

        test_data_ = test_data[test_data['loadingOrder']==order]
        
        result_order, result_label = self.get_final_label(order, trace, traj, test_data_)
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
            [list]: 1行3列，3列分别是订单列、标签列、轨迹列
        """
        # 准备相关航线，写入self的字典
        trace_str = start_port+'-'+end_port

        if trace_str in self.match_df_dict:
            return self.match_df_dict[trace_str]
        else:
            match_df = self.__get_match_df(
                start_port, end_port, reset_index=True, for_df=True)

            if match_df is not None:
                self.match_df_dict[trace_str] = match_df.copy()
                return match_df
            else:
                return match_df

            
    def modify_traj_label(self, df):
        # df为test的中的部分，只有单个订单
        # 该函数用于并行化处理
        order = df.loc[df.index[0],'loadingOrder']
        
        trace = df.loc[df.index[0],'TRANSPORT_TRACE'].split('-')
        strat_port = portsUtils.get_mapped_port_name(trace[0])[0]
        end_port = portsUtils.get_mapped_port_name(trace[1])[0]
        
        trace_str = strat_port+'-'+end_port
        
        if trace_str not in self.match_df_dict:
            return [None, None, None]
        
        match_df = self.match_df_dict[trace_str]
        
        # ! 考虑船号
        try:
            match_df_ = match_df[match_df['vesselMMSI']==test_data.vesselMMSI.unique().tolist()[0]]
            
            if len(match_df_) == 0 and len(match_df) != 0:
                match_df_ = match_df
            elif len(match_df_) == 0 and len(match_df) == 0:
                return [None, None, None]
            
            match_df = match_df_.reset_index(drop=True)
        except:
            print('船号匹配处错误，test_order:',order)
        
        cutted_df = self.__cutTrace.cut_trace_for_test(
            df, match_df, self.cut_distance_threshold, for_parallel=False)
        
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
    TRAIN_GPS_PATH = './data/train_drift.csv'
    train_data = pd.read_csv(TRAIN_GPS_PATH)
    train_data.columns = ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']

    TEST_GPS_PATH = './data/new_test_data_B.csv'
    test_data = pd.read_csv(TEST_GPS_PATH)

    pandarallel.initialize()

    trajectoryMatching = TrajectoryMatching(
        train_data, geohash_precision=5, cut_distance_threshold=1.3, metric='sspd')
    
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
    
    with open('./final_order_label_0625.txt', 'w')as f:
        f.write(str(final_order_label))

    final_order_label_dict = {}
    for i in range(len(final_order_label)):
        final_order_label_dict[final_order_label[i][0]] = final_order_label[i][2]

    with open('final_order_label_dict_0625.txt', 'w')as f:
        f.write(str(final_order_label_dict))