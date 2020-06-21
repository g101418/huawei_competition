'''
@Author: Gao S
@Date: 2020-06-20 18:09:10
@LastEditTime: 2020-06-21 15:24:09
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


class TrajectoryMatching(object):
    """用于匹配轨迹

    Args:

    """

    def __init__(self, train_data, geohash_precision=4, cutting_proportion=-1, metric="sspd"):
        super().__init__()
        self.train_data = train_data
        self.__cutTrace = CutTrace()
        self.match_traj_dict = {}
        self.match_df_dict = {}
        self.geohash_precision = geohash_precision
        self.cutting_proportion = cutting_proportion
        self.metric = metric

    def __get_traj_order_label(self, start_port, end_port):
        """按照起止港得到相关训练集
        确保数据集已经排序
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

        traj_list_label_series = match_df.groupby('loadingOrder')[
            'timestamp', 'longitude', 'latitude'].apply(lambda x: self.__get_traj_list_label(x))
        traj_list_label_series = np.array(traj_list_label_series.tolist())

        label_list = list(traj_list_label_series[:, 1])

        traj_list = list(traj_list_label_series[:, 0])
        traj_list = list(map(lambda x: np.array(x), traj_list))

        return order_list, label_list, traj_list

    # TODO 函数：返回相关训练集，并重新排序，写入字典
    def __get_match_df(self, start_port, end_port, reset_index=True):
        """得到与trace相关的训练集，训练集可选是否排序

        Args:
            start_port (str): 起始港
            end_port (str): 终点港
            reset_index (Bool): 选择是否返回index重排的df

        Returns:
            match_df (pd.DataFrame): 与trace相关的训练集，可选择是否
        """

        start_port = portsUtils.get_mapped_port_name(start_port)[0]
        end_port = portsUtils.get_mapped_port_name(end_port)[0]

        if self.cutting_proportion > 0:
            result = self.__cutTrace.get_use_indexs(start_port, end_port, line=False)
            result_ = []
            for row in result:
                result_ += list(range(row[0], int((row[1]+1)*self.cutting_proportion)))
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

    def __get_label(self, df):
        """得到标签列，即时间差

        Args:
            df (pd.DataFrame): 按group划分后的数据集

        Returns:
            label (): 时间差
        """
        first_time = pd.to_datetime(df.loc[df.index[0], 'timestamp'])
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
            x[1], x[0], precision=self.geohash_precision), traj_list))
        traj_list = [k for k, g in itertools.groupby(traj_list)]
        traj_list = list(
            map(lambda x: [geohash.decode(x)[1], geohash.decode(x)[0]], traj_list))

        if self.cutting_proportion > 0:
            traj_list = traj_list[:int(len(traj_list)*self.cutting_proportion)]

        return traj_list

    def __get_traj_list_label(self, df):
        """得到轨迹列表和标签

        Args:
            df (pd.DataFrame): 按group划分后的数据集

        Returns:
            traj_list, label (np.array, ): 轨迹列表(n行2列的GPS轨迹列表)，标签
        """
        label = self.__get_label(df)
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

        traj_list = test_data.groupby('loadingOrder')[
            'longitude', 'latitude'].apply(self.__get_traj_list)
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
        result = self.get_related_traj(start_port, end_port)[0]
        if result is None:
            return 0
        else:
            return len(result)

    def get_final_label(self, order, trace, traj):
        """输入某一订单的订单名称、trace、轨迹，得到最相似轨迹的label并返回
        在调用此函数前，应该确认该trace存在相关轨迹！
        Args:
            order (str): 订单名
            trace (list): 1行2列list，分别是开始港、结束港
            traj (np.array): 轨迹

        Returns:
            order, label () :订单名称、时间差
        """

        train_order_list, train_label_list, train_traj_list = self.get_related_traj(
            trace[0], trace[1])

        if train_label_list is None:
            return None, None

        cdist = list(tdist.cdist(
            [traj], train_traj_list, metric=self.metric)[0])
        min_traj_index = cdist.index(min(cdist))

        return train_order_list[min_traj_index], train_label_list[min_traj_index]

    def parallel_get_label(self, df):
        """用于并行化处理，得到order及label

        Args:
            df (pd.DataFrame): test集中相关数据构造的数据集的某一行

        Returns:
            [result_order, result_label] ([str, ]): 返回order及对应的label列表
        """
        order = df.loc[df.index[0], 'loadingOrder']
        trace = df.loc[df.index[0], 'trace']
        traj = df.loc[df.index[0], 'traj']

        result_order, result_label = self.get_final_label(order, trace, traj)
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
        输入参数应该已经通过map映射，即使用get_test_trace()函数得到的数据
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
                start_port, end_port, reset_index=True)

            if match_df is not None:
                self.match_df_dict[trace_str] = match_df
                return match_df
            else:
                return match_df


if __name__ == "__main__":
    TRAIN_GPS_PATH = './data/_train_drift.csv'
    train_data = pd.read_csv(TRAIN_GPS_PATH)

    TEST_GPS_PATH = './data/A_testData0531.csv'
    test_data = pd.read_csv(TEST_GPS_PATH)

    pandarallel.initialize()

    trajectoryMatching = TrajectoryMatching(
        train_data, geohash_precision=5, cutting_proportion=0.5, metric='sspd')

    order_list, trace_list, traj_list = trajectoryMatching.get_test_trace(
        test_data)
    # ! 此处得到的trace做了map映射

    # 找到可以匹配到的order
    matched_index_list = []
    for i in range(len(order_list)):
        length = trajectoryMatching.get_related_traj_len(
            trace_list[i][0], trace_list[i][1])
        if length != 0:
            matched_index_list.append(i)

    matched_order_list, matched_trace_list, matched_traj_list = [], [], []
    for i in matched_index_list:
        matched_order_list.append(order_list[i])
        matched_trace_list.append(trace_list[i])
        matched_traj_list.append(traj_list[i])

    matched_test_data = pd.DataFrame(
        {'loadingOrder': matched_order_list, 'trace': matched_trace_list, 'traj': matched_traj_list})

    final_order_label = matched_test_data.groupby('loadingOrder').parallel_apply(
        lambda x: trajectoryMatching.parallel_get_label(x))
    final_order_label = final_order_label.tolist()

    with open('./final_order_label_0621.txt', 'w')as f:
        f.write(str(final_order_label))

    final_order_label_dict = {}
    for i in range(len(final_order_label)):
        final_order_label_dict[final_order_label[i]
                               [0]] = final_order_label[i][2]

    with open('final_order_label_dict.txt', 'w')as f:
        f.write(str(final_order_label_dict))


# if __name__ == "__main__":
#     TRAIN_GPS_PATH = './data/_train_drift.csv'
#     train_data = pd.read_csv(TRAIN_GPS_PATH)

#     TEST_GPS_PATH = './data/A_testData0531.csv'
#     test_data = pd.read_csv(TEST_GPS_PATH)

#     pandarallel.initialize()

#     # !此处cutting_proportion为切割比例，0.5意为前50%
#     trajectoryMatching = TrajectoryMatching(
#         train_data, geohash_precision=5, cutting_proportion=0.5, metric='dtw')

#     order_list, trace_list, traj_list = trajectoryMatching.get_test_trace(
#         test_data)
#     # ! 此处得到的trace做了map映射

#     # 找到可以匹配到的order
#     matched_index_list = []
#     for i in range(len(order_list)):
#         length = trajectoryMatching.get_related_traj_len(
#             trace_list[i][0], trace_list[i][1])
#         if length != 0:
#             matched_index_list.append(i)

#     matched_df_list = []
#     for i in matched_index_list:
#         match_df = trajectoryMatching.get_related_df(
#             trace_list[i][0], trace_list[i][1])
#         matched_df_list.append(match_df)

#     # ! 此处得到了matched_df_list，每一行即对应的训练集
