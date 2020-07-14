'''
@Author: Gao S
@Date: 2020-06-16 14:46:14
@LastEditTime: 2020-07-14 12:27:06
@Description: 
@FilePath: /HUAWEI_competition/delete_drift.py
'''
from utils import haversine
import pandas as pd
from pandarallel import pandarallel
from config import config

class DriftPoint(object):
    """删除漂移点

    """
    def __init__(self):
        """初始化

        Args:
            speed_threshold (int, optional): 速度阈值，超过者删除. Defaults to 50.
        """
        super().__init__()
        self.speed_threshold=speed_threshold

    def __get_delete_drift_point(self, df):
        """根据速度得到要删除的点

        Args:
            df (pd.DataFrame): 单个loadingOrder的训练集数据

        Returns:
            (list): 要删除的相对下标
        """
        delete_indexs = []
        anchor_point_i = df.index[0]
        anchor_point_lon = df.loc[df.index[0]]['longitude']
        anchor_point_lat = df.loc[df.index[0]]['latitude']
        anchor_point_time = pd.to_datetime(df.loc[df.index[0], 'timestamp'])
        for i in range(df.index[0]+1, df.index[-1]+1):
            cur_lon = df.loc[i]['longitude']
            cur_lat = df.loc[i]['latitude']
            cur_time = pd.to_datetime(df.loc[i, 'timestamp'])

            distance = haversine(
                anchor_point_lon, anchor_point_lat, cur_lon, cur_lat)
            time_delta_hour = (
                (cur_time - anchor_point_time).total_seconds() / 3600)

            assert time_delta_hour >= 0

            if time_delta_hour <= 0.0027:  # 10秒
                if distance >= self.speed_threshold * 0.0027:
                    delete_indexs.append(i)
                    continue
                else:
                    anchor_point_i = i
                    anchor_point_lon = cur_lon
                    anchor_point_lat = cur_lat
                    anchor_point_time = cur_time
                    continue

            speed = distance / time_delta_hour

            if speed >= self.speed_threshold:
                delete_indexs.append(i)
            else:
                anchor_point_i = i
                anchor_point_lon = cur_lon
                anchor_point_lat = cur_lat
                anchor_point_time = cur_time

        return delete_indexs
    
    def delete_drift_point(self, df, speed_threshold=50):
        """删除漂移点

        Args:
            df (pd.DataFrame): 训练集数据，全部

        Returns:
            (pd.DataFrame): 删除漂移点后数据集
        """
        self.speed_threshold=speed_threshold
        
        pandarallel.initialize(nb_workers=config.nb_workers)
        
        delete_indexs = df.groupby('loadingOrder')[
            ['timestamp', 'longitude', 'latitude']].parallel_apply(self.__get_delete_drift_point)

        # ! 此处可能有bug
        delete_indexs = [j for i in delete_indexs for j in i]

        df.drop(labels=delete_indexs, axis=0, inplace=True)
        df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
        df = df.reset_index(drop=True)
        
        return df
    
driftPoint = DriftPoint()

if __name__ == '__main__':
    train_data = pd.read_csv(config.train_gps_path, header=None)
    train_data.columns = config.train_data_columns

    train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    train_data = train_data.reset_index(drop=True)
    
    
    
    train_data = driftPoint.delete_drift_point(train_data, 50)

    train_data.to_csv(config.train_data_drift, index=False)
