from math import radians, cos, sin, asin, sqrt
from pandarallel import pandarallel
import geohash

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

train_gps_path = './data/train0523.csv'

train_data = pd.read_csv(train_gps_path)

train_data.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                      'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                      'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']


def delete_drift_point(df):

    speed_ranger = 50  # km/h

    delete_indexs = []
    anchor_point_i = df.index[0]
    anchor_point_lon = df.loc[df.index[0]]['longitude']
    anchor_point_lat = df.loc[df.index[0]]['latitude']
    anchor_point_time = pd.to_datetime(df.loc[df.index[0], 'timestamp'])
    for i in range(df.index[0]+1, df.index[-1]+1):
        cur_lon = df.loc[i]['longitude']
        cur_lat = df.loc[i]['latitude']
        cur_time = pd.to_datetime(df.loc[i, 'timestamp'])

        distance = Haversine(
            anchor_point_lon, anchor_point_lat, cur_lon, cur_lat)
        time_delta_hour = (
            (cur_time - anchor_point_time).total_seconds() / 3600)

        assert time_delta_hour >= 0

        if time_delta_hour <= 0.0027:  # 10秒
            if distance >= speed_ranger * 0.0027:
                delete_indexs.append(i)
                continue
            else:
                anchor_point_i = i
                anchor_point_lon = cur_lon
                anchor_point_lat = cur_lat
                anchor_point_time = cur_time
                continue

        speed = distance / time_delta_hour

        if speed >= speed_ranger:
            delete_indexs.append(i)
        else:
            anchor_point_i = i
            anchor_point_lon = cur_lon
            anchor_point_lat = cur_lat
            anchor_point_time = cur_time


#     test_df = test_df.drop(labels=delete_indexs, axis=0)

    return delete_indexs


# Haversine(lon1, lat1, lon2, lat2)的参数代表：经度1，纬度1，经度2，纬度2（十进制度数）


def Haversine(lon1, lat1, lon2, lat2):
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    d = c * r
    return d


train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
train_data = train_data.reset_index(drop=True)

pandarallel.initialize()

delete_indexs = train_data.groupby('loadingOrder')[
    ['timestamp', 'longitude', 'latitude']].parallel_apply(delete_drift_point)
delete_indexs = [j for i in delete_indexs for j in i]


train_data.drop(labels=delete_indexs, axis=0, inplace=True)
train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
train_data = train_data.reset_index(drop=True)

train_data.to_csv('./data/train_drift.csv', header=None, index=False)
