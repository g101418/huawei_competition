import itertools
from pandarallel import pandarallel
import geohash
from utils import haversine, get_port

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


def find_all_ports_from_order(df):
    """
    用于发现轨迹中所有经过的港口
    ports包含了所有港口及其对应的下标(绝对下标/相对下标？)如(A:B)
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

        if speed < 3:
            port = get_port(cur_lon, cur_lat, port_dict=ports_points_dict,
                            distance_threshold=25)[0]
            if port is not None:
                cur_in_port_state = True
                if len(ports) == 0:
                    ports.append((port, (i, -1)))
                elif ports[-1][0] == port:  # 还在该港口内
                    last_port_end_index = i
                else:  # 别的港口
                    ports[-1][1][1] = last_port_end_index
                    ports.append((port, (i, -1)))
            else:
                cur_in_port_state = False
        else:
            cur_in_port_state = False

        if last_in_port_state == True and cur_in_port_state == False:
            final_port_end_index = i

    if ports[-1][1][1] == -1:
        ports[-1][1][1] = final_port_end_index

    return ports


if __name__ == '__main__':
    TRAIN_GPS_PATH = './data/train_drift.csv'

    train_data = pd.read_csv(TRAIN_GPS_PATH, header=None)

    train_data.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                          'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                          'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']

