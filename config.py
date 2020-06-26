'''
@Author: Gao S
@Date: 2020-06-16 14:59:17
@LastEditTime: 2020-06-26 13:49:42
@Description: 
@FilePath: /HUAWEI_competition/config.py
'''
class Config:
    def __init__(self):
        super().__init__()
        self.data_path = './data/'
        self.train_gps_path = './data/train0523.csv'
        self.test_gps_path = './data/A_testData0531.csv'
        self.tracemap_dir_path = './tracemaps/'
        self.nb_workers = 24
        self.train_data_columns = [
            'loadingOrder', 'carrierName', 'timestamp', 'longitude', 'latitude',
            'vesselMMSI', 'speed', 'direction', 'vesselNextport', 'vesselNextportETA',
            'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE'
        ]
        self.orders_ports_dict_filename='./tool_file/orders_ports_dict.txt'
        self.port_map_dict_filename='./tool_file/port_map_dict.txt'
        # self.all_port_points_filename='./tool_file/all_port_points.txt'
        self.all_port_points_filename='./tool_file/new_port_points.txt'

config = Config()



