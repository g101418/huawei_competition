'''
@Author: Gao S
@Date: 2020-06-16 14:59:17
@LastEditTime: 2020-06-30 08:07:31
@Description: 
@FilePath: /HUAWEI_competition/config.py
'''


class Config:
    """配置文件，包含相关文件路径等
    
    """
    def __init__(self):
        super().__init__()
        self.data_dir_path = './data/'
        self.tracemap_dir_path = './tracemaps/'
        self.txt_file_dir_path = './txt_file/'
        self.tool_file_dir_path = './tool_file/'
        

        self.test_data_path = self.data_dir_path + 'new_test_data_B.csv'

        self.train_gps_path = self.data_dir_path + 'train0523.csv'
        self.train_data_drift_drop = self.data_dir_path + '_train_drift.csv'
        self.train_data_drift = self.data_dir_path + 'train_drift.csv'

        self.nb_workers = 24
        self.train_data_columns = [
            'loadingOrder', 'carrierName', 'timestamp', 'longitude', 'latitude',
            'vesselMMSI', 'speed', 'direction', 'vesselNextport', 'vesselNextportETA',
            'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE'
        ]

        # self.orders_ports_dict_filename='./tool_file/orders_ports_dict.txt'
        self.port_map_dict_filename = self.tool_file_dir_path + 'port_map_dict.txt'
        # self.all_port_points_filename='./tool_file/all_port_points.txt'
        self.all_port_points_filename = self.tool_file_dir_path+'new_port_points.txt'


config = Config()
