'''
@Author: Gao S
@Date: 2020-06-16 14:59:17
@LastEditTime: 2020-07-15 16:50:44
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
        self.log_file_dir_path = './log_file/'
        

        self.test_data_path = self.data_dir_path + 'test_data0711.csv'

        self.test_data_drift = self.data_dir_path + 'test_data_drift.csv'

        self.train_gps_path = self.data_dir_path + 'train0711.csv'
        self.train_data_dup = self.data_dir_path + 'train_dup.csv'
        self.train_data_drift_dup = self.data_dir_path + 'train_drift_dup.csv'
        self.train_data_drift_dup_direc_MMSI = self.data_dir_path + 'train_drift_dup_direc_MMSI.csv'
        
        
        self.train_data_drift_dup_drop = self.data_dir_path + '_train_drift.csv'
        self.train_data_drift = self.data_dir_path + 'train_drift.csv'
        

        self.nb_workers = 24
        self.train_data_columns = [
            'loadingOrder', 'carrierName', 'timestamp', 'longitude', 'latitude',
            'vesselMMSI', 'speed', 'direction', 'vesselNextport', 'vesselNextportETA',
            'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE'
        ]

        self.orders_ports_dict_filename= self.tool_file_dir_path + 'orders_ports_dict_0719.txt'
        self.port_map_dict_filename = self.tool_file_dir_path + 'port_map_dict.txt'
        # self.all_port_points_filename='./tool_file/all_port_points.txt'
        self.all_port_points_filename = self.tool_file_dir_path + 'new_port_points.txt'
        
        # 包含匹配到轨迹相关港口比例的全部数据，字典
        self.matched_port_dict_filename = self.tool_file_dir_path + 'matched_port_dict.txt'
        # 截留最大比例后，残余400项，字典，只有order名
        self.max_length_matched_port_filename = self.tool_file_dir_path + 'max_length_matched_port.txt'
        
        self.port_alias_filename = self.tool_file_dir_path + 'port_alias.txt'
        self.port_near_filename = self.tool_file_dir_path + 'port_near.txt'
        
        self.tracemap1_filename = self.tool_file_dir_path + 'tracemap1.txt'
        self.tracemap2_filename = self.tool_file_dir_path + 'tracemap2.txt'

config = Config()
