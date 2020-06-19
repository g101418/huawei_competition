class Config:
    def __init__(self):
        super().__init__()

        self.data_path = './data/'
        self.train_gps_path = './data/train0523.csv'
        self.test_gps_path = './data/A_testData0531.csv'
        self.tracemap_dir_path = './tracemaps/'
        self.nb_workers = 20
        self.train_data_columns = [
            'loadingOrder', 'carrierName', 'timestamp', 'longitude', 'latitude',
            'vesselMMSI', 'speed', 'direction', 'vesselNextport', 'vesselNextportETA',
            'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE'
        ]


config = Config()



