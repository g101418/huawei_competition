from config import config
from find_all_ports_on_trace import findPorts

import pandas as pd

if __name__ == '__main__':
    test_data = pd.read_csv(config.test_data_drift)

    pandarallel.initialize(nb_workers=config.nb_workers)

    test_ports_dict = findPorts.find_ports(test_data)
    
    test_port_arrive = {}
    for key, value in test_ports_dict.items():
        df = test_data[test_data['loadingOrder']==key]
        
        trace = df.loc[df.index[0], 'TRANSPORT_TRACE'].split('-')[-1]
        trace = portsUtils.get_alias_name(trace)
        
        ports = [i[0] for i in value]
        
        
        if len(set([trace])&set(ports)) > 0:
            value_ = [item for item in value if item[0]==trace][0]
    #         print(key, trace, value_)
        
            index = value_[1][0] if value_[1][0]>0 else value_[1][1]
            
            port_time = pd.to_datetime(test_data.loc[index, 'timestamp'])
            new_time = pd.to_datetime(port_time).strftime('%Y/%m/%d  %H:%M:%S')
        
            # print(key, trace, value_, new_time)
            
            test_port_arrive[key] = new_time
    
    with open(config.txt_file_dir_path + 'test_arrive_ports_dict.txt', 'w') as f:
        f.write(str(test_port_arrive))