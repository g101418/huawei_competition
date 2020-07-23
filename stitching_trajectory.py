"""
@ 描述：用于将轨迹按照
"""

from utils import portsUtils

import itertools

class StitchTrajectory(object):
    
    def __init__(self):
        super().__init__()
    
    def __get_port_combinations(self):
        port_names = list(portsUtils.all_port_points.keys())
        # 得到起止港的排列组合，包括正反，但不重复
        port_names_permutations = list(itertools.permutations(port_names, 2))
        
        return port_names_permutations
    
    def split_traj_by_port(self):
        # {'CNSHA-PAMIT': [['AA1233456789', [2003, 4890]]]}
        
        # 得到港口的绝对下标，或first或second
        def get_index(value, start=False, end=False):
            if start:
                result = value[1][1] if value[1][1] > 0 else value[1][0]
            else:
                result = value[1][0] if value[1][0] > 0 else value[1][1]
            
            if result < 0:
                raise Exception('下标为负！' + str(result))
            
            return result
        # 得到起止港的排列组合
        port_names_permutations = self.__get_port_combinations()
        
        result = {}
        for start_port, end_port in port_names_permutations:
            result[start_port+'-'+end_port] = []
            
            for order, ports_value in portsUtils.orders_ports_dict.items():
                port_names = [i[0] for i in ports_value]
                
                if start_port not in port_names or end_port not in port_names:
                    continue
                
                start_port_indexs = [i for i, j in enumerate(port_names) if j == start_port]
                end_port_indexs = [i for i, j in enumerate(port_names) if j == end_port]
                
                for start_port_index in start_port_indexs:
                    for end_port_index in end_port_indexs:
                        
                        if start_port_index >= end_port_index:
                            continue
                        
                        if start_port_index + 1 == end_port_index:
                            continuous = True
                        else:
                            continuous = False
                            
                        start_index = get_index(ports_value[start_port_index], start=True)
                        end_index = get_index(ports_value[end_port_index], end=True)
                        
                        
                        result[start_port+'-'+end_port].append([order, [start_index, end_index], continuous])
        
        return result
    
    