from config import config
from utils import portsUtils
import itertools

class PortMatching(object):
    def __init__(self,
                 orders_ports_dict_filename=config.orders_ports_dict_filename,
                 orders_ports_dict=None):
        super().__init__()
        
        if orders_ports_dict is None:
            with open(orders_ports_dict_filename, 'r') as f:
                self.orders_ports_dict = eval(f.read())
        else:
            self.orders_ports_dict = orders_ports_dict
            
        self.__orders_ports_name_dict = {}
        for key in self.orders_ports_dict.keys():
            if len(self.orders_ports_dict[key]) == 0:
                self.__orders_ports_name_dict[key] = []
            else:
                self.__orders_ports_name_dict[key] = [
                    k[0] for k in self.orders_ports_dict[key]]  # 港口名list
        
    def match(self, trace, cut_num=None):
        trace = trace.split('-')
        trace = list(map(lambda x: portsUtils.get_alias_name(x), trace))
        
        start_port = trace[0]
        end_port = trace[-1]
        
        # 处理起止点的near港问题
        start_port_near_names = portsUtils.get_near_name(start_port)
        end_port_near_names = portsUtils.get_near_name(end_port)
        # 起止港的全部near港集
        near_name_pairs = [(i,j) for i in start_port_near_names for j in end_port_near_names]
        
        # TODO 此处利用set进行比较
        # 处理中间港的near问题
        middle_ports = trace[1:-1] if len(trace) > 2 else []
        if len(middle_ports) != 0:
            middle_port_set = set(middle_ports)
            
            middle_port_near = [portsUtils.get_near_name(port) for port in middle_port_set]
            
            middle_port_sets = map(lambda x: frozenset(x), itertools.product(*middle_port_near))
            middle_port_sets = set(middle_port_sets)
        else:
            middle_port_sets = set()
        
        result = []
        
        # 循环所有train中港口
        for key, ports in self.__orders_ports_name_dict.items():
            
            if len(ports) < 2:
                continue
            
            # 用于找到near的最高匹配值
            temp_lengths = []
            for start_port, end_port in near_name_pairs:
                
                if start_port not in ports or end_port not in ports:
                    continue
                
                start_index =  [index for index, value in enumerate(ports) if value == start_port][0]
                end_index = [index for index, value in enumerate(ports) if value == end_port][-1]
                
                if start_index >= end_index:
                    continue
                
                ports_set = set(ports[start_index: end_index+1]) - {start_port, end_port}
                
                
                if len(middle_port_sets) != 0:
                    temp_middle_ports_length = []
                    for middle_ports in middle_port_sets:
                        match_middle_port_len = len(middle_ports & ports_set)
                        temp_middle_ports_length.append(match_middle_port_len / len(ports_set | middle_ports))
                    
                    match_middle_port_len = max(temp_middle_ports_length)
                else:
                    match_middle_port_len = 0
                
                temp_lengths.append(match_middle_port_len)
            
            if len(temp_lengths) == 0:
                pass
            else:
                result.append((key, max(temp_lengths)))
        
        result.sort(key=lambda x: x[1], reverse=True)
        
        if cut_num is not None:
            if len(result) == 0:
                return result
            
            max_length = result[0][1]
            
            remain_order = []
            for order, length in result:
                if length == max_length:
                    remain_order.append(order)
            
            return remain_order[:cut_num]
        
        return result