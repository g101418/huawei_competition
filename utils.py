from math import radians, cos, sin, asin, sqrt
import itertools
import heapq
import time
import functools

from config import config

def haversine(lon1, lat1, lon2, lat2):
    """计算两个经纬度坐标之间的距离

    Args:
        lon1 (float): 
        lat1 (float): 
        lon2 (float): 
        lat2 (float): 

    Returns:
        distance: 距离
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    d = c * r
    return d


def timethis(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('开始运行程序')
        start_time = time.time()
        print('当前时间：', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        result = func(*args, **kwargs)
        end_time = time.time()
        print('程序结束')
        print('当前时间：', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        period_time = end_time - start_time
        print('程序运行时长：', 
              str(int(period_time//3600))+'小时 '+str(int(period_time%3600)//60)+'分钟 '+str(int(period_time%60))+'秒 ')
        return result
    return wrapper

class PortsUtils(object):
    """用于处理港口相关的数据
    包括根据经纬度获取港口，港口名字转换

    Args:

    """
    def __init__(self,
                 port_map_dict_filename=config.port_map_dict_filename,
                 all_port_points_filename=config.all_port_points_filename,
                 port_alias_filename=config.port_alias_filename,
                 port_near_filename=config.port_near_filename,
                 orders_ports_dict_filename=config.orders_ports_dict_filename,
                 port_map_dict=None,
                 all_port_points=None,
                 port_alias=None,
                 port_naer=None,
                 orders_ports_dict=None):
        super().__init__()
        self.__port_map_dict_filename = port_map_dict_filename
        self.__all_port_points_filename = all_port_points_filename
        self.__port_alias_filename = config.port_alias_filename
        self.__port_near_filename = config.port_near_filename
        self.__orders_ports_dict_filename = orders_ports_dict_filename
        
        if port_map_dict is None:
            with open(self.__port_map_dict_filename, 'r') as f:
                self.port_map_dict = eval(f.read())
        else:
            self.port_map_dict = port_map_dict
        if all_port_points is None:
            with open(self.__all_port_points_filename, 'r') as f:
                self.all_port_points = eval(f.read())
        else:
            self.all_port_points = all_port_points
        if port_alias is None:
            with open(self.__port_alias_filename, 'r') as f:
                self.port_alias = eval(f.read())
        else:
            self.port_alias = port_alias
        if port_naer is None:
            with open(self.__port_near_filename, 'r') as f:
                self.port_naer = eval(f.read())
        else:
            self.port_naer = port_naer
        if orders_ports_dict is None:
            with open(self.__orders_ports_dict_filename, 'r') as f:
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

    def get_port(self, lon, lat, distance_threshold=10.0):
        """根据经纬度得到最匹配的港口名称及坐标、距离。返回值的名称前后无空格

        Args:
            lon (float): 经度
            lat (float): 纬度
            distance_threshold (float, optional): 
                距离阈值，只匹配该距离范围内的港口. Defaults to 10.0.

        Returns:
            port_name, [lon, lat], distance (str, [float, float], float): 
                港口名，经纬度，点到港口距离
        """
        min_distance_point = ''
        min_distance = float('inf')

        for key in self.all_port_points.keys():
            distance = haversine(
                lon, lat, self.all_port_points[key][0], self.all_port_points[key][1])

            if distance <= distance_threshold:
                if distance < min_distance:
                    min_distance_point = key
                    min_distance = distance

        if min_distance_point == '':
            return None, [0.0, 0.0], -1.0
        else:
            min_distance_point = self.get_mapped_port_name(min_distance_point)[0]
            return min_distance_point, self.all_port_points[min_distance_point], min_distance

    def get_mapped_port_name(self, port_name):
        """用于得到映射后的港口名字，优先匹配为删除左右空格的名称

        Args:
            port_name (str): 将要匹配的港口名

        Returns:
            port_name, [lon, lat] (str, [float, float]): 被匹配到的港口名，经纬坐标
        """
        port_name_strip = port_name.strip()

        if port_name_strip in self.all_port_points:
            return port_name_strip, self.all_port_points[port_name_strip]
        elif port_name_strip in self.port_map_dict:
            return self.port_map_dict[port_name_strip][0], self.all_port_points[self.port_map_dict[port_name_strip][0]]
        else:
            pass

        if port_name in self.all_port_points:
            return port_name, self.all_port_points[port_name]
        elif port_name in self.port_map_dict:
            return self.port_map_dict[port_name][0], self.all_port_points[self.port_map_dict[port_name][0]]
        else:
            return None, [0.0, 0.0]
        
    def get_alias_name(self, port_name):
        if port_name in self.port_alias:
            return self.port_alias[port_name]
        else:
            return port_name
    
    def get_near_name(self, port_name):
        for row in self.port_naer:
            if port_name in row:
                # return [k for k in row if k != port_name]
                return [k for k in row]

        return [port_name]
    
    def match_middle_port(self, trace):
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
                
                # TODO 第一个出现港/最后一个
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
        
        return result
    
    def test_match_middle_port_num(self, test_trace, train_order):
        trace = test_trace.split('-')
        trace = list(map(lambda x: portsUtils.get_alias_name(x), trace))
        
        start_port = trace[0]
        end_port = trace[-1]
        
        middle_ports = trace[1:-1] if len(trace) > 2 else []
        if len(middle_ports) != 0:
            middle_port_set = set(middle_ports)
            
            middle_port_near = [portsUtils.get_near_name(port) for port in middle_port_set]
            
            middle_port_sets = map(lambda x: frozenset(x), itertools.product(*middle_port_near))
            middle_port_sets = set(middle_port_sets)
        else:
            middle_port_sets = set()
            
        ports = self.__orders_ports_name_dict[train_order]
        
        if len(ports) < 2:
            return -1
            
        # 用于找到near的最高匹配值
        if start_port not in ports or end_port not in ports:
            return -1
        
        # TODO 第一个出现港/最后一个
        start_index =  [index for index, value in enumerate(ports) if value == start_port][0]
        end_index = [index for index, value in enumerate(ports) if value == end_port][-1]
        
        if start_index >= end_index:
            return -1
        
        ports_set = set(ports[start_index: end_index+1]) - {start_port, end_port}
        
        if len(middle_port_sets) != 0:
            temp_middle_ports_length = []
            for middle_ports in middle_port_sets:
                match_middle_port_len = len(middle_ports & ports_set)
                temp_middle_ports_length.append(match_middle_port_len)
            
            match_middle_port_len = max(temp_middle_ports_length)
        else:
            match_middle_port_len = 0
        
        return len(middle_ports), match_middle_port_len
        
    
    def get_max_match_ports(self, trace, cut_level=1, cut_num=400, matching_down=True):
        result = self.match_middle_port(trace)
        
        if cut_num is not None:
            if len(result) == 0:
                return result, None
            
            if cut_level is not None:
                max_lengths_set = set([item[1] for item in result])
                max_lengths = heapq.nlargest(min(len(result), cut_level), max_lengths_set)
                
                is_single_level = True if len(max_lengths) == 1 else False
                
                if matching_down != True:
                    max_lengths = [max_lengths[min(len(max_lengths)-1, cut_level-1)]]
                
                remain_order = []
                for order, length in result:
                    if length in max_lengths:
                        remain_order.append(order)
                
                return remain_order[:cut_num], is_single_level
            else:
                result_ = result[:cut_num]
                result_ = [item[0] for item in result_]
                return result_, None
            
        return result, None

    def merge_port(self, port_name_1, port_name_2):
        """用来删除/合并两个港口(名称)

        Args:
            port_name_1 (str): 将保留的港口
            port_name_2 (str): 将删除的港口
        """
        self.port_map_dict[port_name_2] = (
            port_name_1, self.get_mapped_port_name(port_name_2)[1])
        del self.all_port_points[port_name_2]

        with open(self.__port_map_dict_filename, 'w') as f:
            f.write(str(self.port_map_dict))
        with open(self.__all_port_points_filename, 'w') as f:
            f.write(str(self.all_port_points))

    def modify_port_coordinates(self, port_name, lon, lat):
        """修改港口坐标位置

        Args:
            port_name (str): 港口名
            lon (float): 经度
            lat (float): 纬度
        """
        self.all_port_points[port_name] = [lon, lat]
        with open(self.__all_port_points_filename, 'w') as f:
            f.write(str(self.all_port_points))

    def modify_port_name(self, port_name_1, port_name_2):
        """
        修改港口名称
        """
        # TODO
        return


class DrawMap(object):
    """用于绘制图案
    """

    def __init__(self,
                 train_data=None,
                 test_data=None,
                 tracemap1_filename=config.tracemap1_filename,
                 tracemap2_filename=config.tracemap2_filename,
                 tracemap_path=config.tracemap_dir_path):
        
        self.__train_data = train_data
        self.__test_data = test_data
        
        with open(tracemap1_filename, 'r') as f:
            self.__tracemap1_txt = str(f.read())
        with open(tracemap2_filename, 'r') as f:
            self.__tracemap2_txt = str(f.read())
        self.__tracemap_path = tracemap_path

    def show_one_map(self, ordername, for_train=False, for_test=False, tracemap_path=None):
        """绘制单个航线的轨迹图

        Args:
            df (DataFrame): 要绘制轨迹的数据集
            ordername (str): 轨迹名称
            tracemap_path (str, optional): 保存路径. Defaults to None.
        """
        if tracemap_path is None:
            tracemap_path = self.__tracemap_path

        if for_train:
            df = self.__train_data
        else:
            df = self.__test_data

        temp = df.loc[df['loadingOrder'] == ordername]
        if len(temp) == 0:
            str_ = '训练' if for_train else '测试'
            raise Exception(str_+'集中无此订单！')
        a = temp.longitude.tolist()
        b = temp.latitude.tolist()
        trace_list = list(map(list, zip(a, b)))

        with open(self.__tracemap_path + 'tracemap_one_'+ordername+'.html', 'w') as f:
            show_list = '{name: "路线' + \
                str(ordername)+'", path:'+str(trace_list)+'},'
            f.write(self.__tracemap1_txt +
                    ''.join(show_list)+self.__tracemap2_txt)

    def show_two_map(self, ordername_test, ordername_train, print_msg=True):
        temp = self.__test_data.loc[self.__test_data['loadingOrder'] == ordername_test]
        if len(temp) == 0:
            raise Exception('测试集中无此订单！')
        a = temp.longitude.tolist()
        b = temp.latitude.tolist()
        trace_list_test = list(map(list, zip(a, b)))
        
        temp = self.__train_data.loc[self.__train_data['loadingOrder'] == ordername_train]
        if len(temp) == 0:
            raise Exception('训练集中无此订单！')
        a = temp.longitude.tolist()
        b = temp.latitude.tolist()
        trace_list_train = list(map(list, zip(a, b)))
        
        trace = self.__test_data[self.__test_data['loadingOrder'] == ordername_test].iloc[0]['TRANSPORT_TRACE']
        
        if print_msg:
            print(trace)
            
        row_test = '{name: "路线'+str(ordername_test)+'", path:'+str(trace_list_test)+'},'
        row_train = '{name: "路线'+str(ordername_train)+'", path:'+str(trace_list_train)+'},'
        show_list = [row_test, row_train]
        
        with open(self.__tracemap_path + 'tracemap_two_'+ordername_test+'-'+ordername_train+'.html', 'w') as f:
            f.write(self.__tracemap1_txt + ''.join(show_list) + self.__tracemap2_txt)
            
    def show_test_train_list_map(self, ordername_test, ordername_train_list, print_msg=True):
        train_df_ = self.__train_data[self.__train_data['loadingOrder'].isin(ordername_train_list)]
        
        temp = self.__test_data.loc[self.__test_data['loadingOrder'] == ordername_test]
        if len(temp) == 0:
            raise Exception('测试集中无此订单！')
        a = temp.longitude.tolist()
        b = temp.latitude.tolist()
        trace_list_test = list(map(list, zip(a, b)))
        
        trace_list_train_list = []
        for ordername_train in ordername_train_list:
            temp = train_df_.loc[train_df_['loadingOrder'] == ordername_train]
            if len(temp) == 0:
                raise Exception('训练集中无此订单！')
            a = temp.longitude.tolist()
            b = temp.latitude.tolist()
            trace_list_train = list(map(list, zip(a, b)))
            trace_list_train_list.append(trace_list_train)
            
        trace = self.__test_data[self.__test_data['loadingOrder'] == ordername_test].iloc[0]['TRANSPORT_TRACE']
        
        if print_msg:
            print(trace)
            
        row_test = '{name: "路线'+str(ordername_test)+'", path:'+str(trace_list_test)+'},'
        show_list = [row_test]
        
        for ordername_train, trace_list_train in zip(ordername_train_list, trace_list_train_list):
            row_train = '{name: "路线'+str(ordername_train)+'", path:'+str(trace_list_train)+'},'
            show_list.append(row_train)
        
        
        with open(config.tracemap_dir_path + '/tracemap_'+ordername_test+'_'+trace+'.html', 'w') as f:
            f.write(self.__tracemap1_txt + ''.join(show_list) + self.__tracemap2_txt)
            
    def show_map_list(self, ordername_list, for_train=False, for_test=False, print_msg=True):
        if for_train:
            df = self.__train_data[self.__train_data['loadingOrder'].isin(ordername_list)]
        else:
            df = self.__test_data[self.__test_data['loadingOrder'].isin(ordername_list)]
        
        trace_list_df_list = []
        for ordername in ordername_list:
            temp = df.loc[df['loadingOrder'] == ordername]
            if len(temp) == 0:
                str_ = '训练' if for_train else '测试'
                raise Exception(str_+'集中无此订单！')
            a = temp.longitude.tolist()
            b = temp.latitude.tolist()
            trace_list_df = list(map(list, zip(a, b)))
            trace_list_df_list.append(trace_list_df)
            
        show_list = []
        
        for ordername, trace_list in zip(ordername_list, trace_list_df_list):
            row_b = '{name: "路线'+str(ordername)+'", path:'+str(trace_list)+'},'
            show_list.append(row_b)
            
        with open(config.tracemap_dir_path + '/tracemap_map_list_'+ordername_list[0]+'.html', 'w') as f:
            f.write(self.__tracemap1_txt + ''.join(show_list) + self.__tracemap2_txt)
            


portsUtils = PortsUtils()

if __name__ == '__main__':
    print("该两点间距离={0:0.3f} km".format(
        haversine(-82.754642, 22.995645, -83.935017, 22.922867)))
    portname, coor, distance = portsUtils.get_port(83.975602, 28.20403)
    print(portname, coor, distance)
    portname, coor = portsUtils.get_mapped_port_name('MXZLO')
    print(portname, coor)
