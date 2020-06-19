# 计算地图上两点经纬度间的距离
from math import radians, cos, sin, asin, sqrt
# Haversine(lon1, lat1, lon2, lat2)的参数代表：经度1，纬度1，经度2，纬度2（十进制度数）


def haversine(lon1, lat1, lon2, lat2):
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


class Ports(object):
    def __init__(self,
                 port_map_dict_filename='./data/port_map_dict.txt',
                 all_port_points_filename='./data/all_port_points.txt',
                 port_map_dict=None,
                 all_port_points=None):
        super().__init__()
        self.port_map_dict_filename = port_map_dict_filename
        self.all_port_points_filename = all_port_points_filename

        if port_map_dict is None:
            with open(self.port_map_dict_filename, 'r') as f:
                self.port_map_dict = eval(f.read())
        if all_port_points is None:
            with open(self.all_port_points_filename, 'r') as f:
                self.all_port_points = eval(f.read())

        self.port_map_dict = port_map_dict
        self.all_port_points = all_port_points

    def get_port(self, lon, lat, distance_threshold=10.0):
        """
        根据经纬度得到最匹配的港口名称及坐标、距离
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
            return min_distance_point, self.all_port_points[min_distance_point], min_distance

    def get_port_lonlat(self, lon, lat, distance_threshold=10.0, filename='./data/all_port_points.txt', port_dict=None):
        """
        根据经纬度得到最匹配的港口名称及坐标、距离。返回值的名称前后无空格
        """

        if port_dict is None:
            with open(filename, 'r') as f:
                port_dict = eval(f.read())

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
            if ' ' in min_distance_point:
                min_distance_point_strip = min_distance_point.strip()
                if min_distance_point_strip in self.all_port_points:
                    if min_distance_point_strip != min_distance_point:
                        min_distance_point = min_distance_point_strip

            return min_distance_point, self.all_port_points[min_distance_point], min_distance

    def get_mapped_port_name(self, port_name,
                             port_map_dict_filename='./data/port_map_dict.txt',
                             all_port_points_filename='./data/all_port_points.txt',
                             port_map_dict=None,
                             all_port_points=None):
        """
        用于得到映射后的港口名字
        优先匹配为删除左右空格的名字
        """
        if port_map_dict is None:
            with open(port_map_dict_filename, 'r') as f:
                port_map_dict = eval(f.read())
        if all_port_points is None:
            with open(all_port_points_filename, 'r') as f:
                all_port_points = eval(f.read())

        port_name_strip = port_name.strip()

        if port_name_strip in all_port_points:
            return port_name_strip, all_port_points[port_name_strip]
        elif port_name_strip in port_map_dict:
            return port_map_dict[port_name_strip][0], all_port_points[port_map_dict[port_name_strip][0]]
        else:
            pass

        if port_name in all_port_points:
            return port_name, all_port_points[port_name]
        elif port_name in port_map_dict:
            return port_map_dict[port_name][0], all_port_points[port_map_dict[port_name][0]]
        else:
            return None, [0.0, 0.0]


if __name__ == '__main__':
    print("该两点间距离={0:0.3f} km".format(
        haversine(-82.754642, 22.995645, -83.935017, 22.922867)))
