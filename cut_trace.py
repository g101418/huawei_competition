'''
@Author: Gao S
@Date: 2020-06-20 13:35:36
@LastEditTime: 2020-06-20 16:01:17
@Description: 切割轨迹
@FilePath: /HUAWEI_competition/cut_trace.py
'''
from utils import portsUtils


class CutTrace(object):
    """用于切割轨迹

    Args:

    """

    def __init__(self,
                 orders_ports_dict_filename='./data/orders_ports_dict.txt',
                 orders_ports_dict=None):
        super().__init__()
        if orders_ports_dict is None:
            with open('./data/orders_ports_dict.txt', 'r') as f:
                self.orders_ports_dict = eval(f.read())
        else:
            self.orders_ports_dict = orders_ports_dict

        # 用于快速匹配，只保留名字，正向反向分别保存
        self.__orders_ports_name_dict = {}
        for key in self.orders_ports_dict.keys():
            if len(self.orders_ports_dict[key]) == 0:
                self.__orders_ports_name_dict[key] = []
            else:
                self.__orders_ports_name_dict[key] = [
                    k[0] for k in self.orders_ports_dict[key]]  # 港口名list

    def get_use_indexs(self, start_port, end_port, line=True):
        """获取与start_port、end_port相关的轨迹的所有下标
        start_port优先匹配最靠前港，end_port优先匹配最靠后港

        Args:
            start_port (str): 起始港，映射后名称
            end_port (str): 终点港，映射后名称
            line (Bool): 是否直接返回平铺的全部索引值，如果False，则会返回对应的订单及子下标
        """

        # TODO list为空

        use_indexs = []

        for key in self.orders_ports_dict.keys():
            if len(self.__orders_ports_name_dict[key]) < 2:
                continue
            if start_port in self.__orders_ports_name_dict[key] and end_port in self.__orders_ports_name_dict[key]:
                # 起止港均在内
                start_indexs = [i for i in range(len(
                    self.__orders_ports_name_dict[key])) if self.__orders_ports_name_dict[key][i] == start_port]
                end_indexs = [i for i in range(len(
                    self.__orders_ports_name_dict[key])) if self.__orders_ports_name_dict[key][i] == end_port]

                if start_indexs[0] >= end_indexs[-1]:
                    continue

                start_index, end_index = -1, -1
                i, j = 0, len(end_indexs)-1
                while i < len(start_indexs) and j >= 0 and start_indexs[i] < end_indexs[j]:
                    start_first, _ = self.orders_ports_dict[key][start_indexs[i]][1]
                    if start_first < 0:
                        i += 1
                        continue
                    _, end_second = self.orders_ports_dict[key][end_indexs[i]][1]
                    if end_second < 0:
                        j -= 1
                        continue
                    if start_first >= end_second:
                        break

                    start_index, end_index = start_first, end_second
                    break

                if start_index == -1 or end_index == -1:
                    continue
                
                if line == True:
                    use_indexs.append([start_index, end_index])
                else:
                    use_indexs.append([key, start_index, end_index])
            else:
                continue
            
        # TODO 解返回值，现在：[[xx, xx], [xx, xx], ...]
        if line == True:
            use_indexs_ = []
            for row in use_indexs:
                use_indexs_ += list(range(row[0], row[1]+1))
            return use_indexs_
        else:
            return use_indexs
    
cutTrace = CutTrace()


if __name__ == '__main__':
    # CNYTN-MXZLO
    start_port, _ = portsUtils.get_mapped_port_name('CNYTN')
    end_port, _ = portsUtils.get_mapped_port_name('MXZLO')
    
    related_indexs = cutTrace.get_use_indexs(start_port, end_port, line=True)
    # print(related_indexs[:10])