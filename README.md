# 2020华为云大数据挑战赛

## 赛题介绍

赛方给出上亿条航船的数据，包括订单在某个时间戳所在的GPS坐标、船号、承运商、速度、方向、路由(到港序列)等信息。要求根据这些历史航线信息，对测试集中给出的裁剪后的轨迹进行到港时间预测(ETA)。

## 代码介绍

### 整体介绍

| 文件                              | 介绍                                                       |
| --------------------------------- | :--------------------------------------------------------- |
| config.py                         | 关键信息的配置文件                                         |
| cut_trace.py                      | 用于根据港口信息得到裁剪后的航线、用于将航线裁剪至合适长度 |
| delete_drift.py                   | 删除漂移点                                                 |
| find_all_ports_on_trace.py        | 根据GPS坐标及速度找到航线经过的港口，及入港出港下标        |
| trajectory_matching.py            | 用于将label和轨迹分流后分别比较并计算结果，是主要代码      |
| utils.py                          | 各种工具性代码                                             |
| stitching_trajectory.py           | 用于轨迹分段                                               |
| find_ports_for_test.py            | 用于找到test集中错误轨迹，得到已经到港的航线ETA            |
| tool_file/new_port_points.txt     | 港口坐标                                                   |
| tool_file/orders_ports_dict_X.txt | 主要入港出港下标字典，由find_all_ports_on_trace.py得到     |
| tool_file/port_alias.txt          | 用于港口别名替换                                           |
| tool_file/port_near.txt           | 用于附近港匹配，扩充训练集                                 |
| tool_file/tracemapX.txt           | 用于调用高德API绘制航线                                    |

### 分别介绍

#### cut_trace.py

主要代码是**CutTrace**类，包括三个函数**get_use_indexs()**、**get_use_indexs_len()**及**cut_trace_for_test()**



##### get_use_indexs(self, start_port, end_port, match_start_end_port=False, line=True)

根据输入函数的起点港口名称**start_port**和终点港口名称**end_port**，迭代检索**orders_ports_dict_X**字典，按照起止港名称切割轨迹，而不只是用原始轨迹起点港到终点港的数据，以此来扩大训练集。



##### get_use_indexs_len(self, start_port, end_port)

调用**get_use_indexs()**函数，得到满足起止港要求航线的数量。



##### cut_trace_for_test(self, test_df, match_df, distance_threshold=80, for_start=False, for_parallel=True)

根据输入的test轨迹(**test_df**)以及根据起止港得到的测试集(**match_df**)，将测试集轨迹从两头向中间逼近test轨迹，符合距离阈值**distance_threshold**要求后返回切割后的轨迹。
切割后的轨迹一般情况下要比test轨迹两端略微突出。label取终点时间戳减切割后train轨迹靠近test轨迹末端的时间戳的时间差。

当**for_start**为True时，是对轨迹进行从train起点到test起点的匹配，切割的轨迹是从train起点到train接近test起点的路径。时间戳则可在函数外以头尾时间戳相减得到。

**for_parallel**主要用于并行化加速。

---

#### delete_drift.py

代码封装在**DriftPoint**类中，主要思路是根据前后点GPS距离和时间戳计算速度，速度过大者标记为异常点并删除。

实际使用中，应该先剔除direction字段异常的行。

---

#### find_all_ports_on_trace.py

代码封装在**FindPorts**类中，主要思路是根据前后点GPS距离和时间戳计算速度，当速度小于某个阈值时，计算是否在某个港内，并将下标记录。

尝试过考虑塞港，按照第二次进港进行判断，但效果不佳。

---

#### trajectory_matching.py

主要代码，封装在**TrajectoryMatching**类内。

主要参数如下：

| 参数                   | 介绍                                                         |
| ---------------------- | ------------------------------------------------------------ |
| geohash_precision      | geohash精度，早期用于轨迹合并等，后期完全只用于加速轨迹相似度匹配 |
| cut_distance_threshold | cut_trace中，切割轨迹时的距离阈值                            |
| metric                 | 轨迹相似度算法，默认sspd                                     |
| use_near               | 是否使用附近港扩大训练集                                     |
| match_start_end_port   | 是否只用起止港满足要求的航线用于训练，或是否不对train轨迹进行切割 |
| mean_label_num         | 最匹配轨迹的前n条平均                                        |
| max_mean_label_ratio   | 轨迹数过少时，参与平均的轨迹数不得超过的比例                 |
| top_N_for_parallel     | 单项并行的轨迹数量，效果不佳，不如全部循环并行               |
| cut_level              | 中间港匹配等级，数字越小中间港匹配越好                       |
| matching_down          | 中间港匹配是否向下整合，以扩大训练集                         |
| cut_num                | train集航线过多时用于扎断训练的数量，后台随机shuffle取得     |
| after_cut_mean_num     | test中间港非空时，参与平均的数量                             |
| get_label_way          | 对label的处理方式，可选mean(平均)、min(最小)、median(中位数) |
| vessel_name            | 是否匹配船号、承运商                                         |

---

#### utils.py

包含各类工具代码，分别封装在**IndexConversion**、**PortsUtils**、**DataAnalyseUtils**、**DrawMap**及单项的函数中。



##### DrawMap

用于调用高德API绘制航线。



##### IndexConversion

用于将航线中的相对坐标和绝对坐标进行转换。



##### DataAnalyseUtils

用于数据去重



##### PortsUtils

与港口处理相关的代码。

**get_port()**函数根据经纬度返回最匹配的港口。

**get_mapped_port_name()**函数将港口名称映射为映射后名称，后期主要用于获取港口坐标。

**get_alias_name()**函数用于得到港口别名

**get_near_name()**函数用于得到附近港口

**get_max_match_ports()**函数用于匹配中间港，并按等级排布

