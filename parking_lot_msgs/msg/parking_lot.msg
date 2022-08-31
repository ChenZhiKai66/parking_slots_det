# 全局
Header header

# 车辆最终泊进车位时 从上往下看
geometry_msgs/Point corner_left_upper
geometry_msgs/Point corner_left_low
geometry_msgs/Point corner_right_upper
geometry_msgs/Point corner_right_low

geometry_msgs/Point center           # 方块的中心
geometry_msgs/Point entrance_center  # 入口边的中心

int8 type  # 0 - 垂直  1 - 侧方位 
int8 is_occupied # 0- 无占据 1- 占据
