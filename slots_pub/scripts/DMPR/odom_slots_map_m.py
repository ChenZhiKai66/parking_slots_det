#!/usr/bin/env python3

# 自己进行坐标变换，监听/odom+/slots_lists=slots_map_lists（和图像的时间戳一致）
import cv2 as cv
import numpy as np
import time
import glob
import math
import time
import os
from parking_lot_msgs.msg import parking_lot  # 自定义停车位消息
from parking_lot_msgs.msg import parking_lots
import config
from util import Timer
import rospy
import roslib
import tf
import tf2_msgs
import tf2_ros
import tf2_py

from tf import TransformerROS
from sensor_msgs.msg import Image, sensor_msgs
from sensor_msgs.msg import PointCloud
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import message_filters
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


# 是否调试
debug = 1
print('debug = ', debug)
timer = Timer()
if debug:  # 定义pc2_map
    slots_map_publisher = rospy.Publisher('slots_map_lists2', PointCloud2, queue_size=1)  # 注意检查topic名称
    pc2_map = PointCloud2()
    pc2_map.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    pc2_map.is_bigendian = False
    pc2_map.point_step = 12
    pc2_map.is_dense = False
    pc2_map.header.frame_id = "map"
else:  # 定义pc2_map
    slots_map_publisher = rospy.Publisher('lots_map_lists', parking_lots, queue_size=1)  # 注意检查topic名称
    pc2_map = parking_lots()
    msg_array = []
points_lists = np.zeros((1, 6, 3), dtype=float, order='C')
count = np.ones(1, dtype=int, order='C')
s_c = 0  # 检测到车位次数


def callback(pc, odom):
    global points_lists  # 需要修改，需要global
    global count
    global s_c
    # print('Message received! ')
    # 坐标变换矩阵
    tran = [odom.pose.position.x, odom.pose.position.y, odom.pose.position.z]
    rot = [odom.pose.orientation.x, odom.pose.orientation.y, odom.pose.orientation.z, odom.pose.orientation.w]
    M_ = np.dot(translation_matrix(tran), quaternion_matrix(rot))
    pc2_map.header.stamp = odom.header.stamp
    i = 0
    th = 1.5
    points_list = np.zeros((pc.width * 4, 3))
    for p in pc2.read_points(pc, field_names=("x", "y", "z"), skip_nans=True):
        p_res = np.dot(M_, np.array([p[0], p[1], p[2], 1.0]))[:3]
        points_list[i] = p_res
        i += 1
    points_list = points_list.reshape(-1, 4, 3)
    points_list_ = np.pad(points_list, ((0, 0), (0, 2), (0, 0)), mode='constant', constant_values=0)  # n*6*3
    points_list_[:, 4, :] = np.mean(points_list, axis=1)  # 中心位置
    points_list_[:, 5, :] = np.mean(points_list[:, :2, :], axis=1)  # 入口中心
    for j in range(points_list.shape[0]):  # 对于每个新的停车位
        de = 0  # 是否已有停车位
        for k in range(points_lists.shape[0]):
            if (pow((points_lists[k][4][0] - points_list_[j][4][0]), 2) + pow(
                    (points_lists[k][4][1] - points_list_[j][4][1]), 2) < th):
                de = 1
                if (pow((points_lists[k][5][0] - points_list_[j][5][0]), 2) + pow(
                        (points_lists[k][5][1] - points_list_[j][5][1]), 2) < th):  # 确定方向相同再融合，否则不管
                    points_lists[k] = (points_lists[k] * count[k] + points_list_[j]) / (count[k] + 1)  # 融合
                    count[k] += 1
                break
        if not de:
            points_lists = np.append(points_lists, [points_list_[j]], axis=0)  # points_list[j]
            count = np.append(count, [1], axis=0)
        s_c += 1
    if debug:
        pc2_map.height = 4
        pc2_map.width = points_lists.shape[0]-1
        pc2_map.row_step = pc2_map.point_step * (points_lists.shape[0]-1)
        pc2_map.data = np.asarray(points_lists[1:, :4, :], np.float32).tobytes()  # 需要重新计算
        print(s_c)
    else:
        for i in range(1, points_lists.shape[0]):  # 忽略第一个点（原点bug修复）
            lot = parking_lot()
            # lot.corner_left_upper = points_lists[i][1]
            # lot.corner_left_low = points_lists[i][3]
            # lot.corner_right_upper = points_lists[i][0]
            # lot.corner_right_low = points_lists[i][2]
            # lot.center = points_lists[i][4]
            # lot.entrance_center = points_lists[i][5]
            lot.corner_left_upper.x = points_lists[i][1][0]
            lot.corner_left_upper.y = points_lists[i][1][1]
            lot.corner_left_upper.z = points_lists[i][1][2]
            lot.corner_left_low.x = points_lists[i][3][0]
            lot.corner_left_low.y = points_lists[i][3][1]
            lot.corner_left_low.z = points_lists[i][3][2]
            lot.corner_right_upper.x = points_lists[i][0][0]
            lot.corner_right_upper.y = points_lists[i][0][1]
            lot.corner_right_upper.z = points_lists[i][0][2]
            lot.corner_right_low.x = points_lists[i][2][0]
            lot.corner_right_low.y = points_lists[i][2][1]
            lot.corner_right_low.z = points_lists[i][2][2]
            lot.center.x = points_lists[i][4][0]
            lot.center.y = points_lists[i][4][1]
            lot.center.z = points_lists[i][4][2]
            lot.entrance_center.x = points_lists[i][5][0]
            lot.entrance_center.y = points_lists[i][5][1]
            lot.entrance_center.z = points_lists[i][5][2]
            distance = sum((points_lists[i][1]/12 - points_lists[i][0]/12) ** 2)
            # print(distance)
            if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
                lot.type = 0
            else:
                lot.type = 1
            msg_array.append(lot)
        pc2_map.parking_lots = msg_array
    slots_map_publisher.publish(pc2_map)


def pub_slots_map():
    rospy.init_node('slots_map_pub_node', anonymous=True)  # 不关心节点唯一性
    # if debug:
    slots_sub = message_filters.Subscriber('slots_lists2', PointCloud2)  # 注意检查topic名称
    # else:
    #     slots_sub = message_filters.Subscriber('slots_lists', parking_lots)
    odom_sub = message_filters.Subscriber('current_pose2', PoseStamped)  # 注意检查topic名称 current_pose地下频率高  odometry
    # odom_sub = message_filters.Subscriber('rslidar_points', PointCloud2, queue_size=10)
    ts = message_filters.ApproximateTimeSynchronizer([slots_sub, odom_sub], queue_size=100, slop=0.1, allow_headerless=False)
    # ts = message_filters.TimeSynchronizer([slots_sub, odom_sub], 100)
    ts.registerCallback(callback)
    rospy.spin()


def translation_matrix(direction):
    """Return matrix to translate by direction vector.
    # >>> v = np.random.random(3) - 0.5
    # >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True """
    M_t = np.identity(4)
    M_t[:3, 3] = direction[:3]  # [-direction[0], -direction[1], -direction[2]]
    return M_t


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    # >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    # >>> np.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    _EPS = np.finfo(float).eps * 4.0
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    T = np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)
    return T


if __name__ == '__main__':
    pub_slots_map()
