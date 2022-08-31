#!/usr/bin/env python3

# 从原始图像进行检测，自己进行坐标变换，发布pointcloud2消息（全局/局部坐标）
import cv2 as cv
import numpy as np
import time
import glob
import math
import torch
import os
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
from model import DirectionalPointDetector
from util import Timer
import rospy
import roslib
import tf
import tf2_msgs
import tf2_ros
import tf2_py
from tf import TransformerROS
from sensor_msgs.msg import Image, sensor_msgs
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from cv_bridge import CvBridge, CvBridgeError
import queue
import threading, time


# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


timer = Timer()
# 参数初始化
input1 = VideoCapture(1)
input2 = VideoCapture(2)
input3 = VideoCapture(3)
input0 = VideoCapture(0)
mtx1 = np.array(
    [[475.75941694062624, 0.0, 953.6875439910156], [0.0, 475.8999573447999, 524.6256242862999], [0.0, 0.0, 1.0]])
dist1 = np.array([[0.061648264718480845], [-0.011213257242909542], [-0.0002448465243520917], [0.0002024884741740028]])
mtx2 = np.array(
    [[474.11482367731435, 0.0, 970.9422789169496], [0.0, 475.0627595972449, 554.2074603636624], [0.0, 0.0, 1.0]])
dist2 = np.array([[0.058049435249809975], [-0.0036229297441117206], [-0.004129058798634489], [0.00097655626676563]])
DIM = (1920, 1080)
mapx, mapy = cv.fisheye.initUndistortRectifyMap(mtx1, dist1, np.eye(3), mtx1, DIM, cv.CV_16SC2)
mapx2, mapy2 = cv.fisheye.initUndistortRectifyMap(mtx2, dist2, np.eye(3), mtx2, DIM, cv.CV_16SC2)

# 透视变换参数
H = 600
a, b = (H-510)/2, (H-690)/2
p1, p2, p3, p4 = [int(150+a), int(540+b)], [int(0+a), int(690+b)], [int(360+a), int(540+b)], [int(510+a), int(690+b)]
p5, p6, p7, p8 = [int(0+a), int(0+b)], [int(130+a), int(130+b)], [int(510+a), int(0+b)], [int(380+a), int(130+b)]
# 0___________________________________________________
pts1 = np.float32([[1614, 876], [1523, 681], [430, 872], [488, 687]])  # ([[1536, 1024], [1520, 790], [561, 942], [539, 769]])
pts2 = np.float32([p1, p2, p3, p4])
M0 = cv.getPerspectiveTransform(pts1, pts2)
# 1___________________________________________________
pts1 = np.float32([[179, 349], [37, 527], [1599, 612], [1563, 392]])
pts2 = np.float32([p7, p8, p3, p4])
M1 = cv.getPerspectiveTransform(pts1, pts2)
# 2___________________________________________________
pts1 = np.float32([[1754, 362], [1893, 540], [284, 664], [321, 426]])
pts2 = np.float32([p5, p6, p1, p2])
M2 = cv.getPerspectiveTransform(pts1, pts2)
# 变换后的图片大小+位置调整=变换后的图片裁剪的效果
# 3___________________________________________________
pts1 = np.float32([[405, 691], [291, 824], [1489, 648], [1602, 769]])
pts2 = np.float32([p5, p6, p7, p8])
M3 = cv.getPerspectiveTransform(pts1, pts2)
# 拼接参数
mask = np.zeros((H, H, 3), np.uint8)  # 注意是否3
y1, y2 = H / 2 + 100, H / 2 - 100
p_1 = [int((p2[0] - p1[0]) * (y1 - p1[1]) / (p2[1] - p1[1]) + p1[0]), int(y1)]  # 求交点
p_2 = [int((p2[0] - p1[0]) * (H - p1[1]) / (p2[1] - p1[1]) + p1[0]), int(H)]  # 求交点
p_3 = [int((p4[0] - p3[0]) * (y1 - p3[1]) / (p4[1] - p3[1]) + p3[0]), int(y1)]  # 求交点
p_4 = [int((p4[0] - p3[0]) * (H - p3[1]) / (p4[1] - p3[1]) + p3[0]), int(H)]  # 求交点
p_6 = [int((p6[0] - p5[0]) * (y2 - p5[1]) / (p6[1] - p5[1]) + p5[0]), int(y2)]  # 求交点
p_5 = [int((p6[0] - p5[0]) * (0 - p5[1]) / (p6[1] - p5[1]) + p5[0]), int(0)]  # 求交点
p_8 = [int((p8[0] - p7[0]) * (y2 - p7[1]) / (p8[1] - p7[1]) + p7[0]), int(y2)]  # 求交点
p_7 = [int((p8[0] - p7[0]) * (0 - p7[1]) / (p8[1] - p7[1]) + p7[0]), int(0)]  # 求交点
pts_1 = np.array([p_1, p_2, p_4, p_3])
pts_2 = np.array([p_5, p_6, p_8, p_7])
cv.drawContours(mask, [pts_1], -1, (255, 255, 255), -1, cv.LINE_AA)
cv.drawContours(mask, [pts_2], -1, (255, 255, 255), -1, cv.LINE_AA)
mask = cv.GaussianBlur(mask, (59, 59), sigmaX=5)
inv_mask = cv.bitwise_not(mask)

# 雷达——环视标定矩阵
M = np.array([-1.33939651e-04, -1.99756968e-02, 6.31833120e+00, -2.02637082e-02, 5.23257834e-05, 6.08154698e+00, 2.12819395e-05, -1.00975669e-05, 9.96769085e-01]).reshape((3, 3))
thresh = 0.05
# timer = Timer()
cv_bridge = CvBridge()
#  消息定义
slots_publisher = rospy.Publisher('slots_lists', PointCloud2, queue_size=1)
msg = PointCloud2()
msg.fields = [
    PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1)]
msg.is_bigendian = False
msg.point_step = 12
msg.is_dense = False
img_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
image = Image()
image.height = H
image.width = H
image.encoding = 'rgb8'
image.step = 600 * 3


def plot_points(image, pred_points):
    """ Plot marking points on the image. """
    if not pred_points:
        return
    height = image.shape[0]  # 高宽像素值
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        p0_x = width * marking_point.x - 0.5  # 为什么-0.5？
        p0_y = height * marking_point.y - 0.5
        cos_val = math.cos(marking_point.direction)
        sin_val = math.sin(marking_point.direction)
        p1_x = p0_x + 50 * cos_val
        p1_y = p0_y + 50 * sin_val
        p2_x = p0_x - 50 * sin_val
        p2_y = p0_y + 50 * cos_val
        p3_x = p0_x + 50 * sin_val
        p3_y = p0_y - 50 * cos_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))  # 四舍五入
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(confidence), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if marking_point.shape > 0.5:
            cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)


def plot_slots(image, pred_points, slots):
    if not pred_points or not slots:
        return
    marking_points = list(list(zip(*pred_points))[1])  # list元组转化列表
    height = image.shape[0]
    width = image.shape[1]
    points_list = np.ones((len(slots), 4, 3))
    i = 0
    for slot in slots:
        point_a = marking_points[slot[0]]
        point_b = marking_points[slot[1]]
        p0_x = width * point_a.x - 0.5
        p0_y = height * point_a.y - 0.5
        p1_x = width * point_b.x - 0.5
        p1_y = height * point_b.y - 0.5
        vec = np.array([p1_x - p0_x, p1_y - p0_y])
        vec = vec / np.linalg.norm(vec)  # 单位矢量
        distance = calc_point_squre_dist(point_a, point_b)
        if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:
            separating_length = config.LONG_SEPARATOR_LENGTH
        elif config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST:
            separating_length = config.SHORT_SEPARATOR_LENGTH
        p2_x = p0_x + height * separating_length * vec[1]
        p2_y = p0_y - width * separating_length * vec[0]
        p3_x = p1_x + height * separating_length * vec[1]
        p3_y = p1_y - width * separating_length * vec[0]
        # 转换到激光雷达坐标系 0、1是入口两个点；2、3是后两个点
        A = np.array([[p0_x], [p0_y], [1]])
        B = np.dot(M, A)
        # print(i, B, len(slots))
        # print(points_list)
        points_list[i][0][0] = B[0] / B[2]
        points_list[i][0][1] = B[1] / B[2]
        points_list[i][0][2] = -2.05642
        A = np.array([[p1_x], [p1_y], [1]])
        B = np.dot(M, A)
        points_list[i][1][0] = B[0] / B[2]
        points_list[i][1][1] = B[1] / B[2]
        points_list[i][1][2] = -2.05642
        A = np.array([[p2_x], [p2_y], [1]])
        B = np.dot(M, A)
        points_list[i][2][0] = B[0] / B[2]
        points_list[i][2][1] = B[1] / B[2]
        points_list[i][2][2] = -2.05642
        A = np.array([[p3_x], [p3_y], [1]])
        B = np.dot(M, A)
        points_list[i][3][0] = B[0] / B[2]
        points_list[i][3][1] = B[1] / B[2]
        points_list[i][3][2] = -2.05642
        i += 1
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)
    return points_list  # 返回一个list：停车位数量*4*2


def preprocess_image(image):
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_marking_points(detector, image, thresh, device):
    """Given image read from opencv, return detected marking points."""
    prediction = detector(preprocess_image(image).to(device))  # [1, 6, 16, 16]
    return get_predicted_points(prediction[0], thresh)


def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):  # 两两分析
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            # print(distance)
            if not (config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST
                    or config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST):
                continue
            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j):  # 通过第三个点
                continue
            result = pair_marking_points(point_i, point_j)
            if result == 1:
                slots.append((i, j))
            elif result == -1:
                slots.append((j, i))
    return slots  #


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


def detect_image(detector, device):
    rospy.init_node('slots_pub_node', anonymous=True)  # 不关心节点唯一性
    msg.header.frame_id = "rslidar"
    image.header.frame_id = 'map'
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time().now()
        ct = 400000000  # 调整的时间
        # 车位超前（车上的改动）
        if msg.header.stamp.nsecs > ct:
            msg.header.stamp.nsecs -= ct  # 手动calibrate time 0.4s
        else:
            msg.header.stamp.secs -= 1
            msg.header.stamp.nsecs += (1000000000-ct)
        # 车位滞后
        #if msg.header.stamp.nsecs + ct < 1000000000:
        #    msg.header.stamp.nsecs += ct  # 手动calibrate time 0.4s
        #else:
        #    msg.header.stamp.secs += 1
        #    msg.header.stamp.nsecs = msg.header.stamp.nsecs + ct - 1000000000
        image.header.stamp = msg.header.stamp
        # 图像处理
        timer.tic()
        img10 = input1.read()
        # cv.imshow('', img10)
        # cv.waitKey(1)
        img20 = input2.read()
        img30 = input3.read()
        img00 = input0.read()
        # timer.toc()
        # 开始处理##############################################################################
        ''' 去畸变============================================================'''
        img0 = cv.remap(img00, mapx2, mapy2, cv.INTER_LINEAR)  # mapx2直一点
        img1 = cv.remap(img10, mapx, mapy, cv.INTER_LINEAR)
        img2 = cv.remap(img20, mapx2, mapy2, cv.INTER_LINEAR)
        img3 = cv.remap(img30, mapx2, mapy2, cv.INTER_LINEAR)
        ''' 透视变换==========================================================='''
        img0 = cv.warpPerspective(img0, M0, (H, H))
        img1 = cv.warpPerspective(img1, M1, (H, H))
        img2 = cv.warpPerspective(img2, M2, (H, H))
        img3 = cv.warpPerspective(img3, M3, (H, H))
        '''拼接==========================================================='''
        img0[int(H / 2) - 150:int(H / 2) + 100, :] = [0, 0, 0]
        img0[0:int(H / 2) - 150, :] = img3[0:int(H / 2) - 150, :]
        img2[:, int(H / 2):H] = img1[:, int(H / 2):H]
        img5 = np.zeros(img0.shape, img0.dtype)
        for i in range(0, img0.shape[0]):
            img5[i] = mask[i] / 255 * img0[i] + inv_mask[i] / 255 * img2[i]
        img50 = cv.cvtColor(img5, cv.COLOR_BGR2RGB)
        image.data = np.array(img50).tobytes()
        img_pub.publish(image)
        gamma_table = [np.power(x / 255.0, 0.4) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        img5 = cv.LUT(img5, gamma_table)
        # 车位检测
        pred_points = detect_marking_points(detector, img5, thresh, device)
        slots = None
        if pred_points:
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)  # 推理框
        plot_points(img5, pred_points)  # 画点
        points_list = plot_slots(img5, pred_points, slots)  # 画框
        cv.imshow('', img5)
        cv.waitKey(1)
        if slots:
            # 获取坐标变换
            # points_list_ = points_list
            # p2map = 0
            # t_czk = rospy.Time(0)  # rospy.Time().now()不行   包的时间92
            # if p2map:  # 是否变换到全局坐标
            #     # 订阅tf
            #     msg.header.frame_id = "map"
            #     tf_listener = tf.TransformListener()  # TransformListener创建后就开始接受tf广播信息，最多可以缓存10s
            #     tf_listener.waitForTransform('map', 'rslidar', t_czk, rospy.Duration(1))
            #     tran, rot = tf_listener.lookupTransform('map', 'rslidar', rospy.Time())
            #     # 开始转换
            #     M_ = np.dot(translation_matrix(tran), quaternion_matrix(rot))
            #     for i in range(points_list.shape[0]):
            #         for j in range(4):
            #             [points_list_[i][j][0], points_list_[i][j][1], points_list_[i][j][2]] = np.dot(M_, np.array(
            #                 [points_list[i][j][0], points_list[i][j][1], points_list[i][j][2], 1.0]))[:3]
            ''' 发布一个point_list  '''
            msg.height = points_list.shape[1]
            msg.width = points_list.shape[0]
            msg.row_step = msg.point_step * points_list.shape[0]
            msg.data = np.asarray(points_list, np.float32).tobytes()
            slots_publisher.publish(msg)


def inference_detector():
    """Inference demo of directional point detector."""
    device = torch.device('cuda:' + '0' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)
    dp_detector = DirectionalPointDetector(
        3, 32, config.NUM_FEATURE_MAP_CHANNEL).to(device)  # 检测器
    dp_detector.load_state_dict(torch.load('/home/nvidia/users/chenzhikai/czk_ws/src/slots_pub/scripts/DMPR/dmpr_pretrained_weights.pth'))  # src/slots_pub/scripts/DMPR/
    dp_detector.eval()
    detect_image(dp_detector, device)


if __name__ == '__main__':
    inference_detector()
