#!/usr/bin/env python3

# 查看前视摄像头，并保存图像（用于与激光雷达标定）
import cv2 as cv
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

n = 0
camera3 = cv.VideoCapture(3)
mtx2 = np.array(
    [[474.11482367731435, 0.0, 970.9422789169496], [0.0, 475.0627595972449, 554.2074603636624], [0.0, 0.0, 1.0]])
dist2 = np.array([[0.058049435249809975], [-0.0036229297441117206], [-0.004129058798634489], [0.00097655626676563]])
DIM = (1920, 1080)
mapx2, mapy2 = cv.fisheye.initUndistortRectifyMap(mtx2, dist2, np.eye(3), mtx2, DIM, cv.CV_16SC2)

rospy.init_node('img_publisher', anonymous=True)  # 不关心节点唯一性
#  创建一个Publisher，发布名为/camera/image_raw的topic，消息类型为Image，队列长度10
img_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
image = Image()
image.header.frame_id = 'map'
image.encoding = 'rgb8'


while not rospy.is_shutdown():
    image.header.stamp = rospy.Time.now()
    ret0, input3 = camera3.read()
    input3 = cv.remap(input3, mapx2, mapy2, cv.INTER_LINEAR)
    cv.imshow('input3', input3)
    if cv.waitKey(1) == ord('s'):  # 等待用户触发事件,等待时间为1ms
        cv.imwrite(f'img_cali2/{n}.png', input3)  # {path}/
        print(f'{n} saved!')
        n += 1
    input3 = cv.cvtColor(input3, cv.COLOR_BGR2RGB)
    image.height = input3.shape[0]
    image.width = input3.shape[1]
    image.step = image.width * 3
    image.data = np.array(input3).tobytes()
    img_pub.publish(image)