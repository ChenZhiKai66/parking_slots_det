# parking_slots_det

## Pipeline

![image-20220830222707548](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20220830222707548.png)

1. 投影变换需要手动选取标定点
2. 图像拼接采用外部标定点空间一致性的方法
3. 停车位检测模型follow的是DMPR https://github.com/Teoge/DMPR-PS
4. 环视图像和激光雷达标定简化为求解一个3*3单应性矩阵
5. 转换为地图坐标需要接收定位模块发送的位置信息，可以进行多帧结果融合 

## Requirements

* PyTorch
* CUDA (optional)
* Other requirements  
    `pip install -r requirements.txt`

## Pre-trained weights

The [pre-trained weights](https://drive.google.com/open?id=1OuyF8bGttA11-CKJ4Mj3dYAl5q4NL5IT) could be used to reproduce the number in the paper.

## Inference

* img_slots.py：读取环视图像视频流，输出停车位雷达坐标msg

- odom_slots_map_m.py：接收停车位雷达坐标msg、全局定位msg，将停车位从局部坐标转为全局坐标

```
roslaunch slots_pub slots_pub.launch
```

