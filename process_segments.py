# -*- coding: utf-8 -*-
"""
# -*- coding: utf-8 -*-
@Time: 2026/3/10 15:46
@Author: LXX
@File: process_segments.py
@IDE：PyCharm
@Motto：ABC(Always Be Coding)
"""
import numpy as np


def processSegments(segments, a, b):
    """
    处理信号段，计算每个点基于方向自适应椭圆的邻域特征
    """
    numPoints = segments.shape[0]

    # 初始化计数数组 E 和邻域索引列表 I
    E = np.zeros(numPoints, dtype=int)
    I = [[] for _ in range(numPoints)]

    # 提取所有点的坐标以供矩阵化快速计算
    all_x = segments[:, 0]
    all_y = segments[:, 1]

    # 遍历每个光子点
    for i in range(numPoints):
        xi = segments[i, 0]
        yi = segments[i, 1]

        # 获取当前点的旋转角度并转为弧度
        rotationAngle = np.deg2rad(segments[i, 2])
        cos_theta = np.cos(-rotationAngle)  # 注意这里用负角度，相当于将其他点反向旋转到以该点为正交的坐标系下
        sin_theta = np.sin(-rotationAngle)

        # 计算所有点相对于当前中心点的坐标差
        dx = all_x - xi
        dy = all_y - yi

        # 通过逆旋转矩阵，将全局坐标系的相对距离转化至椭圆主轴坐标系下
        dx_rot = dx * cos_theta - dy * sin_theta
        dy_rot = dx * sin_theta + dy * cos_theta

        # 判断点是否在椭圆内： (X^2 / a^2) + (Y^2 / b^2) <= 1
        withinEllipse = ((dx_rot ** 2) / (a ** 2) + (dy_rot ** 2) / (b ** 2)) <= 1

        # 计算椭圆内部点的数量 (包含自身)
        count = np.sum(withinEllipse)
        E[i] = count

        # 寻找处于邻域内的点索引，存储于列表 I 中
        neighborIndices = np.where(withinEllipse)[0]
        I[i] = neighborIndices.tolist()

    return E, I