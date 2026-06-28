# -*- coding: utf-8 -*-
"""
# -*- coding: utf-8 -*-
@Time: 2026/3/10 15:45
@Author: LXX
@File: dp.py
@IDE：PyCharm
@Motto：ABC(Always Be Coding)
"""
import numpy as np


def dp(pntSet, TH):
    """
    dp - Douglas-Peucker (道格拉斯-普克) 算法用于简化折线

    参数:
        pntSet : Nx2 numpy数组，表示折线的2D数据点
        TH     : 距离阈值
    返回:
        nPntSet: Mx2 numpy数组，表示简化后的折线
    """
    pntSet = np.array(pntSet)
    if len(pntSet) < 2:
        return pntSet

    # 向量化操作：计算所有点到首尾两点连线的垂直距离
    # 首尾连线向量 vertV
    vertV = np.array([pntSet[-1, 1] - pntSet[0, 1], -pntSet[-1, 0] + pntSet[0, 0]])
    norm_vertV = np.linalg.norm(vertV)

    if norm_vertV == 0:
        baseL = np.zeros(len(pntSet))
    else:
        # 点到直线距离公式的向量化实现
        baseL = np.abs(np.sum((pntSet - pntSet[0, :]) * (vertV / norm_vertV), axis=1))

    if np.max(baseL) < TH:
        # 如果最大距离小于阈值，则仅保留首尾两个点
        nPntSet = np.array([pntSet[0, :], pntSet[-1, :]])
    else:
        # 如果最大距离大于阈值，则在距离最大的点处分为左右两支，并进行递归
        maxPos = np.argmax(baseL)

        # 递归调用
        L_PntSet = dp(pntSet[:maxPos + 1, :], TH)
        R_PntSet = dp(pntSet[maxPos:, :], TH)

        # 合并结果，注意去掉 R_PntSet 的第一个点以免重复（即原最大距离点）
        nPntSet = np.vstack((L_PntSet, R_PntSet[1:, :]))

    return nPntSet