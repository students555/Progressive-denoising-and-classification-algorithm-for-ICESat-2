# -*- coding: utf-8 -*-
"""
@Time: 2026/3/10 15:47
@Author: LXX
@File: main.py
@Description: ICESat-2 光子点云去噪流程，三步去噪
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

# 导入自定义模块
from dp import dp
from process_segments import processSegments
from elliptical_clustering import Ellipticalclustering

# ================== 设置全局绘图样式==================
plt.rcParams['font.sans-serif'] = ['SimHei']      # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'         # 字体族
plt.rcParams['font.size'] = 12                      # 基础字体大小
plt.rcParams['axes.titlesize'] = 14                 # 标题大小
plt.rcParams['axes.labelsize'] = 12                 # 坐标轴标签大小
plt.rcParams['xtick.labelsize'] = 10                # X轴刻度大小
plt.rcParams['ytick.labelsize'] = 10                # Y轴刻度大小
plt.rcParams['legend.fontsize'] = 10                # 图例大小
plt.rcParams['lines.linewidth'] = 1.5               # 线条宽度
plt.rcParams['grid.linestyle'] = '--'               # 网格线样式
plt.rcParams['grid.alpha'] = 0.3                    # 网格线透明度
# =================================================================


def main():
    print("开始执行去噪流程...")

    # ================== 配置列名（根据实际Excel表头修改）==================
    DIST_COL = 'Along-track distance'   # 沿轨距离列名
    ELEV_COL = 'Elevation'              # 高程列名
    # =====================================================================

    # -------------------------------------------------------------------------
    # 1. 基于多阈值策略的孤立噪声光子去除
    # -------------------------------------------------------------------------
    print(">> 阶段 1: 孤立噪声移除")

    # 读取Excel文件，获取所有数据
    df = pd.read_excel('Sample1_tbl.xlsx')

    # 只提取需要的两列，并转换为numpy数组（N×2）
    data = df[[DIST_COL, ELEV_COL]].values

    # 构建 KDTree 并进行 KNN 搜索 (k=55, 第0个近邻是自身)
    tree = cKDTree(data)
    dist, idx = tree.query(data, k=55)

    # 计算除去自身的后54个邻居的平均距离
    mdist = np.mean(dist[:, 1:], axis=1)

    dynamic_thresh = np.zeros_like(mdist)
    window_size = 100
    step_size = 50

    # 存储每个窗口的阈值，用于后续可视化
    window_centers = []
    window_thresholds = []

    for i in range(0, len(mdist), step_size):
        end_idx = min(i + window_size, len(mdist))
        local_mdist = mdist[i:end_idx]

        # 大津法要求数组必须有不同的值
        if len(np.unique(local_mdist)) > 1:
            local_thresh = threshold_otsu(local_mdist)
        else:
            local_thresh = local_mdist[0]

        dynamic_thresh[i:end_idx] = local_thresh

        # 记录窗口中心位置和阈值，用于后续绘图
        window_center = (data[i, 0] + data[min(end_idx-1, len(data)-1), 0]) / 2
        window_centers.append(window_center)
        window_thresholds.append(local_thresh)

    # 过滤出小于动态距离阈值的点
    clean_data_mask = mdist < dynamic_thresh
    clean_data = data[clean_data_mask, :]

    # --- 阶段1主可视化（原有）---
    plt.figure(figsize=(12, 5))
    # 左图：原始数据
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], s=2, c='blue', label='原始光子')
    plt.title(f'原始数据（总计 {len(data)} 点）')
    plt.xlabel('沿轨距离 (m)')
    plt.ylabel('高程 (m)')
    plt.legend(loc='upper right')
    plt.grid(True)
    # 右图：阈值曲线与保留点
    plt.subplot(1, 2, 2)
    plt.plot(data[:, 0], mdist, '.', markersize=2, label='平均距离', alpha=0.5, color='gray')
    plt.plot(data[:, 0], dynamic_thresh, 'r-', linewidth=1.5, label='动态阈值')
    plt.fill_between(data[:, 0], 0, dynamic_thresh, where=(mdist < dynamic_thresh),
                     color='green', alpha=0.2, label='保留点区域')
    plt.fill_between(data[:, 0], dynamic_thresh, np.max(mdist), where=(mdist >= dynamic_thresh),
                     color='gray', alpha=0.2, label='去除点区域')
    plt.title(f'阶段1：孤立噪声去除（保留 {len(clean_data)} 点）')
    plt.xlabel('沿轨距离 (m)')
    plt.ylabel('54邻域平均距离')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 阶段1新增可视化：平均距离直方图与阈值 ---
    plt.figure(figsize=(8, 5))
    plt.hist(mdist, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='平均距离分布')
    plt.axvline(x=np.mean(dynamic_thresh), color='red', linestyle='--', linewidth=2, label='平均动态阈值')
    plt.axvline(x=np.median(dynamic_thresh), color='orange', linestyle='--', linewidth=2, label='中位数动态阈值')
    plt.xlabel('54邻域平均距离')
    plt.ylabel('频数')
    plt.title('平均距离分布与动态阈值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    # -------------------

    # -------------------------------------------------------------------------
    # 2. 地形坡度的自适应计算以及低密度聚集噪声光子去除
    # -------------------------------------------------------------------------
    print(">> 阶段 2: 坡度自适应计算与低密度聚集噪声去除")
    max_distance = np.max(clean_data[:, 0])
    interval_length = 50
    num_intervals = int(np.ceil(max_distance / interval_length))
    interval_boundaries = np.arange(0, num_intervals + 1) * interval_length

    # 根据沿轨距离划分区间
    interval_indices = np.digitize(clean_data[:, 0], interval_boundaries, right=True)

    signal_centers = []
    # 存储每个区间的高程直方图数据，用于后续可视化
    interval_hist_data = []  # 每个元素为 (bins, hist, center_elevation)

    # 提取每个区间的核心光子
    for i in range(1, num_intervals + 1):
        current_interval_data = clean_data[interval_indices == i, :]
        if len(current_interval_data) == 0:
            interval_hist_data.append(None)
            continue

        min_elev = np.floor(np.min(current_interval_data[:, 1]))
        max_elev = np.ceil(np.max(current_interval_data[:, 1]))

        if min_elev == max_elev:
            max_elev += 1

        elevation_bins = np.arange(min_elev, max_elev + 1, 1)
        elevation_hist, _ = np.histogram(current_interval_data[:, 1], bins=elevation_bins)

        # 寻找众数高程
        max_idx = np.argmax(elevation_hist)
        center_elevation = elevation_bins[max_idx]

        # 存储直方图数据
        interval_hist_data.append((elevation_bins[:-1], elevation_hist, center_elevation))

        # 寻找距离中心高程最近的点
        min_dist_idx = np.argmin(np.abs(current_interval_data[:, 1] - center_elevation))
        signal_center = current_interval_data[min_dist_idx, :]
        signal_centers.append(signal_center)

    signal_centers = np.array(signal_centers)

    # Douglas-Peucker 算法合并相似地形段
    pntSet = signal_centers[:, :2]
    nPntSet = dp(pntSet, 0.5)

    # 计算合并后地形的坡度（角度）
    points = nPntSet
    num_points = points.shape[0]
    angles = np.zeros(num_points - 1)

    for i in range(num_points - 1):
        delta_x = points[i + 1, 0] - points[i, 0]
        delta_y = points[i + 1, 1] - points[i, 1]
        angles[i] = np.rad2deg(np.arctan2(delta_y, delta_x))

    angles = np.mod(angles, 180)
    # 首尾拼接，保证段数一致
    angles = np.concatenate(([angles[0]], angles, [angles[-1]]))

    # 分配角度至每一段
    split_distances = nPntSet[:, 0]
    split_indices = np.zeros(len(split_distances), dtype=int)
    for i in range(len(split_distances)):
        split_indices[i] = np.argmin(np.abs(clean_data[:, 0] - split_distances[i]))

    segments = []
    for i in range(len(split_distances) + 1):
        # 注意: 此处为了模拟 MATLAB 代码中的重复包含边界行为，切片做了 +1
        if i == 0:
            indices = slice(0, split_indices[0] + 1)
        elif i == len(split_distances):
            indices = slice(split_indices[-1], len(clean_data))
        else:
            indices = slice(split_indices[i - 1], split_indices[i] + 1)

        segment_data = clean_data[indices].copy()
        # 拼接对应的角度列
        angle_col = np.full((segment_data.shape[0], 1), angles[i])
        segments.append(np.hstack((segment_data, angle_col)))

    # 将拆分的段纵向合并回完整的大矩阵
    bigMatrix = np.vstack(segments)

    # 基于椭圆连通性迭代生长算法
    a = 24
    b = a / 6
    E, I = processSegments(bigMatrix, a, b)
    minpts = 8
    IDX, isnoise, signalPoints, noisePoints = Ellipticalclustering(E, I, minpts, bigMatrix)

    # --- 阶段2主可视化---
    plt.figure(figsize=(12, 5))
    # 左图：地形分段核心点与DP简化线
    plt.subplot(1, 2, 1)
    plt.scatter(clean_data[:, 0], clean_data[:, 1], s=1, c='lightgray', label='阶段1保留点', alpha=0.6)
    plt.plot(signal_centers[:, 0], signal_centers[:, 1], 'bo', markersize=4, label='区间核心点')
    plt.plot(nPntSet[:, 0], nPntSet[:, 1], 'r-', linewidth=2, label='DP简化地形')
    plt.title('阶段2：地形分段与坡度计算')
    plt.xlabel('沿轨距离 (m)')
    plt.ylabel('高程 (m)')
    plt.legend(loc='upper right')
    plt.grid(True)
    # 右图：聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(signalPoints[:, 0], signalPoints[:, 1], s=2, c='red', label='信号点（聚类）')
    plt.scatter(noisePoints[:, 0], noisePoints[:, 1], s=1, c='gray', label='噪声点')
    plt.title(f'阶段2：椭圆邻域聚类结果（信号 {len(signalPoints)} 点，噪声 {len(noisePoints)} 点）')
    plt.xlabel('沿轨距离 (m)')
    plt.ylabel('高程 (m)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 阶段2可视化---
    num_to_show = min(5, len(interval_hist_data))
    if num_to_show > 0:
        fig, axes = plt.subplots(1, num_to_show, figsize=(4*num_to_show, 4))
        if num_to_show == 1:
            axes = [axes]
        for idx, ax in enumerate(axes):
            hist_data = interval_hist_data[idx]
            if hist_data is None:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center')
                ax.set_title(f'区间 {idx+1}')
                continue
            bins, hist, center_elev = hist_data
            ax.bar(bins, hist, width=1.0, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=center_elev, color='red', linestyle='--', linewidth=2, label=f'众数 = {center_elev:.1f}')
            ax.set_xlabel('高程 (m)')
            ax.set_ylabel('频数')
            ax.set_title(f'区间 {idx+1} 高程直方图')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
        plt.suptitle('典型区间高程分布与核心点选择', y=1.02)
        plt.tight_layout()
        plt.show()
    # -------------------

    # -------------------------------------------------------------------------
    # 3. 基于箱线图分析的外部聚集噪声光子去除
    # -------------------------------------------------------------------------
    print(">> 阶段 3: 外部聚集噪声去除并制图输出")
    data_signal = signalPoints
    window_size_box = 150
    min_dist = np.min(data_signal[:, 0])
    max_dist = np.max(data_signal[:, 0])

    cleaned_data = []
    outliers_data = []

    # 存储每个窗口的统计量，用于后续可视化
    window_stats = []  # 每个元素为 (start, end, Lowerlimit, Upperlimit, count)

    # 沿轨移动窗口进行箱线图清洗
    start_dist = min_dist
    # 用于过程显示的窗口示例（取中间一个窗口画箱线图）
    example_window = None
    example_window_bounds = None
    while start_dist <= max_dist:
        end_dist = start_dist + window_size_box
        window_indices = (data_signal[:, 0] >= start_dist) & (data_signal[:, 0] < end_dist)
        window_photons = data_signal[window_indices, :]

        if len(window_photons) > 0:
            Q1 = np.percentile(window_photons[:, 1], 25)
            Q3 = np.percentile(window_photons[:, 1], 75)
            IQR = Q3 - Q1

            Upperlimit = Q3 + 3 * IQR
            Lowerlimit = Q1 - 3 * IQR

            non_outliers = (window_photons[:, 1] >= Lowerlimit) & (window_photons[:, 1] <= Upperlimit)
            cleaned_data.append(window_photons[non_outliers, :])
            outliers_data.append(window_photons[~non_outliers, :])

            # 保存统计量
            window_stats.append((start_dist, end_dist, Lowerlimit, Upperlimit, len(window_photons)))

            # 保存一个中间窗口用于过程显示（例如窗口中心附近）
            if example_window is None and len(window_photons) > 20:
                example_window = window_photons
                example_window_bounds = (Lowerlimit, Upperlimit)

        start_dist += window_size_box

    # 处理空列表
    if len(cleaned_data) > 0:
        cleaned_data = np.vstack(cleaned_data)
    else:
        cleaned_data = np.empty((0, data_signal.shape[1]))

    if len(outliers_data) > 0:
        outliers_combined = np.vstack(outliers_data)
    else:
        outliers_combined = np.empty((0, data_signal.shape[1]))

    # --- 阶段3主可视化---
    plt.figure(figsize=(12, 5))
    # 左图：清洗前的信号点
    plt.subplot(1, 2, 1)
    plt.scatter(data_signal[:, 0], data_signal[:, 1], s=2, c='blue', label='箱线图前信号点')
    if example_window is not None:
        xmin, xmax = example_window[0, 0], example_window[-1, 0]
        plt.axhline(y=example_window_bounds[0], color='orange', linestyle='--', linewidth=1.5, label='下限')
        plt.axhline(y=example_window_bounds[1], color='orange', linestyle='--', linewidth=1.5, label='上限')
        plt.axvspan(xmin, xmax, alpha=0.2, color='yellow', label='示例窗口')
    plt.title('阶段3：箱线图清洗前')
    plt.xlabel('沿轨距离 (m)')
    plt.ylabel('高程 (m)')
    plt.legend(loc='upper right')
    plt.grid(True)
    # 右图：清洗后的信号与离群点
    plt.subplot(1, 2, 2)
    if len(cleaned_data) > 0:
        plt.scatter(cleaned_data[:, 0], cleaned_data[:, 1], s=2, c='red', label='最终信号点')
    if len(outliers_combined) > 0:
        plt.scatter(outliers_combined[:, 0], outliers_combined[:, 1], s=2, c='orange', label='箱线图离群点')
    plt.title(f'阶段3：箱线图清洗后（信号 {len(cleaned_data)} 点，离群 {len(outliers_combined)} 点）')
    plt.xlabel('沿轨距离 (m)')
    plt.ylabel('高程 (m)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 阶段3可视化：窗口上下限及点数 ---
    if len(window_stats) > 0:
        stats_array = np.array(window_stats)
        starts = stats_array[:, 0]
        ends = stats_array[:, 1]
        lowers = stats_array[:, 2]
        uppers = stats_array[:, 3]
        counts = stats_array[:, 4]
        window_centers = (starts + ends) / 2

        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:red'
        ax1.set_xlabel('沿轨距离 (m)')
        ax1.set_ylabel('高程上下限 (m)', color=color)
        ax1.plot(window_centers, lowers, 'o-', color=color, markersize=4, label='下限')
        ax1.plot(window_centers, uppers, 's-', color=color, markersize=4, label='上限')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('窗口内点数', color=color)
        ax2.bar(window_centers, counts, width=window_size_box*0.8, color=color, alpha=0.3, label='窗口点数')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        plt.title('箱线图窗口统计：上下限与点数')
        fig.tight_layout()
        plt.show()
    # -------------------

    # 打印统计
    print("\n=== 去噪过程统计 ===")
    print(f"原始数据点数: {len(data)}")
    print(f"阶段1后点数: {len(clean_data)}")
    print(f"阶段2信号点数: {len(signalPoints)}  阶段2噪声点数: {len(noisePoints)}")
    print(f"阶段3信号点数: {len(cleaned_data)}  阶段3离群点数: {len(outliers_combined)}")

    # -------------------------------------------------------------------------
    # 最终分类结果
    # -------------------------------------------------------------------------
    # 重新读取原始Excel（包含所有列）
    lable_df = pd.read_excel('Sample1_tbl.xlsx')
    # 提取距离和高程列用于匹配
    lable_coords = lable_df[[DIST_COL, ELEV_COL]].values
    label_status = np.zeros(len(lable_coords))

    # 使用最终清洗后的信号点
    signal = cleaned_data if len(cleaned_data) > 0 else signalPoints
    signal_set = set(tuple(x) for x in signal[:, :2])   # signal已经是两列

    for i in range(len(lable_coords)):
        if tuple(lable_coords[i]) in signal_set:
            label_status[i] = 1

    # 合并标签列，最后一列为标签
    lable = np.column_stack((lable_df.values, label_status))

    # 可视化
    plt.figure(figsize=(10, 6))
    signal_mask = lable[:, -1] == 1
    noise_mask = lable[:, -1] == 0

    plt.scatter(lable[signal_mask, 0], lable[signal_mask, 1], s=4, c='red', label='信号光子', marker='.')
    plt.scatter(lable[noise_mask, 0], lable[noise_mask, 1], s=2, c='green', label='噪声光子', marker='.')

    plt.xlabel('沿轨距离 (m)', fontsize=12)
    plt.ylabel('高程 (m)', fontsize=12)
    plt.title('最终分类结果', fontsize=14)
    plt.legend(prop={'size': 10})
    plt.xlim([0, 1500])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()