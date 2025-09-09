import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def correct_steep_drops_cumulatively(input_filepath, output_filepath=None, plot_filepath=None, drop_threshold=-0.001):
    """
    通过“累积补偿”的方式，修正数据中由于陡降造成的问题。
    它会一次性计算所有陡降的累积效应，然后统一应用补偿。

    Args:
        input_filepath (str): 输入的CSV文件路径。
        output_filepath (str, optional): 保存修正后数据的CSV文件路径。
        plot_filepath (str, optional): 保存修正方案对比图的路径。
        drop_threshold (float, optional): 定义陡降的阈值 (V)。
                                        例如, -0.001 表示下降超过0.001V。
    """
    # --- 1. 文件和路径设置 ---
    if not os.path.exists(input_filepath):
        print(f"错误：文件未找到 '{input_filepath}'")
        return

    if output_filepath is None:
        base, ext = os.path.splitext(input_filepath)
        output_filepath = f"{base}_restored.csv"
    if plot_filepath is None:
        plot_filepath = "drop_correction_visualization.png"

    # --- 2. 读取数据 ---
    try:
        df = pd.read_csv(input_filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_filepath, encoding='gbk')
    print(f"--- 正在处理文件 (累积补偿模式): {input_filepath} ---")

    # --- 3. 核心修正逻辑 ---
    df_corrected = df.copy()
    corrected_groups = []
    detected_drops = {}

    for cycle_num, group in df.groupby('循环号'):
        group_copy = group.copy()
        # 基于原始数据计算电压差
        diffs = group_copy['弛豫段电压'].diff()

        # 基于原始数据找到所有满足陡降条件的点
        drop_indices = diffs[diffs < drop_threshold].index

        if not drop_indices.empty:
            detected_drops[cycle_num] = [{'index': idx, 'drop_value': diffs.loc[idx]} for idx in drop_indices]

            # 创建一个补偿序列，并在陡降点位置填入需要补偿的正值 (-diffs)
            compensations = pd.Series(0.0, index=group.index)
            compensations.loc[drop_indices] = -diffs.loc[drop_indices]

            # 计算累积补偿值
            cumulative_compensation = compensations.cumsum()

            # 将这个最终的补偿向量一次性地加到原始电压上
            group_copy['弛豫段电压'] = group_copy['弛豫段电压'] + cumulative_compensation + 0.0001

            # 新增代码：仅在突变点本身额外加上 0.001
            group_copy.loc[drop_indices, '弛豫段电压'] += 0.00

        corrected_groups.append(group_copy)

    df_corrected = pd.concat(corrected_groups)

    # --- 4. 保存恢复后的数据 ---
    df_corrected.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    print(f"--- 数据恢复完成。已保存至: {output_filepath} ---\n")

    # --- 5. 可视化恢复效果 ---
    cycle_to_plot = next(iter(detected_drops), None)  # 选取第一个有陡降的循环进行绘图
    if cycle_to_plot:
        plt.figure(figsize=(12, 7))
        original_cycle = df[df['循环号'] == cycle_to_plot]
        corrected_cycle = df_corrected[df_corrected['循环号'] == cycle_to_plot]

        plt.plot(original_cycle.index, original_cycle['弛豫段电压'], label=f'原始数据 (循环 {cycle_to_plot})',
                 alpha=0.8)
        plt.plot(corrected_cycle.index, corrected_cycle['弛豫段电压'], label=f'恢复后数据 (循环 {cycle_to_plot})',
                 linestyle='--', color='red', linewidth=2)
        plt.title(f'循环 {cycle_to_plot} 的电压陡降及恢复方案')
        plt.xlabel('数据点索引')
        plt.ylabel('电压 (V)')
        plt.legend()
        plt.grid(True)
        # 支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.savefig(plot_filepath)
        print(f"已生成恢复方案对比图: {plot_filepath}")


# --- 如何运行 ---
if __name__ == '__main__':
    # 1. 将您的CSV文件名填入下方
    input_file = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval\relaxation_battery8_fixed_simple.csv'

    # 2. 运行此脚本
    correct_steep_drops_cumulatively(input_file)