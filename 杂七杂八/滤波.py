import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


def smooth_voltage_curves(input_filepath, output_filepath=None):
    """
    使用 Savitzky-Golay 滤波器平滑电压曲线。

    Args:
        input_filepath (str): 原始数据CSV文件路径。
        output_filepath (str, optional): 保存平滑后数据的CSV文件路径。
    """
    # --- 1. 读取数据 ---
    if not os.path.exists(input_filepath):
        print(f"错误：文件 '{input_filepath}' 未找到。请确保它和脚本在同一个文件夹下。")
        return None, None

    try:
        df = pd.read_csv(input_filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_filepath, encoding='gbk')

    print(f"--- 正在处理文件: {input_filepath} ---")

    # 检查必要的列是否存在
    required_columns = ['循环号', '弛豫段电压']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：CSV文件中缺少必要的列。请确保文件包含 {required_columns}。")
        return None, None

    # 假设数据点序号是按顺序排列的，如果不存在则创建
    if '数据点序号' not in df.columns:
        df['数据点序号'] = df.groupby('循环号').cumcount()

    # --- 2. 应用S-G滤波器 ---
    corrected_groups = []
    for cycle_num, group in df.groupby('循环号'):
        group_copy = group.copy()

        original_voltage = group_copy['弛豫段电压']

        # *** 核心步骤: 应用 Savitzky-Golay 滤波器 ***
        window_length = 11
        polyorder = 3

        # 确保窗口长度小于数据点数量
        if len(original_voltage) > window_length:
            smoothed_voltage = savgol_filter(original_voltage, window_length, polyorder)
            group_copy['弛豫段电压'] = smoothed_voltage
        else:
            # 如果数据点太少，无法滤波，则保留原始数据
            print(f"警告：循环 {cycle_num} 的数据点数量 ({len(original_voltage)}) 过少，无法应用滤波，将保留原始数据。")

        corrected_groups.append(group_copy)

    df_corrected = pd.concat(corrected_groups)

    # --- 3. 保存校正后的数据 ---
    if output_filepath:
        df_corrected.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"--- 数据平滑完成。已保存至: {output_filepath} ---")

    return df, df_corrected


def plot_comparison(df_original, df_corrected, cycle_to_plot):
    """
    可视化原始数据与平滑后数据的对比。
    """
    plt.figure(figsize=(12, 7))

    original_cycle = df_original[df_original['循环号'] == cycle_to_plot]
    corrected_cycle = df_corrected[df_corrected['循环号'] == cycle_to_plot]

    plt.plot(original_cycle['数据点序号'], original_cycle['弛豫段电压'], label=f'原始数据 (循环 {cycle_to_plot})',
             marker='.', linestyle='-', alpha=0.7)
    plt.plot(corrected_cycle['数据点序号'], corrected_cycle['弛豫段电压'],
             label=f'S-G平滑后数据 (循环 {cycle_to_plot})', color='red', linewidth=2)

    plt.title(f'循环 {cycle_to_plot} 的电压曲线平滑校正对比')
    plt.xlabel('数据点序号 (Data Point Index)')
    plt.ylabel('弛豫段电压 (V)')
    plt.legend()
    plt.grid(True)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.savefig('comparison_plot.png')  # 保存图片而不是直接显示
    print("已生成对比图：comparison_plot.png")


# --- 如何运行 ---
if __name__ == '__main__':
    # 1. 将您的CSV文件名填入下方
    # 假设您的数据文件名为 'battery_data.csv'
    input_file = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval\relaxation_battery1_fixed_simple_restored.csv'
    output_file = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval\relaxation_battery1_fixed_simple_corrected.csv'

    # 2. 运行平滑函数
    df_orig, df_corr = smooth_voltage_curves(input_file, output_file)

    # 3. 选择一个有抖动的循环号（例如第200个循环）进行可视化对比
    # 从您的图中看，可以选择一个大于150的循环号来观察效果
    # 4. 选择一个有抖动的循环号进行可视化对比
    if df_orig is not None:
        cycle_for_visualization = 200  # 您可以修改这个循环号来查看不同周期的效果
        if cycle_for_visualization in df_orig['循环号'].values:
            plot_comparison(df_orig, df_corr, cycle_for_visualization)
        else:
            # 如果指定的循环号不存在，就选择最后一个循环号
            last_cycle = df_orig['循环号'].max()
            print(f"警告：选择的循环号 {cycle_for_visualization} 不在数据中，将使用最后一个循环 {last_cycle} 生成对比图。")
            plot_comparison(df_orig, df_corr, last_cycle)