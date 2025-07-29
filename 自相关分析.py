import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf  # 导入计算acf数值的函数

# --- 1. 设置参数 ---
# 请将此路径替换为您存放电池文件的实际文件夹路径
DATA_FOLDER_PATH = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval-singleraw-100x'

# 要分析的电压列名
VOLTAGE_COLUMNS = ['弛豫段电压1', '弛豫段电压2', '弛豫段电压3', '弛豫段电压4', '弛豫段电压5', '弛豫段电压6', '弛豫段电压7', '弛豫段电压8', '弛豫段电压9', '弛豫段电压10', '弛豫段电压11', '弛豫段电压12']

# --- 2. 准备数据 ---
# (这部分与之前的代码相同，用于加载和准备数据)
try:
    all_files = [f for f in os.listdir(DATA_FOLDER_PATH) if f.endswith(('.csv', '.xlsx'))]
    if not all_files:
        print(f"错误：在文件夹 '{DATA_FOLDER_PATH}' 中没有找到任何CSV或Excel文件。")
        print("将创建一个虚拟的DataFrame进行演示。")
        data = {
            "弛豫段电压1": np.sin(np.linspace(0, 80, 200) * (2 * np.pi / 3)),
            "弛豫段电压2": np.sin(np.linspace(0, 80, 200) * (2 * np.pi / 3) + 0.2),
            "弛豫段电压3": np.sin(np.linspace(0, 80, 200) * (2 * np.pi / 3) + 0.4)
        }
        df = pd.DataFrame(data)
        file_to_process = "virtual_data.csv"
    else:
        file_to_process = all_files[0]
        full_path = os.path.join(DATA_FOLDER_PATH, file_to_process)
        print(f"正在处理文件: {file_to_process}")
        if file_to_process.endswith('.csv'):
            df = pd.read_csv(full_path, encoding='gbk')
        else:
            df = pd.read_excel(full_path)

except FileNotFoundError:
    print(f"错误：找不到文件夹 '{DATA_FOLDER_PATH}'。请检查路径是否正确。")
    print("将创建一个虚拟的DataFrame进行演示。")
    data = {
        "弛豫段电压1": np.sin(np.linspace(0, 80, 200) * (2 * np.pi / 3)),
        "弛豫段电压2": np.sin(np.linspace(0, 80, 200) * (2 * np.pi / 3) + 0.2),
        "弛豫段电压3": np.sin(np.linspace(0, 80, 200) * (2 * np.pi / 3) + 0.4)
    }
    df = pd.DataFrame(data)
    file_to_process = "virtual_data.csv"

if not all(col in df.columns for col in VOLTAGE_COLUMNS):
    print(f"错误：文件 '{file_to_process}' 中缺少必要的列。需要以下列: {VOLTAGE_COLUMNS}")
else:
    voltage_series = df[VOLTAGE_COLUMNS].values.flatten()
    print(f"已将 {len(df)} 个循环转换为长度为 {len(voltage_series)} 的时间序列。")

    # --- 3. 计算ACF值并绘制论文风格的图 ---

    # 设置matplotlib以匹配论文风格
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用一个干净的网格背景

    # 手动计算ACF值，nlags是你希望看到的最大延迟
    max_lags = 1000  # 可以根据需要调整
    acf_values = acf(voltage_series, nlags=max_lags)

    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 5))

    lags_range = range(len(acf_values))

    # 使用fill_between来创建填充面积图
    ax.fill_between(lags_range, acf_values, color='tab:blue')

    # 设置标题和坐标轴标签
    ax.set_title(f'(x) {os.path.splitext(file_to_process)[0]}, W = ?', fontsize=14)
    ax.set_xlabel('Lags', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)

    # 设置坐标轴范围，使其与论文风格更接近
    ax.set_xlim(0, max_lags)
    ax.set_ylim(min(acf_values.min() - 0.1, 0), 1.05)  # 动态调整Y轴下限

    plt.tight_layout()

    # 保存图表
    output_filename_paper_style = f'acf_plot_paper_style_{os.path.splitext(file_to_process)[0]}2.png'
    plt.savefig(output_filename_paper_style, dpi=1200)
    print(f"论文风格的ACF分析图已保存为: {output_filename_paper_style}")