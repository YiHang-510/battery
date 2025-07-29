import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from matplotlib.font_manager import FontManager


# --- 设置中文字体 (如果标题或标签仍需中文) ---
# 这部分代码保持不变，以防万一图表的其他部分仍需中文
def set_chinese_font():
    try:
        fm = FontManager()
        font_names = ['SimHei', 'Heiti TC', 'Microsoft YaHei', 'PingFang SC']
        for font in fm.ttflist:
            if font.name in font_names:
                plt.rcParams['font.sans-serif'] = [font.name]
                print(f"✅ 成功找到并设置中文字体: {font.name}")
                return
        print("⚠️ 警告: 未在系统中找到推荐的中文字体。")
    except Exception as e:
        print(f"❌ 设置中文字体时出错: {e}")


set_chinese_font()
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------

def analyze_and_plot_acf(dataframe, column_name, max_lags=100):
    """
    对指定的列进行ACF分析并绘制图表。
    """
    print(f"--- Analyzing column: {column_name} ---")

    if column_name not in dataframe.columns:
        print(f"Error: Column '{column_name}' not found in the file.")
        return

    time_series = dataframe[column_name].dropna()

    if len(time_series) < max_lags:
        max_lags = len(time_series) - 1

    plt.style.use('seaborn-v0_8-whitegrid')
    acf_values = acf(time_series, nlags=max_lags, fft=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    lags_range = range(len(acf_values))

    ax.fill_between(lags_range, acf_values, color='tab:blue')

    ax.set_title(f'ACF for {column_name}', fontsize=14)
    ax.set_xlabel('Lag (Number of Cycles)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_xlim(0, max_lags)
    ax.set_ylim(min(acf_values.min() - 0.1, 0), 1.05)

    plt.tight_layout()

    safe_col_name = "".join([c for c in column_name if c.isalnum() or c in ['_', '-']]).rstrip()
    output_filename = f'acf_plot_{safe_col_name}.png'
    plt.savefig(output_filename)
    print(f"Chart saved as: {output_filename}\n")

# --- 1. 设置参数 ---
# 请将此路径替换为您的新文件所在的文件夹路径
DATA_FOLDER_PATH = r'D:\任务归档\电池\研究\data\selected_feature\statistic'
# 假设这是您的新文件名，请根据实际情况修改
NEW_FILE_NAME = 'battery1_SOH健康特征提取结果.csv'

full_file_path = os.path.join(DATA_FOLDER_PATH, NEW_FILE_NAME)
try:
    if NEW_FILE_NAME.endswith('.csv'):
        main_df = pd.read_csv(full_file_path, encoding='gbk')
    else:
        main_df = pd.read_excel(full_file_path)

    # --- 新增步骤：重命名列 ---
    rename_dict = {
        "恒压充电时间(s)": "Constant_Voltage_Charging_Time_s",
        "3.3~3.6V充电时间(s)": "Charge_Time_3.3_to_3.6V_s"
    }
    main_df = main_df.rename(columns=rename_dict)
    print("列已重命名为英文:")
    print(main_df.head(2)) # 打印前两行以检查新列名
    # --------------------------

    # --- 更新要分析的列列表为英文名 ---
    COLUMNS_TO_ANALYZE = ["Constant_Voltage_Charging_Time_s", "Charge_Time_3.3_to_3.6V_s"]

    # --- 3. 循环分析每一列 ---
    for column in COLUMNS_TO_ANALYZE:
        analyze_and_plot_acf(main_df, column, max_lags=400)

except FileNotFoundError:
    print(f"错误: 找不到文件 '{full_file_path}'。")