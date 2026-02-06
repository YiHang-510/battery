import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl

# --- 基本绘图设置 ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# --- 数据读取和合并部分 ---
try:
    i = 20  # 电池编号
    # 文件路径
    file_name = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\relaxation\\Interval\\relaxation_battery{i}.csv'
    df = pd.read_csv(file_name)

    capacity_file_name = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\statistic\\battery{i}_SOH健康特征提取结果.csv'
    capacity_df = pd.read_csv(capacity_file_name)

    # --- ⭐ 新增步骤 0: 计算累计放电容量 ---
    # 确保按循环号排序，防止累加顺序错误
    capacity_df = capacity_df.sort_values('循环号')
    
    # 检查列名是否存在，如果不存在则手动计算
    # 假设 '最大容量(Ah)' 是单次循环的放电容量
    if '累计放电容量(Ah)' not in capacity_df.columns:
        print("正在计算累计放电容量...")
        capacity_df['累计放电容量(Ah)'] = capacity_df['最大容量(Ah)'].cumsum()
    
    # 合并
    df = pd.merge(df, capacity_df, on='循环号', how='left')

    required_columns = ['循环号', '弛豫段电压', '累计放电容量(Ah)']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：合并后缺失必要列。请检查数据。")
        exit()

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}")
    exit()
except Exception as e:
    print(f"发生错误: {e}")
    exit()


# --- ⭐ 步骤 1: 遍历所有循环并提取特征 ---

all_cycles = sorted(df['循环号'].unique())
extracted_features = []
num_points_to_sample = 7

print("正在提取特征点...")

for cycle in all_cycles:
    cycle_data = df[df['循环号'] == cycle].reset_index(drop=True)
    
    if cycle_data.empty:
        continue

    # 如果电压数据点太少，跳过
    voltages = cycle_data['弛豫段电压'].values
    if len(voltages) < num_points_to_sample:
        continue

    # --- 获取该循环的累计容量 (取第一个值即可，因为同一循环内该值相同) ---
    current_cum_capacity = cycle_data['累计放电容量(Ah)'].iloc[0]

    # --- 均匀选取索引 ---
    sampled_indices = np.linspace(0, len(voltages) - 1, num_points_to_sample, dtype=int)
    sampled_voltages = voltages[sampled_indices]
    
    # --- 存储数据: [循环号, 累计容量, v1, v2, ..., v7] ---
    # 将 numpy array 转为 list 并拼接
    record = [cycle, current_cum_capacity] + sampled_voltages.tolist()
    extracted_features.append(record)

# 转换为 DataFrame
# 列名: Cycle, Cumulative_Capacity, Point_1, Point_2, ...
columns = ['Cycle', 'Cumulative_Capacity'] + [f'Point_{k+1}' for k in range(num_points_to_sample)]
feature_df = pd.DataFrame(extracted_features, columns=columns)


# --- ⭐ 步骤 2: 绘制轨迹图 (横坐标改为累计容量) ---

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

# 使用 plt.get_cmap 修复之前的报错
colors = plt.get_cmap('viridis', num_points_to_sample)

for k in range(num_points_to_sample):
    col_name = f'Point_{k+1}'
    
    # --- ⭐ 修改: X轴使用 'Cumulative_Capacity' ---
    ax.plot(feature_df['Cumulative_Capacity'], feature_df[col_name], 
            lw=1.5, 
            color=colors(k), 
            label=f'Sample Point {k+1}')

# --- ⭐ 步骤 3: 美化图形 ---

# 修改X轴标签
ax.set_xlabel('Cumulative Discharge Capacity (Ah)', fontsize=18)
ax.set_ylabel('Relaxation Voltage (V)', fontsize=18)
# ax.set_title(f'Trajectory of {num_points_to_sample} Sampled Points vs Cumulative Capacity', fontsize=14)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

# 图例位置
ax.legend(loc='lower left', fontsize=16, frameon=True)

plt.tight_layout()

save_path_png = r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\sampled_points_trajectory_capacity.png"
fig.savefig(save_path_png, dpi=600, bbox_inches="tight")

print(f"绘图完成！已保存至: {save_path_png}")
plt.close()