import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.ticker as ticker  # <--- 新增这行
# --- 基本绘图设置 (与您原来的一致) ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# --- 数据读取和合并部分 (与您原来的一致) ---
try:
    i = 20  # 假设我们处理 battery 20
    # 文件1: 弛豫段电压数据
    file_name = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\relaxation\\Interval\\relaxation_battery{i}.csv'
    df = pd.read_csv(file_name)

    # 文件2: 容量/SOH数据
    capacity_file_name = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\statistic\\battery{i}_SOH健康特征提取结果.csv'
    capacity_df = pd.read_csv(capacity_file_name)

    # 合并两个DataFrame
    df = pd.merge(df, capacity_df, on='循环号', how='left')

    # 验证列是否存在
    required_columns = ['循环号', '弛豫段电压', '最大容量(Ah)']
    if not all(col in df.columns for col in required_columns):
        print(f"错误：合并后的数据中缺少必需的列。")
        exit()

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"发生错误: {e}")
    exit()


# --- ⭐ 步骤 1: 指定要绘制的循环并筛选数据 ---
cycle_to_plot = 100  # <--- 在这里修改为您想绘制的特定循环号

# 筛选出该循环的所有数据，并重置索引以便于后续操作
cycle_data = df[df['循环号'] == cycle_to_plot].reset_index(drop=True)

# 检查该循环是否存在数据
if cycle_data.empty:
    print(f"错误：在数据中找不到循环号 {cycle_to_plot}。请检查您的数据或修改 'cycle_to_plot' 的值。")
    exit()

# # --- ⭐ 新增步骤: 创建 t<0 的扩充数据 ---
# # 获取第一个点的电压值
# first_voltage = y_original.iloc[0]
# # 定义扩充数据的X轴范围，例如从-20到0
# prepend_x_range = -20
# # 创建用于绘制水平线的数据点 (只需起点和终点)
# x_prepend = np.array([prepend_x_range, 0])
# y_prepend = np.array([first_voltage, first_voltage])

# --- ⭐ 步骤 2: 绘制单条曲线 ---
fig, ax = plt.subplots(figsize=(4, 3.5), dpi=600)

# 定义X轴和Y轴
time_step = 1  # 您可以根据实际情况调整时间步长，这里设为1代表数据点索引
x_axis_values = cycle_data.index * time_step
y_axis_values = cycle_data['弛豫段电压']

# 去除顶部和右侧的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 绘制弛豫电压曲线，颜色参考了您提供的图片
ax.plot(x_axis_values, y_axis_values, color='#3875A4', lw=6)


# --- ⭐ 步骤 3: 均匀选取7个点并绘制 ---
num_points_to_sample = 7
total_points = len(cycle_data)

# 使用 np.linspace 生成7个在 [0, total_points-1] 区间内均匀分布的整数索引
sampled_indices = np.linspace(0, total_points - 1, num_points_to_sample, dtype=int)

# 获取这些索引对应的X和Y坐标
x_sampled = x_axis_values[sampled_indices]
y_sampled = y_axis_values[sampled_indices]

# 使用 scatter 绘制这7个点，zorder=5 确保点在曲线之上
ax.scatter(x_sampled, y_sampled,
           marker='o',               # 标记样式为圆圈
           s=200,                    # 点的大小
           facecolors='#FDE0C0',        # 填充颜色为青色
           edgecolors='#DD8E4A', # 边缘颜色
           linewidth=1.5,
        #    label=f'{num_points_to_sample} Sampled Points',
           zorder=5)


# --- ⭐ 步骤 4: 调整并美化图形 ---
# (移除了与颜色条相关的代码)
# ax.set_xlabel('Relaxation time (s)', fontsize=14) # 建议X轴标签改为时间
    # --- ⭐ 修改: 仅改变X轴标签显示的数值 (数值除以10) ---
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10:g}'))
# ax.set_ylabel('Relaxation Voltage (V)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend(fontsize=12)


# --- 保存并显示图形 ---
plt.tight_layout()
fig.savefig(r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\relaxation_voltage_feature_single.png", dpi=600, bbox_inches="tight", transparent=True)
fig.savefig(r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\relaxation_voltage_feature_single.pdf", bbox_inches="tight", transparent=True)
# 如果需要，可以取消下面的注释来保存图片
# fig.savefig(r"D:\任务归档\电池\研究\小论文1号\DOCUMENT\fig\single_cycle_relaxation.png", dpi=600, bbox_inches="tight")
plt.close()