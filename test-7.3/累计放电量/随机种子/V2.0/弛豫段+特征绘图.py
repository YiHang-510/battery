import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# matplotlib.use('TkAgg')  # 如需弹窗显示请取消注释
print(matplotlib.get_backend())

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

for i in range(20, 21):
    try:
        # --- 文件1: 弛豫段电压数据 ---
        file_name = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\relaxation\\Interval\\relaxation_battery{i}.csv'
        df = pd.read_csv(file_name)

        # --- 文件2: 容量数据 ---
        capacity_file_name = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\statistic\\battery{i}_SOH健康特征提取结果.csv'
        capacity_df = pd.read_csv(capacity_file_name)

        # --- ⭐ 步骤 0: 预处理并计算累计放电容量 ---
        capacity_df = capacity_df.sort_values('循环号')
        if '累计放电容量(Ah)' not in capacity_df.columns:
            print("正在计算累计放电容量...")
            capacity_df['累计放电容量(Ah)'] = capacity_df['最大容量(Ah)'].cumsum()

        # --- 合并DataFrame ---
        df = pd.merge(df, capacity_df, on='循环号', how='left')

        # 验证必要的列
        required_columns = ['循环号', '弛豫段电压', '累计放电容量(Ah)']
        if not all(col in df.columns for col in required_columns):
            print(f"错误：合并后的数据中必须包含 {required_columns} 列。")
            exit()

    except Exception as e:
        print(f"发生错误: {e}")
        exit()

    # --- ⭐ 修改 1: 颜色映射值改为 '累计放电容量(Ah)' ---
    # 之前是 df['最大容量(Ah)']
    df['color_value'] = df['累计放电容量(Ah)']

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

    cycles = df['循环号'].unique()
    min_val = df['color_value'].min()
    max_val = df['color_value'].max()

    # --- 颜色反转 (使用 _r) ---
    cmap = plt.get_cmap('summer_r') 

    # 绘制背景曲线
    for cycle_num in cycles:
        cycle_data = df[df['循环号'] == cycle_num]
        x_axis_values = np.arange(len(cycle_data)) * 1 
        y_axis_values = cycle_data['弛豫段电压']

        current_color_value = cycle_data['color_value'].iloc[0]
        norm_val = (current_color_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        color = cmap(norm_val)

        ax.plot(x_axis_values, y_axis_values, color=color, lw=1.5)

    # --- 在最后一个循环上标记7个点 (绿色) ---
    last_cycle = cycles[-1]
    last_cycle_data = df[df['循环号'] == last_cycle]
    
    x_last = np.arange(len(last_cycle_data)) * 1
    y_last = last_cycle_data['弛豫段电压'].values
    
    num_points = 7
    sampled_indices = np.linspace(0, len(last_cycle_data) - 1, num_points, dtype=int)
    
    ax.scatter(x_last[sampled_indices], y_last[sampled_indices], 
               color='#329966',          
               s=120,                    
               edgecolors='black',       
               linewidth=0.8,
               zorder=10,                
               label='7 Sampled Points') 

    # --- 添加颜色条 ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)

    # --- ⭐ 修改 2: 反转颜色条坐标轴 & 更新标签 ---
    cbar.ax.invert_yaxis()
    cbar.set_label('Cumulative Discharge Capacity (Ah)', fontsize=14, rotation=270, labelpad=20)

    # 设置主图坐标轴
    ax.set_xlabel('Relaxation time (s)', fontsize=14)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/10:g}'))
    
    ax.set_ylabel('Relaxation voltage (V)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 图例位置: 左下角
    ax.legend(loc='lower left', fontsize=12, frameon=True) 
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\relaxation_feature.pdf", dpi=600, bbox_inches="tight")
    plt.close()