import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- 设置全局字体 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def load_battery_data(battery_id=20):
    """ 读取并处理数据 """
    try:
        file_path_voltage = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\relaxation\\Interval\\relaxation_battery{battery_id}.csv'
        file_path_capacity = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\statistic\\battery{battery_id}_SOH健康特征提取结果.csv'
        print(f"正在读取: Battery {battery_id} ...")
        df_voltage = pd.read_csv(file_path_voltage)
        df_capacity = pd.read_csv(file_path_capacity)
        
        # 预处理
        df_capacity = df_capacity.sort_values('循环号')
        if '累计放电容量(Ah)' not in df_capacity.columns:
            df_capacity['累计放电容量(Ah)'] = df_capacity['最大容量(Ah)'].cumsum()
            
        # 合并
        df_merged = pd.merge(df_voltage, df_capacity[['循环号', '累计放电容量(Ah)']], on='循环号', how='left')
        if df_merged['累计放电容量(Ah)'].isnull().any():
            df_merged = df_merged.dropna(subset=['累计放电容量(Ah)'])
            
        return df_merged
    except Exception as e:
        print(f"读取数据发生错误: {e}")
        return None

def plot_3d_final_fixed_axis(df):
    if df is None or df.empty:
        return

    # 1. 准备画布
    fig = plt.figure(figsize=(8, 6), dpi=600) 
    ax = fig.add_subplot(111, projection='3d')

    cycles = df['循环号'].unique()
    
    # 2. 颜色映射
    min_cap = df['累计放电容量(Ah)'].min()
    max_cap = df['累计放电容量(Ah)'].max()
    norm_bg = mcolors.Normalize(vmin=min_cap, vmax=max_cap)
    
    try:
        cmap_bg = matplotlib.colormaps['summer_r']
        cmap_points = matplotlib.colormaps['viridis'] 
    except:
        cmap_bg = plt.get_cmap('summer_r')
        cmap_points = plt.get_cmap('viridis')

    # 3. 抽样绘制背景线
    total_cycles = len(cycles)
    target_lines = 60 
    step = max(1, total_cycles // target_lines)
    cycles_to_plot = list(cycles[::step])
    if cycles[0] not in cycles_to_plot: cycles_to_plot.insert(0, cycles[0])
    if cycles[-1] not in cycles_to_plot: cycles_to_plot.append(cycles[-1])

    num_points = 7
    len_time = len(df[df['循环号']==cycles[0]])
    sample_time_indices = np.linspace(0, len_time - 1, num_points, dtype=int)
    trajectory_data = {k: {'x': [], 'y': [], 'z': []} for k in sample_time_indices}

    print("绘制背景线条...")
    for cycle_num in cycles_to_plot:
        cycle_data = df[df['循环号'] == cycle_num]
        
        # X=Capacity, Y=Time
        time_values = np.arange(len(cycle_data)) 
        voltage_values = cycle_data['弛豫段电压'].values
        current_cap = cycle_data['累计放电容量(Ah)'].iloc[0]
        capacity_values = np.full_like(time_values, current_cap)

        color = cmap_bg(norm_bg(current_cap))
        ax.plot(capacity_values, time_values, voltage_values, 
                color=color, linewidth=1.5, alpha=0.8)

    # 4. 收集轨迹数据
    traj_step = max(1, total_cycles // 100)
    for cycle_num in cycles[::traj_step]:
        cycle_data = df[df['循环号'] == cycle_num]
        current_cap = cycle_data['累计放电容量(Ah)'].iloc[0]
        vals = cycle_data['弛豫段电压'].values
        
        for k in sample_time_indices:
            if k < len(vals):
                trajectory_data[k]['x'].append(current_cap)
                trajectory_data[k]['y'].append(k)
                trajectory_data[k]['z'].append(vals[k])

    # 5. 绘制特征点与轨迹
    print("绘制特征点与轨迹...")
    point_colors = [cmap_points(i / (num_points - 1)) for i in range(num_points)]

    for idx, k in enumerate(sample_time_indices):
        data = trajectory_data[k]
        this_color = point_colors[idx]
        
        ax.plot(data['x'], data['y'], data['z'], 
                color=this_color, linewidth=2.5, alpha=1.0, zorder=100)
        
        ax.scatter([data['x'][0]], [data['y'][0]], [data['z'][0]], 
                   color=this_color, s=80, edgecolors='black', linewidth=1.0, zorder=200)

        ax.scatter([data['x'][-1]], [data['y'][-1]], [data['z'][-1]], 
                   color=this_color, s=80, edgecolors='white', linewidth=1.0, zorder=200)

    # --- 6. 坐标轴与视角设置 (核心修复) ---
    
    # 标签设置 (增加 labelpad 防止被切掉)
    ax.set_xlabel('Cumulative Discharge Capacity (Ah)', fontsize=14, labelpad=4)
    ax.set_ylabel('Relaxation time (s)', fontsize=14, labelpad=8)
    # ⭐ 关键修改：增加 labelpad 让电压标签显示出来
    ax.set_zlabel('Voltage (V)', fontsize=14, labelpad=8) 

    # 刻度设置 (增加 pad 防止刻度数字消失)
    ax.xaxis.set_tick_params(labelsize=13, pad=0)
    ax.yaxis.set_tick_params(labelsize=13, pad=3)
    ax.zaxis.set_tick_params(labelsize=13, pad=3) # 增加 Z 轴刻度距离

    # 【反转 X 轴】 (Capacity: Max -> Min)
    # 之前是 set_xlim(min, max)，现在反过来
    ax.set_xlim(min_cap, max_cap)

    # 【反转 Y 轴】 (Time: Max -> Min)
    ax.invert_yaxis()

    # 【视角调整】
    # 将 azim 从 -30 调整为 -45 或 -60，有助于在 X 轴反转时让 Z 轴视觉上更靠左
    ax.view_init(elev=18, azim=-60)

    # 样式微调
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    z_min, z_max = df['弛豫段电压'].min(), df['弛豫段电压'].max()
    ax.set_zlim(z_min - 0.002, z_max + 0.002)

    # 保存
    plt.tight_layout()
    output_path = r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\sampled_points_final_fixed.png"
    # pad_inches 稍微加大一点，确保 Z 轴标签不被裁剪
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.3)
    print(f"绘图完成，已修复标签显示，保存至: {output_path}")
    plt.close()

if __name__ == '__main__':
    df_real = load_battery_data(battery_id=20)
    if df_real is not None:
        plot_3d_final_fixed_axis(df_real)