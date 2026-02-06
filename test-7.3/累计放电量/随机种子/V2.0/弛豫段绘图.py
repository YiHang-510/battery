import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

print(matplotlib.get_backend())  # 查看后端
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

        # --- ⭐ 新增步骤 1: 读取容量数据文件 ---
        # --- 请将下面的路径替换成您的容量数据文件名 ---
        capacity_file_name = f'D:\任务归档\电池\研究\data\selected_feature\statistic\\battery{i}_SOH健康特征提取结果.csv'  # <--- 在这里修改您的容量文件名
        capacity_df = pd.read_csv(capacity_file_name)

        # --- ⭐ 新增步骤 2: 合并两个DataFrame ---
        # 假设两个文件都有一个名为 '循环号' 的列用于关联
        # 如果列名不同, 可以这样修改: pd.merge(df, capacity_df, left_on='df的循环列名', right_on='capacity_df的循环列名')
        df = pd.merge(df, capacity_df, on='循环号', how='left')

        # 验证必要的列是否存在 (增加了'最大容量(Ah)')
        required_columns = ['循环号', '弛豫段电压', '最大容量(Ah)', '累计放电容量(Ah)']
        if not all(col in df.columns for col in required_columns):
            print(f"错误：合并后的数据中必须包含 '循环号', '弛豫段电压', '最大容量(Ah)', '累计放电容量(Ah)' 列。")
            print(f"当前列名: {df.columns.tolist()}")
            exit()

    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。请确保文件路径正确。")
        exit()
    except Exception as e:
        print(f"发生错误: {e}")
        exit()

    # --- ⭐ 新增步骤 3: 计算用于颜色映射的新列 ---
    df['color_value'] = df['最大容量(Ah)'] / 3.5

    # --- 步骤 2: 绘制图形 ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

    # 获取所有唯一的循环号
    cycles = df['循环号'].unique()

    # --- ⭐ 修改: 获取颜色映射的最大最小值 ---
    min_val = df['color_value'].min()
    max_val = df['color_value'].max()

    # my_colors = ['#fc757b','#f97f5f', '#faa26f', '#FDCD94', '#fee199', '#b0d6a9','#65bdba','#3c9bc9']  # 从红 -> 黄 -> 绿 -> 青 -> 蓝
    # my_colors = ['#F8D0B8','#F8F0C8', '#D8F0F8', '#D0E0B8', '#D8E8F0']  # 从红 -> 黄 -> 绿 -> 青 -> 蓝
    my_colors = ['#faa26f', '#FDCD94', '#fee199']  # 从红 -> 黄 -> 绿 -> 青 -> 蓝

    # 2. 使用 ListedColormap 创建 Colormap 对象
    # my_cmap = ListedColormap(my_colors)
    my_cmap = LinearSegmentedColormap.from_list("my_cmap", my_colors)
    # cmap = my_cmap

    cmap = plt.get_cmap('summer')
    # 遍历每一个循环号
    for cycle_num in cycles:
        cycle_data = df[df['循环号'] == cycle_num]
        # 同一循环号下的横坐标改为对应的“累计放电容量(Ah)”，每个点沿该容量绘制
        capacity_x = cycle_data['累计放电容量(Ah)'].iloc[0]
        x_axis_values = np.full(len(cycle_data), capacity_x)
        y_axis_values = cycle_data['弛豫段电压']

        # --- ⭐ 修改: 使用新的 'color_value' 进行归一化以确定颜色 ---
        # (因为一个循环里所有行的这个值都一样,取第一个即可)
        current_color_value = cycle_data['color_value'].iloc[0]
        norm_val = (current_color_value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        color = cmap(norm_val)

        ax.plot(x_axis_values, y_axis_values, color=color, lw=1.5)

    # --- 步骤 3: 添加颜色条和标签 ---

    # --- ⭐ 修改: 使用新的min_val和max_val创建颜色条 ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)

    # --- ⭐ 修改: 更新颜色条的标签 ---
    cbar.set_label('SOH', fontsize=14, rotation=270, labelpad=20) # 您可以自定义标签名称

    # 设置坐标轴标签和标题
    ax.set_xlabel('累计放电容量 (Ah)', fontsize=14)
    ax.set_ylabel('Relaxation voltage (V)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.grid(True, linestyle='--', alpha=0.6)

    # 自动调整布局，防止标签显示不全
    plt.tight_layout()
    fig.savefig(r"D:\任务归档\电池\研究\小论文1号\DOCUMENT\fig\relaxation_voltage.png", dpi=600, bbox_inches="tight")
    fig.savefig(r"D:\任务归档\电池\研究\小论文1号\DOCUMENT\fig\relaxation_voltage.pdf", bbox_inches="tight")
    plt.show()
