import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
print(matplotlib.get_backend())  #查看后端
plt.rcParams['font.sans-serif'] = ['SimHei']
# --- 步骤 1: 读取您的两列CSV文件 ---
# 重要：请将下面的 'your_data_file.csv' 替换成您自己的文件名。
# 并确保列名与文件中的一致，如果不一致，请在 header=0, names=['列1', '列2'] 中指定。
for i in range(1,25):
    try:
        file_name = f'D:\任务归档\电池\研究\data\弛豫电压\\relaxation_battery{i}.csv'  # <--- 在这里修改您的文件名

        # 假设您的列名是 'cycle' 和 'voltage'
        # 如果不是，可以这样强制指定：
        # df = pd.read_csv(file_name, header=0, names=['循环号', '弛豫段电压'])
        df = pd.read_csv(file_name, encoding='utf-8-sig')

        # 为了方便处理，我们将列名统一为 'cycle' 和 'voltage'
        # 如果您的列名是中文，请取消下面这行的注释并修改
        # df.columns = ['cycle', 'voltage']

        # 验证必要的列是否存在
        required_columns = ['循环号', '弛豫段电压']
        if not all(col in df.columns for col in required_columns):
            print(f"错误：CSV文件 '{file_name}' 中必须包含表示循环和电压的列。")
            print(f"当前列名: {df.columns.tolist()}")
            exit()

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_name}'。请确保文件与脚本在同一目录下，或者提供了正确的文件路径。")
        exit()

    # --- 步骤 2: 绘制图形 ---

    fig, ax = plt.subplots(figsize=(8, 6))

    # 获取所有唯一的循环号
    cycles = df['循环号'].unique()
    min_cycle = cycles.min()
    max_cycle = cycles.max()

    # 选择颜色映射
    cmap = plt.get_cmap('summer_r')

    # 遍历每一个循环号
    for cycle_num in cycles:
        # 筛选出当前循环的所有电压数据
        cycle_data = df[df['循环号'] == cycle_num]

        # --- 关键步骤: 生成X轴 ---
        # 因为没有时间列，我们用数据点的序号(0, 1, 2, ...)作为X轴
        # 这假设每个循环的数据点数量和时间间隔是相同的
        x_axis_values = np.arange(len(cycle_data))

        # 获取电压值
        y_axis_values = cycle_data['弛豫段电压']

        # ----------------------------------------------------
        # 可选：如果您知道时间间隔，可以生成真实的时间轴
        # 例如，如果每个数据点之间相隔0.5秒，请取消下面两行的注释
        time_step = 0.1  # 秒
        x_axis_values = x_axis_values * time_step
        # ----------------------------------------------------

        # 归一化循环号以获取颜色
        norm_cycle = (cycle_num - min_cycle) / (max_cycle - min_cycle) if max_cycle > min_cycle else 0.5
        color = cmap(norm_cycle)

        # 绘制曲线: X轴是序号, Y轴是电压
        ax.plot(x_axis_values, y_axis_values, color=color, lw=1.5)

    # --- 步骤 3: 添加颜色条和标签 ---

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_cycle, vmax=max_cycle))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('循环号 (Cycle Number)', fontsize=14, rotation=270, labelpad=20)

    # 设置坐标轴标签和标题
    # 注意X轴标签已更改
    ax.set_xlabel('数据点序号 (Data Point Index)', fontsize=14)
    ax.set_ylabel('弛豫段电压 (V)', fontsize=14)
    ax.set_title(f'不同循环下的电压弛豫曲线-电池{i}', fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.show()


