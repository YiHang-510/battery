import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 加载和准备数据 ---

# !!! 重要: 请将 'your_file.csv' 替换为您的 CSV 文件名
csv_filename = r'D:\任务归档\电池\研究\data\selected_data\battery1\battery1_record.csv'

try:
    # 加载数据
    df = pd.read_csv(csv_filename, encoding='gbk')

    # 筛选“循环号”为 100 的数据
    # .copy() 是为了避免后续操作出现 SettingWithCopyWarning
    df_cycle_100 = df[df['循环号'] == 100].copy()

    if df_cycle_100.empty:
        print(f"错误: 在文件 {csv_filename} 中没有找到 '循环号' 为 100 的数据。")
    else:
        print(f"成功加载并筛选了 '循环号' 100 的数据，共 {len(df_cycle_100)} 个数据点。")

        # 将用作 X 轴的索引
        x_data = df_cycle_100.index

        # --- 2. 开始绘图 ---

        # 创建画布和第一个 Y 轴 (ax1)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 定义颜色和线型 (参考您给定的图像)
        color_voltage = 'solid'  # 电压: 实线
        color_current = 'dotted'  # 电流: 点线
        color_charging = 'dashed'  # 充电容量: 虚线
        color_discharging = 'solid'  # 放电容量: 实线

        # --- 3. 绘制左 Y 轴 (电流和电压) ---
        # 绘制 电压(V)
        ax1.plot(x_data, df_cycle_100['电压(V)'],
                 color='tab:red', linestyle=color_voltage, label='电压(V)')

        # 绘制 电流(A)
        ax1.plot(x_data, df_cycle_100['电流(A)'],
                 color='tab:blue', linestyle=color_current, label='电流(A)')

        # 设置 X 轴和左 Y 轴的标签
        ax1.set_xlabel('Sampling Steps (Index)')  # X 轴标签
        ax1.set_ylabel('Y1: 电流(A) / Y2: 电压(V)')  # 左 Y 轴标签

        # 设置左 Y 轴的范围 (根据您的图像示例)
        ax1.set_ylim(-5, 5)

        # --- 4. 创建并绘制右 Y 轴 (充电和放电容量) ---
        ax2 = ax1.twinx()  # 关键步骤: 创建共享 X 轴的第二个 Y 轴

        # 绘制 充电容量(Ah)
        ax2.plot(x_data, df_cycle_100['充电容量(Ah)'],
                 color='tab:blue', linestyle=color_charging, label='充电容量(Ah)')

        # 绘制 放电容量(Ah)
        ax2.plot(x_data, df_cycle_100['放电容量(Ah)'],
                 color='tab:green', linestyle=color_discharging, label='放电容量(Ah)')

        # 设置右 Y 轴的标签
        ax2.set_ylabel('Y3: 充电容量(Ah) / Y4: 放电容量(Ah)')

        # 设置右 Y 轴的范围 (根据您的图像示例)
        ax2.set_ylim(0, 4)

        # --- 5. 创建统一的图例 ---
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax2.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=4, frameon=False)

        # --- 6. 调整布局并保存/显示 ---
        fig.tight_layout()

        plt.savefig('cycle_100_plot.png', dpi=300)
        print("图像已保存为 'cycle_100_plot.png'")

        # plt.show()

except FileNotFoundError:
    print(f"错误: 文件 {csv_filename} 未找到。请确保文件在正确的路径下。")
except KeyError as e:
    print(f"错误: 数据中缺少必要的列: {e}。")
    print(f"请检查您的 CSV 文件是否包含 '循环号', '电流(A)', '电压(V)', '充电容量(Ah)', '放电容量(Ah)' 这些列。")