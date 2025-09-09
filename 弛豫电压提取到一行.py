import pandas as pd
import os

# 获取当前文件夹路径
folder_path = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval-Downsampling_200x'
out_path = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval-singleraw-200x'
os.makedirs(out_path, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否是CSV文件，并且不是已经处理过的文件
    if filename.endswith('.csv') and not filename.startswith('processed_'):
        file_path = os.path.join(folder_path, filename)

        print(f"正在处理文件: {filename} ...")

        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 确保文件包含预期的列
            if '循环号' not in df.columns or '弛豫段电压' not in df.columns:
                print(f"  -> 文件 {filename} 缺少 '循环号' 或 '弛豫段电压' 列，已跳过。")
                continue

            # 为每个循环号内的“弛豫段电压”进行编号 (1, 2, 3)
            df['voltage_num'] = df.groupby('循环号').cumcount() + 1

            # 使用pivot_table将数据从长格式转换为宽格式
            df_pivoted = df.pivot_table(index='循环号', columns='voltage_num', values='弛豫段电压').reset_index()

            # 重命名列
            df_pivoted.columns = ['循环号', '弛豫段电压1', '弛豫段电压2', '弛豫段电压3', '弛豫段电压4', '弛豫段电压5', '弛豫段电压6']
#,'弛豫段电压7','弛豫段电压8','弛豫段电压9','弛豫段电压10','弛豫段电压11','弛豫段电压12'
            # 创建新的文件名并保存
            output_filename = f"{filename}"
            output_path = os.path.join(out_path, output_filename)
            df_pivoted.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"  -> 文件处理完成，已保存为: {output_filename}")

        except Exception as e:
            print(f"  -> 处理文件 {filename} 时发生错误: {e}")

print("\n所有文件处理完毕！")