import pandas as pd
import os
import re

# --- 用户需要修改的设置 ---
# 请将您的24个CSV文件所在的文件夹路径替换到下面的引号中

# folder_path = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\End'  # 使用 "./" 表示当前脚本所在的文件夹
#
# # 合并后输出的文件名
# output_filename = r"D:\任务归档\电池\研究\data\selected_feature\relaxation\End\EndVrlx_data_all_battery.csv"

#
folder_path = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\interval'  # 使用 "./" 表示当前脚本所在的文件夹

# 合并后输出的文件名
output_filename = r"D:\任务归档\电池\研究\data\selected_feature\relaxation\interval\relaxation_interval_all_battery.csv"

# --- 设置结束 ---


def combine_battery_data(path, output_file):
    """
    合并指定文件夹中所有电池数据的CSV文件，并只提取特定列。

    Args:
        path (str): 存放CSV文件的文件夹路径。
        output_file (str): 合并后输出的CSV文件名。
    """
    # 获取文件夹下所有的文件名
    try:
        all_files = os.listdir(path)
    except FileNotFoundError:
        print(f"错误：找不到文件夹 '{path}'。请检查路径是否正确。")
        return

    # 筛选出所有csv文件
    csv_files = [f for f in all_files if f.endswith('.csv') and f != output_file]

    if not csv_files:
        print(f"在文件夹 '{path}' 中没有找到任何CSV文件。")
        return

    print(f"找到了 {len(csv_files)} 个CSV文件，将进行合并...")

    # 创建一个空列表，用于存放每个文件读取后的DataFrame
    all_dataframes = []

    # 定义需要从每个文件中提取的列
    columns_to_extract = ['循环号', '弛豫段电压']

    # 遍历所有找到的CSV文件
    for file in csv_files:
        # 构建完整的文件路径
        file_path = os.path.join(path, file)

        try:
            # 读取CSV文件到DataFrame
            df = pd.read_csv(file_path, encoding='gbk')

            # 检查所需的列是否存在
            if not all(col in df.columns for col in columns_to_extract):
                print(f"警告: 文件 {file} 缺少必要的列({', '.join(columns_to_extract)})，将跳过此文件。")
                continue

            # 只保留指定的列
            df_filtered = df[columns_to_extract].copy()

            # --- 提取电池编号 ---
            # 使用正则表达式从文件名中提取 "batteryX" 部分
            # 例如，从 "EndVrlx_battery1.csv" 提取 "battery1"
            match = re.search(r'(battery\d+)', file, re.IGNORECASE)
            if match:
                battery_id = match.group(1)
            else:
                # 如果正则不匹配，就使用文件名（不含扩展名）作为ID
                battery_id = os.path.splitext(file)[0]

            # 在过滤后的DataFrame上新建一列 "电池编号"，并赋值
            df_filtered['电池编号'] = battery_id

            # 将处理好的DataFrame添加到列表中
            all_dataframes.append(df_filtered)
            print(f"已处理文件: {file}, 添加编号: {battery_id}, 提取了列: {', '.join(columns_to_extract)}")

        except Exception as e:
            print(f"处理文件 {file} 时发生错误: {e}")

    # 检查列表是否为空
    if not all_dataframes:
        print("没有成功处理任何文件，无法进行合并。")
        return

    # 使用pd.concat将列表中的所有DataFrame合并成一个
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # --- 保存结果 ---
    try:
        # 将合并后的DataFrame保存为新的CSV文件
        combined_df.to_csv(os.path.join(path, output_file), index=False, encoding='gbk')
        print("\n==========================================")
        print(f"🎉 合并完成！数据已保存到文件: {os.path.join(path, output_file)}")
        print(f"总共合并了 {len(combined_df)} 行数据。")
        print("==========================================")

        # 显示合并后数据的前5行和最后5行，以及基本信息
        print("\n合并后数据预览 (前5行):")
        print(combined_df.head())
        print("\n合并后数据预览 (后5行):")
        print(combined_df.tail())
        print("\n电池编号统计:")
        print(combined_df['电池编号'].value_counts())

    except Exception as e:
        print(f"保存文件时发生错误: {e}")


# --- 运行主函数 ---
if __name__ == "__main__":
    combine_battery_data(folder_path, output_filename)
