import pandas as pd
import os
import sys

# --- 1. 请在这里设置您的文件夹路径 ---
# 文件夹A的路径，包含CSV文件
folder_a_path = r'D:\任务归档\电池\研究\data\selected_feature\statistic'
# 文件夹B的路径，包含名为'battery1', 'battery2', ...的子文件夹
folder_b_path = r'D:\任务归档\电池\研究\data\原始xlsx'

# 检查路径是否存在
if not os.path.isdir(folder_a_path):
    print(f"错误：文件夹A路径不存在: {folder_a_path}")
    sys.exit()  # 如果路径不正确，则退出程序
if not os.path.isdir(folder_b_path):
    print(f"错误：文件夹B路径不存在: {folder_b_path}")
    sys.exit()  # 如果路径不正确，则退出程序

# --- 2. 核心处理逻辑 ---
# 假设电池编号从 1 到 24
num_batteries = 24

print("--- 开始处理 ---")

for i in range(1, num_batteries + 1):
    try:
        # --- 3. 构建文件和文件夹路径 ---
        # 请根据您的实际命名规则调整 'battery' 前缀
        csv_file_name = f'battery{i}_SOH健康特征提取结果.csv'
        csv_file_path = os.path.join(folder_a_path, csv_file_name)

        battery_subfolder_path = os.path.join(folder_b_path, f'battery{i}')

        # 检查文件和文件夹是否存在，如果不存在则跳过
        if not os.path.exists(csv_file_path):
            print(f"警告: CSV文件未找到，跳过: {csv_file_path}")
            continue
        if not os.path.exists(battery_subfolder_path):
            print(f"警告: 电池数据文件夹未找到，跳过: {battery_subfolder_path}")
            continue

        # --- 4. 在子文件夹中找到第一个XLSX文件 ---
        # 获取所有文件名并排序，确保 '...48.xlsx' 在 '...48_1.xlsx' 之前
        all_files_in_subdir = sorted(os.listdir(battery_subfolder_path))

        first_xlsx_file = None
        for file in all_files_in_subdir:
            # 找到第一个以 .xlsx 结尾的文件
            if file.endswith('.xlsx'):
                first_xlsx_file = file
                break

        if first_xlsx_file is None:
            print(f"警告: 在文件夹 {battery_subfolder_path} 中没有找到任何XLSX文件，跳过。")
            continue

        xlsx_file_path = os.path.join(battery_subfolder_path, first_xlsx_file)

        print(f"\n正在处理 battery{i}...")
        print(f"读取CSV: {csv_file_name}")
        print(f"读取XLSX: {first_xlsx_file}")

        # --- 5. 使用pandas读取数据 ---
        # 读取CSV文件
        df_csv = pd.read_csv(csv_file_path)

        # 读取XLSX文件中的'cycle'工作表
        df_excel_cycle = pd.read_excel(xlsx_file_path, sheet_name='cycle')

        # --- 6. 数据合并 ---
        # 从Excel数据中只提取我们需要的列，以避免列名冲突
        df_capacity = df_excel_cycle[['循环号', '放电容量(Ah)']]

        # 使用'left'方式合并。这会保留CSV中的所有行，
        # 并根据匹配的“循环号”从Excel数据中添加“放电容量(Ah)”。
        # 如果某个循环号在Excel中不存在，对应的值将为NaN（空值）。
        df_merged = pd.merge(df_csv, df_capacity, on='循环号', how='left')

        # --- 7. 保存更新后的CSV文件 ---
        # index=False 表示不将DataFrame的索引写入文件
        # encoding='utf-8-sig' 确保包含中文字符的CSV能被Excel正确打开
        df_merged.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

        print(f"成功: {csv_file_name} 已更新。")

    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查 battery{i} 的相关文件路径和名称。")
    except KeyError as e:
        print(f"错误: 在处理 battery{i} 的文件时，找不到指定的列: {e}。请检查CSV或XLSX文件中的列名是否正确。")
    except Exception as e:
        print(f"处理 battery{i} 时发生未知错误: {e}")

print("\n--- 所有文件处理完毕 ---")