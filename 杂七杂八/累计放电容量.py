import os
import pandas as pd

def process_csv_files_in_folder(folder_path, num_batteries):
    """
    根据 'battery{i}_...' 的命名规则处理CSV文件，
    计算“累计放电容量(Ah)”，并更新文件。

    Args:
        folder_path (str): 包含CSV文件的文件夹路径。
        num_batteries (int): 电池文件的总数 (例如 24)。
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：找不到文件夹 '{folder_path}'。请确保文件夹路径正确。")
        return

    # 从 1 循环到电池总数 (例如 24)
    for i in range(1, num_batteries + 1):
        # 根据您提供的命名规则构建文件名
        filename = f'battery{i}_SOH健康特征提取结果.csv'
        file_path = os.path.join(folder_path, filename)

        # 在处理前，先检查文件是否存在
        if os.path.exists(file_path):
            print(f"正在处理文件: {file_path}")
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 检查所需列是否存在
                if '放电容量(Ah)' in df.columns:
                    # 计算累计和并创建新列
                    df['累计放电容量(Ah)'] = df['放电容量(Ah)'].cumsum()

                    # 保存回原文件
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    print(f"已成功更新文件: {filename}")
                else:
                    print(f"警告: 在文件 {filename} 中未找到 '放电容量(Ah)' 列，已跳过。")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
        else:
            # 如果文件不存在，则打印警告并跳过
            print(f"警告: 文件 {filename} 不存在，已跳过。")


# --- 使用说明 ---
# 1. 将此脚本与您的文件夹A放在同一个目录下。
# 2. 如果文件夹A不在同一目录下，请修改下面的 'folder_a_path' 变量。
# 3. 如果文件总数不是24，请修改 'total_battery_files' 变量。
# 4. 运行此脚本。

if __name__ == "__main__":
    # 定义包含CSV文件的文件夹路径
    folder_a_path = r'D:\任务归档\电池\研究\data\selected_feature\statistic'

    # 定义要处理的文件总数
    total_battery_files = 24

    process_csv_files_in_folder(folder_a_path, total_battery_files)
    print("\n所有文件处理完毕。")