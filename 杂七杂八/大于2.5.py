import os
import pandas as pd


def process_and_save_individual_csvs(input_folder, output_folder):
    """
    处理输入文件夹中的每个CSV文件，并将结果保存到输出文件夹中，
    文件名与原文件相同。

    Args:
        input_folder (str): 包含源CSV文件的文件夹路径。
        output_folder (str): 用于保存处理后文件的文件夹路径。
    """
    # --- 1. 检查并创建输出文件夹 ---
    if not os.path.exists(output_folder):
        print(f"输出文件夹 '{output_folder}' 不存在，正在创建...")
        os.makedirs(output_folder)

    # --- 2. 检查输入文件夹 ---
    try:
        all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        if not all_files:
            print(f"在输入文件夹 '{input_folder}' 中没有找到CSV文件。")
            return
    except FileNotFoundError:
        print(f"错误：找不到输入文件夹 '{input_folder}'。请检查路径是否正确。")
        return

    # --- 3. 循环处理每个CSV文件 ---
    for filename in all_files:
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        print(f"正在处理: {filename}...")

        try:
            # 读取CSV文件
            df = pd.read_csv(input_file_path, encoding='gbk')

            # 检查关键列是否存在
            if '最大容量(Ah)' not in df.columns:
                print(f"  -> 警告: 文件中缺少 '最大容量(Ah)' 列，将跳过此文件。")
                continue

            # 筛选“最大容量(Ah)” >= 2.5 的所有行
            high_capacity_df = df[df['最大容量(Ah)'] >= 2.5]

            # 筛选“最大容量(Ah)” < 2.5 的第一行
            low_capacity_first_row = df[df['最大容量(Ah)'] < 2.5].head(1)

            # 合并当前文件的筛选结果
            result_df = pd.concat([high_capacity_df, low_capacity_first_row], ignore_index=True)

            # 如果结果为空，可以选择不生成文件
            if result_df.empty:
                print(f"  -> 提示: '{filename}' 没有符合条件的数据，不生成输出文件。")
                continue

            # --- 4. 将处理结果保存到新文件夹中 ---
            result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            print(f"  -> 已保存至: {output_file_path}")

        except Exception as e:
            print(f"  -> 处理文件 '{filename}' 时发生错误: {e}")

    print("\n所有文件处理完成！")


if __name__ == '__main__':
    # --- 请在这里修改您的文件夹路径 ---

    # 1. 包含原始CSV文件的文件夹
    # Windows示例: 'C:/Users/YourUser/Desktop/data'
    # macOS/Linux示例: '/home/user/documents/my_csvs'
    source_folder_path = r'D:\任务归档\电池\研究\data\selected_feature\statistic-old'

    # 2. 用于存放处理后结果的新文件夹
    # 脚本会自动创建此文件夹（如果不存在）
    destination_folder_path = r'D:\任务归档\电池\研究\data\selected_feature\statistic'  # 例如: 'processed_files'

    # 执行处理函数
    process_and_save_individual_csvs(source_folder_path, destination_folder_path)