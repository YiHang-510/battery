import pandas as pd
import os
import re


def merge_battery_data_robust(source_folder, output_folder):
    """
    【修正版】合并指定文件夹下按电池和循环命名的CSV文件。
    此版本通过在合并前排序来保证数据顺序的正确性。

    :param source_folder: str, 包含原始CSV文件的文件夹路径。
    :param output_folder: str, 用于保存合并后文件的文件夹路径。
    """
    # 步骤 1: 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    print(f"输出文件夹 '{output_folder}' 已准备就绪。")

    # 步骤 2: 创建一个字典来存储每个电池的数据
    # 结构: { battery_id: [(cycle_id, DataFrame), ...], ... }
    battery_data_collector = {}

    # 步骤 3: 遍历源文件夹中的所有文件
    print(f"开始扫描源文件夹 '{source_folder}'...")
    for filename in os.listdir(source_folder):
        match = re.match(r'battery_(\d+)_cycle_(\d+)\.csv', filename)

        if match:
            # 提取电池ID和循环ID
            battery_id = int(match.group(1))
            cycle_id = int(match.group(2))

            print(f"找到文件: {filename} (电池ID: {battery_id}, 循环ID: {cycle_id})")

            file_path = os.path.join(source_folder, filename)

            try:
                # 读取CSV，并命名唯一的列
                df_cycle = pd.read_csv(file_path, header=None, names=['弛豫段电压'])

                # 【重要】这里不再立即添加列，而是将循环号和数据作为一个元组储存
                if battery_id not in battery_data_collector:
                    battery_data_collector[battery_id] = []
                battery_data_collector[battery_id].append((cycle_id, df_cycle))

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    print("\n所有文件扫描完毕，开始合并和保存...")

    # 步骤 4: 遍历收集器，对每个电池的数据进行排序、合并和保存
    if not battery_data_collector:
        print("警告：在源文件夹中没有找到任何匹配的文件。")
        return

    for battery_id, cycle_data_list in battery_data_collector.items():
        # 【核心修正】根据循环号(元组的第一个元素)对列表进行排序
        cycle_data_list.sort(key=lambda x: x[0])

        # 创建一个新的列表，只用来存放排序后的DataFrame
        sorted_dataframes = []
        for cycle_id, df_cycle in cycle_data_list:
            # 在这里才为每个DataFrame添加'循环号'列
            df_cycle['循环号'] = cycle_id
            sorted_dataframes.append(df_cycle)

        # 使用排序后的DataFrame列表进行合并
        final_df = pd.concat(sorted_dataframes, ignore_index=True)

        # 调整列的顺序
        final_df = final_df[['循环号', '弛豫段电压']]

        # 构建输出文件路径
        output_filename = f"relaxation_battery{battery_id}.csv"
        output_path = os.path.join(output_folder, output_filename)

        # 保存到CSV，不包含索引
        final_df.to_csv(output_path, index=False)
        print(f"成功为电池 {battery_id} 创建合并文件: {output_path}")

    print("\n所有任务已完成！")


# --- 如何使用 ---
if __name__ == "__main__":
    # 1. 请将此路径修改为你的原始CSV文件所在的文件夹路径
    source_directory = r'D:\任务归档\电池\研究\data\SCU0628弛豫电压数据\VCC-CCD'   # <-- 修改这里

    # 2. 这是输出文件的保存位置，可以按需修改
    output_directory = r'D:\任务归档\电池\研究\data\弛豫电压\VCC-CCD'  # <-- 修改这里

    # 运行修正后的主函数
    merge_battery_data_robust(source_directory, output_directory)

