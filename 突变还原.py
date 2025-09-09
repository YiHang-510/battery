import pandas as pd
import numpy as np
import os


def correct_jumps_independently(input_filepath, output_filepath=None, jump_threshold=0.0):
    """
    通过“独立修正”的方式校正电压突变，不使用累积修正值。
    对于每个检测到的突变点，都将其后的所有点减去该突变值。

    Args:
        input_filepath (str): 输入的CSV文件路径。
        output_filepath (str, optional): 保存修正后数据的CSV文件路径。
        jump_threshold (float, optional): 定义正向电压突变的阈值 (V)。
    """
    # --- 1. 文件和路径设置 ---
    if not os.path.exists(input_filepath):
        print(f"错误：文件未找到 '{input_filepath}'")
        return

    if output_filepath is None:
        base, ext = os.path.splitext(input_filepath)
        output_filepath = f"{base}_fixed_simple{ext}"

    # --- 2. 读取数据 ---
    try:
        df = pd.read_csv(input_filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_filepath, encoding='gbk')
    print(f"--- 正在处理文件 (独立修正模式): {input_filepath} ---")

    # --- 3. 核心修正逻辑 (非累积) ---
    df_corrected = df.copy()

    # 按“循环号”分组处理
    for cycle_num, group in df.groupby('循环号'):
        group_copy = df_corrected.loc[group.index].copy()

        # 计算当前组（可能已被部分修正）的电压差
        diffs = group_copy['弛豫段电压'].diff()

        # 查找所有电压增加超过阈值的突变点
        jump_indices = diffs[diffs > jump_threshold].index

        if not jump_indices.empty:
            print(f"  - 在循环 {cycle_num} 中发现 {len(jump_indices)} 个突变点。正在独立修正...")

            # 遍历每一个突变点并独立应用修正
            for jump_index in jump_indices:
                # 重新计算当前点的diff，因为前面的修正可能已经改变了它的值
                current_diff = group_copy.loc[jump_index, '弛豫段电压'] - group_copy.loc[jump_index - 1, '弛豫段电压']

                # 仅当重新计算的差值仍然是突变时才修正
                if current_diff > jump_threshold:
                    jump_value = current_diff

                    # --- 新逻辑 ---
                    # 步骤一：单独处理突变点本身
                    extra_compensation = 0.0001
                    group_copy.loc[jump_index, '弛豫段电压'] -= (jump_value + extra_compensation)

                    # 步骤二：处理突变点之后的所有点
                    indices_to_correct_after = group_copy.loc[jump_index + 1:].index
                    if not indices_to_correct_after.empty:
                        group_copy.loc[indices_to_correct_after, '弛豫段电压'] -= jump_value + 0.0001

            # 将修正后的group_copy更新回主DataFrame
            df_corrected.loc[group_copy.index] = group_copy

    # --- 4. 保存修正后的数据 ---
    df_corrected.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    print(f"--- 修正完成。已保存至: {output_filepath} ---\n")


# --- 如何运行 ---
if __name__ == '__main__':
    # 1. 将您的CSV文件名填入下方
    input_file = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval\relaxation_battery8.csv'

    # 2. 运行此脚本
    correct_jumps_independently(input_file)