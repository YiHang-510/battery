import os
import pandas as pd


def merge_relaxation_voltage(folder_a, folder_b, battery_count):
    """
    从文件夹A提取每个循环的弛豫末端电压，并添加到文件夹B的对应文件中。

    Args:
        folder_a (str): 存放 "relaxation_batteryi.csv" 的文件夹路径。
        folder_b (str): 存放 "batteryi_SOH健康特征提取结果.csv" 的文件夹路径。
        battery_count (int): 电池的总数量。
    """
    print("--- 开始处理 ---")

    # --- 请根据您的实际情况修改这里的列名 ---
    # 文件夹A中代表“循环”的列名
    cycle_column_in_a = '循环号'
    # 文件夹B中代表“循环”的列名
    cycle_column_in_b = '循环号'
    # 文件夹A中要提取数据的列名
    voltage_column_in_a = '弛豫段电压'
    # 在文件夹B中要新建的列名
    new_column_name_in_b = '弛豫段电压7'
    # -----------------------------------------

    # 循环处理每一组电池文件
    for i in range(1, battery_count + 1):
        # 构建文件名
        file_a_name = f"relaxation_battery{i}.csv"
        file_b_name = f"relaxation_battery{i}.csv"

        # 构建完整的文件路径
        path_a = os.path.join(folder_a, file_a_name)
        path_b = os.path.join(folder_b, file_b_name)

        print(f"\n正在处理电池 {i}:")
        print(f"  - 读取A文件: {path_a}")
        print(f"  - 读取B文件: {path_b}")

        try:
            # 读取两个对应的CSV文件
            df_a = pd.read_csv(path_a)
            df_b = pd.read_csv(path_b)

            # --- 核心步骤 1: 从A文件中提取数据 ---
            # 按“循环”分组，并获取“弛豫段电压”列的最后一个值
            # 结果是一个以“循环”号为索引，末端电压为值的Series
            last_voltages = df_a.groupby(cycle_column_in_a)[voltage_column_in_a].last()

            # --- 核心步骤 2: 将数据映射并添加到B文件 ---
            # 使用B文件中的循环号，去last_voltages中查找对应的电压值
            df_b[new_column_name_in_b] = df_b[cycle_column_in_b].map(last_voltages)

            # --- 核心步骤 3: 保存修改后的B文件 ---
            # index=False表示不将DataFrame的索引写入文件
            # encoding='utf-8-sig'确保中文在Excel中正常显示
            df_b.to_csv(path_b, index=False, encoding='utf-8-sig')

            print(f"  -> 成功！已将“{new_column_name_in_b}”更新至 {file_b_name}")

        except FileNotFoundError:
            print(f"  -> 错误: 找不到文件。请检查文件是否存在于以下路径：")
            print(f"    - {path_a}")
            print(f"    - {path_b}")
            continue  # 跳过当前循环，继续处理下一个
        except KeyError as e:
            print(f"  -> 错误: 找不到列名 {e}。请检查脚本中设置的列名是否与CSV文件中的一致。")
            continue
        except Exception as e:
            print(f"  -> 处理电池 {i} 时发生未知错误: {e}")
            continue

    print("\n--- 全部处理完成！ ---")


if __name__ == '__main__':
    # --- 请在这里配置您的文件夹路径和电池数量 ---

    # 1. 文件夹A的路径 ("relaxation_batteryi.csv" 所在位置)
    folder_a_path = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval'

    # 2. 文件夹B的路径 ("batteryi_SOH健康特征提取结果.csv" 所在位置)
    folder_b_path = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval-singleraw-200x'

    # 3. 电池的总数量
    total_batteries = 24

    # 执行主函数
    merge_relaxation_voltage(folder_a_path, folder_b_path, total_batteries)