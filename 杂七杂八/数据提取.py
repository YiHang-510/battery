import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 数据提取
for i in range(15,25):
    folder_path = f'F:\\code\\battery\\data\\battery{i}'
    output_dir = f'F:\\code\\battery\\data\\selected_data\\battery{i}'
    os.makedirs(output_dir, exist_ok=True)

    # record 表
    record_columns = ["循环号", "电流(A)", "电压(V)", "容量(Ah)", "充电容量(Ah)", "放电容量(Ah)", "dQ/dV(mAh/V)"]
    record_data = []

    # cycle 表
    cycle_columns = ["循环号", "充电容量(Ah)", "放电容量(Ah)"]
    cycle_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            try:
                # record 表
                df_record = pd.read_excel(file_path, sheet_name='record', engine='openpyxl')
                df_record = df_record[record_columns]
                df_record['文件名'] = filename
                record_data.append(df_record)

                # cycle 表
                df_cycle = pd.read_excel(file_path, sheet_name='cycle', engine='openpyxl')
                df_cycle = df_cycle[cycle_columns]
                df_cycle['文件名'] = filename
                cycle_data.append(df_cycle)

            except Exception as e:
                print(f"处理文件 {filename} 时出错：{e}")

    # 保存record
    if record_data:
        combined_record_df = pd.concat(record_data, ignore_index=True)
        record_output_path = os.path.join(output_dir, f'battery{i}_record.csv')
        combined_record_df.to_csv(record_output_path, index=False, encoding='gbk')
        print(f"record 表数据已保存至：{record_output_path}")
    else:
        print("未提取到任何 record 表数据。")

    # 保存cycle
    if cycle_data:
        combined_cycle_df = pd.concat(cycle_data, ignore_index=True)
        cycle_output_path = os.path.join(output_dir, f'battery{i}_cycle.csv')
        combined_cycle_df.to_csv(cycle_output_path, index=False, encoding='gbk')
        print(f"cycle 表数据已保存至：{cycle_output_path}")
    else:
        print("未提取到任何 cycle 表数据。")
