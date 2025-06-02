import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#数据提取
folder_path = 'F:\\code\\battery\\data\\battery1'
output_path = 'F:\\code\\battery\\data\\selected_data\\battery1.csv'

columns_to_extract = ["循环号", "电流(A)", "电压(V)", "容量(Ah)", "充电容量(Ah)", "放电容量(Ah)", "dQ/dV(mAh/V)"]
all_data = []

for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_excel(file_path, sheet_name='record', engine='openpyxl')
            df = df[columns_to_extract]  # 提取所需列
            df['文件名'] = filename  # 可选：加一列标记来源文件
            all_data.append(df)
        except Exception as e:
            print(f"无法处理文件 {filename}，错误：{e}")

# 合并所有DataFrame
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"所有数据已成功保存到 {output_path}")
else:
    print("未找到任何可用的xlsx文件或符合条件的数据。")