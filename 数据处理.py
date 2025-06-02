import pandas as pd
import numpy as np
import os

# === 参数设置 ===
file_name = 'F:\\code\\battery\\data\\selected_data\\battery1.csv'

# === 读取数据 ===
df = pd.read_excel(file_name, engine='openpyxl')

def str_to_list(x):
    try:
        return eval(x) if isinstance(x, str) else x
    except:
        return []

# 将所有列转换为列表列（确保每个单元格都是一个list）
df['Current'] = df['Current'].apply(str_to_list)
df['Voltage'] = df['Voltage'].apply(str_to_list)
df['Capacity'] = df['Capacity'].apply(str_to_list)

# === 提取弛豫电压 ===
"""
Vrlx:弛豫电压段
EndVrlx:弛豫电压的最后一个点
"""
Vrlx = []
EndVrlx = []

for cycle in range(len(df['Capacity']) - 1):
    current = df.at[cycle, 'Current']
    voltage = df.at[cycle, 'Voltage']

    if not isinstance(current, list) or not isinstance(voltage, list):
        continue

    index_c = [i for i, v in enumerate(current) if v == 0]
    index_v = [i for i, v in enumerate(voltage) if v >= 4]
    index_r = sorted(set(index_c).intersection(index_v))

    v_rlx = [voltage[i] for i in index_r]
    if len(v_rlx) > 1200:
        v_rlx = v_rlx[:1200]

    if v_rlx:
        Vrlx.append(v_rlx)
        EndVrlx.append(v_rlx[-1])

# === 保存结果 ===
np.save('F:\\code\\battery\\data\\selected_data\\battery1\\Vrlx.npy', Vrlx)
np.save('F:\\code\\battery\\data\\selected_data\\battery1\\EndVrlx.npy', EndVrlx)

print(f"已提取 {len(Vrlx)} 条弛豫电压曲线，保存为 Vrlx.npy 和 EndVrlx.npy。")
