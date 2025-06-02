import pandas as pd
import numpy as np
import os

# === 参数设置 ===
i = 1
base_path = f'F:\\code\\battery\\data\\selected_data\\battery{i}'
file_path = f'{base_path}\\battery{i}_record.csv'

# === 读取数据 ===
df = pd.read_csv(file_path, encoding='gbk')

# === 初始化变量 ===
all_relaxation_segments = []   # 每段弛豫时段的所有列数据
EndVrlx = []                   # 每段末端电压
CycleIDs = []                 # 对应循环号

# === 逐循环号处理 ===
for cycle_id, group in df.groupby('循环号'):
    current = group['电流(A)'].values
    voltage = group['电压(V)'].values

    mask = (np.isclose(current, 0, atol=1e-3)) & (voltage >= 4)
    matched_rows = group[mask]

    if len(matched_rows) > 1200:
        matched_rows = matched_rows.iloc[:1200]

    if not matched_rows.empty:
        all_relaxation_segments.append(matched_rows)
        EndVrlx.append(matched_rows['电压(V)'].values[-1])
        CycleIDs.append(cycle_id)

# === 合并 & 保存 CSV + NPY ===
if all_relaxation_segments:
    df_vrlx_all = pd.concat(all_relaxation_segments, ignore_index=True)

    # 保存为 CSV
    csv_path = f'{base_path}\\Vrlx_full_segment.csv'
    df_vrlx_all.to_csv(csv_path, index=False, encoding='gbk')

    # 保存为 NPY（转为 dict 列表或结构化 numpy array）
    npy_path = f'{base_path}\\Vrlx_full_segment.npy'
    np.save(npy_path, df_vrlx_all.to_dict(orient='records'))  # 每行作为一个dict保存

    print(f"完整数据已保存为 CSV: {csv_path}")
    print(f"完整数据也已保存为 NPY: {npy_path}")
else:
    print("未提取到任何符合条件的弛豫时段。")

# === 保存末端电压 CSV + NPY ===
if EndVrlx:
    df_end = pd.DataFrame({'循环号': CycleIDs, '末端电压(V)': EndVrlx})
    df_end.to_csv(f'{base_path}\\EndVrlx.csv', index=False, encoding='gbk')
    np.save(f'{base_path}\\EndVrlx.npy', EndVrlx)
    print(f"末端电压已保存为 CSV 和 NPY")
