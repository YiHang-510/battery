import numpy as np
import pandas as pd
import os
import re

# === 1. 读取统计特征文件 ===
df_stat1 = pd.read_csv(r'D:\任务归档\电池\研究\data\selected_feature\relaxation\End\EndVrlx_data_all_battery.csv', encoding="gbk")
df_stat2 = pd.read_csv(r'D:\任务归档\电池\研究\data\selected_feature\statistic\all_batteries_statistical_characteristics.csv', encoding="gbk")
df_stat = pd.merge(df_stat1, df_stat2, on=["电池编号", "循环号"], how="inner")
stat_feature_cols = df_stat.drop(columns=["电池编号", "循环号"]).columns.tolist()

# === 2. 处理每个电池文件 ===
input_dir = r"D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval"
output_dir = r'D:\任务归档\电池\研究\data\selected_feature\combine'
os.makedirs(output_dir, exist_ok=True)


# 获取所有符合命名规则的CSV文件
all_files = [f for f in os.listdir(input_dir) if re.match(r"relaxation_battery\d+\.csv", f)]

for filename in all_files:
    full_path = os.path.join(input_dir, filename)
    try:
        df_seq = pd.read_csv(full_path, encoding="gbk", sep=None, engine='python')  # 自动识别分隔符
    except Exception as e:
        print(f"❌ 无法读取文件 {filename}：{e}")
        continue


    # 自动提取电池编号，例如 battery1、battery12
    match = re.search(r"relaxation_(battery\d+)\.csv", filename)
    if not match:
        print(f"⚠️ 无法解析电池编号: {filename}")
        continue
    battery_id = match.group(1)  # ✅ 正确提取 battery1、battery12 等作为变量

    all_cycles = []
    combined_csv_rows = []

    # 获取该电池的所有循环号
    cycles = df_seq["循环号"].unique()

    for cycle in cycles:
        seq_slice = df_seq[df_seq["循环号"] == cycle]
        if seq_slice.shape[0] != 1200:
            print(f"⚠️ 跳过循环 {cycle}，点数不足 1200")
            continue

        voltage = seq_slice["弛豫段电压"].to_numpy().reshape(-1, 1)  # shape: [1200, 1]

        # 查找统计特征
        stat_row = df_stat[(df_stat["电池编号"] == battery_id) & (df_stat["循环号"] == cycle)]
        if stat_row.shape[0] != 1:
            print(f"⚠️ 统计特征缺失: {battery_id} - 循环 {cycle}")
            continue

        stat_values = stat_row.drop(columns=["电池编号", "循环号"]).to_numpy()
        stat_repeated = np.repeat(stat_values, 1200, axis=0)  # shape: [1200, K]
        combined = np.concatenate([voltage, stat_repeated], axis=1)  # shape: [1200, 1+K]
        all_cycles.append(combined)

        # 保存为 CSV 行
        stat_df = pd.DataFrame(stat_repeated, columns=stat_feature_cols)
        one_cycle_df = pd.DataFrame(voltage, columns=["弛豫段电压"])
        one_cycle_df.insert(0, "循环号", cycle)
        one_cycle_df.insert(0, "电池编号", battery_id)
        combined_df = pd.concat([one_cycle_df, stat_df], axis=1)
        combined_csv_rows.append(combined_df)

    # 保存每个电池的拼接结果
    if all_cycles:
        final_tensor = np.stack(all_cycles)  # shape: [num_cycles, 1200, 1+K]
        np.save(os.path.join(output_dir, f"{battery_id}_combined_features.npy"), final_tensor)
        final_csv = pd.concat(combined_csv_rows, ignore_index=True)
        final_csv.to_csv(os.path.join(output_dir, f"{battery_id}_combined_features.csv"), index=False, encoding="gbk")
        print(f"✅ 已保存 {battery_id}, shape: {final_tensor.shape}")
    else:
        print(f"⚠️ {battery_id} 无有效数据")