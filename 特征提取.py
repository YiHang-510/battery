import pandas as pd
import numpy as np
import os
from glob import glob

root_dir = r'/home/scuee_user06/myh/电池/data/xlsx/2'
battery_prefix = "battery"
output_dir = r"/home/scuee_user06/myh/电池/data/feature_results"
os.makedirs(output_dir, exist_ok=True)

def extract_features(df, sample_period=1):
    results = []
    for cyc, data in df.groupby('循环号'):
        # ICA峰值及其位置
        if 'dQ/dV(mAh/V)' in data.columns:
            ica_peak = data['dQ/dV(mAh/V)'].max()
            ica_peak_idx = data['dQ/dV(mAh/V)'].idxmax()
            ica_peak_voltage = data.loc[ica_peak_idx, '电压(V)']
        else:
            ica_peak = ica_peak_voltage = np.nan

        # 2.8~3.4V放电面积和时间
        discharge_mask = (data['工步类型'].str.contains('放电')) & (data['电压(V)'] >= 2.8) & (data['电压(V)'] <= 3.4)
        discharge_seg = data[discharge_mask]
        if len(discharge_seg) > 1:
            discharge_area = discharge_seg['容量(Ah)'].max() - discharge_seg['容量(Ah)'].min()
            discharge_time = len(discharge_seg) * sample_period
        else:
            discharge_area = discharge_time = np.nan

        # 恒流充电时间
        cc_charge_mask = (data['工步类型'] == '恒流充电')
        cc_charge_time = len(data[cc_charge_mask]) * sample_period

        # 恒压充电时间
        cv_charge_mask = (data['工步类型'] == '恒压充电')
        cv_charge_time = len(data[cv_charge_mask]) * sample_period

        # 恒流与恒压充电时间比值
        cc_cv_ratio = cc_charge_time / cv_charge_time if cv_charge_time > 0 else np.nan

        # 3.3~3.6V充电时间
        cv_window_mask = (data['工步类型'].str.contains('充电')) & (data['电压(V)'] >= 3.3) & (data['电压(V)'] <= 3.6)
        cv_window_time = len(data[cv_window_mask]) * sample_period

        # 最大容量（本循环）
        max_cap = data['容量(Ah)'].max()

        results.append({
            '循环号': cyc,
            '最大容量(Ah)': max_cap,
            'ICA峰值': ica_peak,
            'ICA峰值位置(V)': ica_peak_voltage,
            '2.8~3.4V放电面积(Ah)': discharge_area,
            '恒流充电时间(s)': cc_charge_time,
            '恒压充电时间(s)': cv_charge_time,
            '恒流与恒压时间比值': cc_cv_ratio,
            '2.8~3.4V放电时间(s)': discharge_time,
            '3.3~3.6V充电时间(s)': cv_window_time,
        })
    return pd.DataFrame(results)

# 扫描Battery*目录
for battery_folder in sorted(glob(os.path.join(root_dir, f"{battery_prefix}*"))):
    if not os.path.isdir(battery_folder):
        continue
    battery_id = os.path.basename(battery_folder)
    print(f"Processing {battery_id} ...")
    # 合并该电池下所有xlsx的record表
    all_df = []
    for file in glob(os.path.join(battery_folder, "*.xlsx")):
        try:
            df = pd.read_excel(file, sheet_name="record")
            all_df.append(df)
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
    if not all_df:
        print(f"{battery_id} 无数据，跳过。")
        continue
    df_all = pd.concat(all_df, ignore_index=True)
    # 特征提取
    feature_df = extract_features(df_all)
    feature_path = os.path.join(output_dir, f"{battery_id}_SOH健康特征提取结果.csv")
    feature_df.to_csv(feature_path, index=False, encoding='gbk')
    print(f"{battery_id} 完成，特征保存在 {feature_path}")

print("全部提取完成。")
