import pandas as pd
import numpy as np
import os
from glob import glob


# --- 2. 更新后的特征提取函数 ---
def extract_features_updated(df):
    """
    根据 record 表提取除最大容量外的所有特征。
    - 时间计算基于'总时间'列，更精确。
    - 放电面积计算基于'放电容量(Ah)'列。
    """
    results = []
    # 预处理：确保'总时间'为数值类型，并按时间排序
    df['总时间'] = pd.to_numeric(df['总时间'], errors='coerce')
    df = df.sort_values(by=['循环号', '总时间'])

    for cyc, data in df.groupby('循环号'):
        charge_data = data[data['工步类型'].str.contains('充电', na=False)]

        # ICA峰值及其位置
        if not charge_data.empty and 'dQ/dV(mAh/V)' in charge_data.columns:
            valid_ica_data = charge_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['dQ/dV(mAh/V)', '电压(V)'])
            if not valid_ica_data.empty:
                ica_peak_idx = valid_ica_data['dQ/dV(mAh/V)'].idxmax()
                ica_peak = valid_ica_data.loc[ica_peak_idx, 'dQ/dV(mAh/V)']
                ica_peak_voltage = valid_ica_data.loc[ica_peak_idx, '电压(V)']
            else:
                ica_peak = ica_peak_voltage = np.nan
        else:
            ica_peak = ica_peak_voltage = np.nan

        # 2.8~3.4V放电面积和时间
        discharge_seg = data[
            (data['工步类型'].str.contains('放电', na=False)) & (data['电压(V)'] >= 2.8) & (data['电压(V)'] <= 3.4)]
        if not discharge_seg.empty:
            discharge_area = discharge_seg['放电容量(Ah)'].max() - discharge_seg['放电容量(Ah)'].min()
            discharge_time = len(discharge_seg)
        else:
            discharge_area = discharge_time = 0

        # 恒流充电时间
        cc_charge_seg = data[data['工步类型'] == '恒流充电']
        cc_charge_time = len(cc_charge_seg)

        # 恒压充电时间
        cv_charge_seg = data[data['工步类型'] == '恒压充电']
        cv_charge_time = len(cv_charge_seg)

        # 恒流与恒压充电时间比值
        cc_cv_ratio = cc_charge_time / cv_charge_time if cv_charge_time > 0 else np.nan

        # 3.3~3.6V充电时间
        charge_window_seg = charge_data[(charge_data['电压(V)'] >= 3.3) & (charge_data['电压(V)'] <= 3.6)]
        cv_window_time = len(charge_window_seg)
        results.append({
            '循环号': cyc,
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

def main(root_dir_xlsx, root_dir_cycle, battery_prefix, output_dir):
        # --- 3. 更新后的主逻辑 ---
    # 扫描Battery*目录
    for battery_folder in sorted(glob(os.path.join(root_dir_xlsx, f"{battery_prefix}*"))):
        if not os.path.isdir(battery_folder):
            continue

        battery_id = os.path.basename(battery_folder)
        print(f"Processing {battery_id} ...")

        # 步骤 A: 从 cycle 文件读取最大容量
        cycle_file_path = os.path.join(root_dir_cycle, f"{battery_id}_cycle.csv")
        capacity_df = None
        if os.path.exists(cycle_file_path):
            try:
                cycle_df_raw = pd.read_csv(cycle_file_path, encoding='gbk')
                max_cap = cycle_df_raw.apply(lambda row: max(row.get('充电容量(Ah)', 0), row.get('放电容量(Ah)', 0)),
                                             axis=1)
                capacity_df = pd.DataFrame({
                    '循环号': cycle_df_raw['循环号'],
                    '最大容量(Ah)': max_cap
                })
            except Exception as e:
                print(f"读取或处理 cycle 文件 {cycle_file_path} 时出错: {e}")
        else:
            print(f"警告: 找不到对应的cycle文件: {cycle_file_path}，'最大容量(Ah)'将为空。")

        # 步骤 B: 读取并合并该电池下所有xlsx的record表 (逻辑与您原脚本一致)
        all_df = []
        for file in glob(os.path.join(battery_folder, "*.xlsx")):
            try:
                df = pd.read_excel(file, sheet_name="record")
                all_df.append(df)
            except Exception as e:
                print(f"读取 {file} 时出错: {e}")

        if not all_df:
            print(f"{battery_id} 无 .xlsx 数据，跳过。")
            continue
        df_all = pd.concat(all_df, ignore_index=True)
        if '工步类型' in df_all.columns:
            df_all['工步类型'] = df_all['工步类型'].str.strip()
        # 步骤 C: 特征提取
        feature_df = extract_features_updated(df_all)

        # 步骤 D: 合并最大容量和其它特征
        if capacity_df is not None:
            final_df = pd.merge(capacity_df, feature_df, on='循环号', how='left')
        else:
            final_df = feature_df  # 如果没有容量数据，则只使用提取的特征

        # 步骤 E: 重新排列列顺序并保存
        column_order = [
            '循环号', '最大容量(Ah)', 'ICA峰值', 'ICA峰值位置(V)',
            '2.8~3.4V放电面积(Ah)', '恒流充电时间(s)', '恒压充电时间(s)',
            '恒流与恒压时间比值', '2.8~3.4V放电时间(s)', '3.3~3.6V充电时间(s)'
        ]
        # 确保所有期望的列都存在，不存在的列会以NaN填充
        final_df = final_df.reindex(columns=column_order)

        feature_path = os.path.join(output_dir, f"{battery_id}_SOH健康特征提取结果.csv")
        final_df.to_csv(feature_path, index=False, encoding='gbk')
        print(f"{battery_id} 完成，特征保存在 {feature_path}")

    print("\n全部提取完成。")

if __name__ == '__main__':
    # --- 1. 配置路径 (请根据您的实际情况修改) ---
    # 'record' xlsx 文件所在的根目录
    root_dir_xlsx = r'/home/scuee_user06/myh/电池/data/xlsx/2'
    # 【新增】存放 'batteryX_cycle.csv' 文件的目录
    # !!! 请务必将此路径修改为您的cycle文件所在的实际文件夹路径 !!!
    root_dir_cycle = r'/home/scuee_user06/myh/电池/data/cycle'

    battery_prefix = "battery"
    output_dir = r"/home/scuee_user06/myh/电池/data/selected_feature/statistic-1"
    os.makedirs(output_dir, exist_ok=True)
    main(root_dir_xlsx, root_dir_cycle, battery_prefix, output_dir)