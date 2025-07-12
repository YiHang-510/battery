import pandas as pd
import os
import re

# === 参数设置 ===
input_dir = r'D:\任务归档\电池\研究\data\RelaxationVoltage_CSVs'  # 修改为你存放 Battery_xx_RelaxationVoltage.csv 的目录
output_dir = r'D:\任务归档\电池\研究\data\selected_feature\relaxation'   # 修改为你希望保存输出文件的目录
output_dir_2 = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\End'
os.makedirs(output_dir, exist_ok=True)

# === 处理 battery1 到 battery24 ===
for i in range(1, 25):
    file_name = f'Battery_{i:02d}_RelaxationVoltage.csv'
    input_path = os.path.join(input_dir, file_name)

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        continue

    df = pd.read_csv(input_path)

    long_format = pd.DataFrame(columns=["循环号", "弛豫段电压"])
    end_voltage_list = []

    for col in df.columns:
        # 提取循环号
        match = re.search(r'\d+', col)
        if match:
            cycle_id = int(match.group())
            col_data = df[col].dropna()  # 去除 NaN，防止末尾是缺失值
            if not col_data.empty:
                # 添加长格式数据
                temp = pd.DataFrame({
                    "循环号": cycle_id,
                    "弛豫段电压": col_data
                })
                long_format = pd.concat([long_format, temp], ignore_index=True)

                # 提取最后一个电压值
                end_voltage = col_data.iloc[-1]
                end_voltage_list.append({"循环号": cycle_id, "弛豫末端电压": end_voltage})

    # # === 保存长格式数据 ===
    # long_csv_path = os.path.join(output_dir, f'relaxation_battery{i}.csv')
    # long_format.to_csv(long_csv_path, index=False, encoding='gbk')
    # print(f"✅ 保存: {long_csv_path}")

    # === 保存末端电压数据 ===
    if end_voltage_list:
        df_end = pd.DataFrame(end_voltage_list)
        end_csv_path = os.path.join(output_dir_2, f'EndVrlx_battery{i}.csv')
        df_end.to_csv(end_csv_path, index=False, encoding='gbk')
        print(f"✅ 末端电压保存: {end_csv_path}")

