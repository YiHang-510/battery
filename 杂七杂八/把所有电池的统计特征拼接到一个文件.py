import os
import pandas as pd
from glob import glob

def batch_calc_cc_cv_ratio(input_folder, output_csv, encoding='gbk'):
    """
    批量处理文件夹内所有csv文件，删除每个文件最后一行，计算恒流与恒压时间比，拼接为总表并导出。
    """
    all_dfs = []
    for file in glob(os.path.join(input_folder, "*.csv")):
        try:
            df = pd.read_csv(file, encoding=encoding)
            # 删除最后一行
            df = df.iloc[:-1, :]
            # 提取 batteryX
            battery_id = os.path.basename(file).split('_')[0]
            df['电池编号'] = battery_id
            all_dfs.append(df)
        except Exception as e:
            print(f"处理 {file} 时出错: {e}")
    # 合并所有
    if all_dfs:
        total_df = pd.concat(all_dfs, ignore_index=True)
        total_df.to_csv(output_csv, index=False, encoding=encoding)
        print(f"全部文件处理完成，已导出到：{output_csv}")
        return total_df
    else:
        print("未找到有效数据文件。")
        return None

# 使用示例
input_folder = r'/home/scuee_user06/myh/电池/data/feature_results'
output_csv = f'{input_folder}/all_batteries_statistical_characteristics.csv'
batch_calc_cc_cv_ratio(input_folder, output_csv)
