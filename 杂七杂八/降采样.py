import pandas as pd
import os

# --- 输入和输出文件夹的路径 ---
# 源文件所在的文件夹
input_dir = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\new'
# 降采样后文件要保存的文件夹
output_dir = r'D:\任务归档\电池\研究\data\selected_feature\relaxation\Interval-Downsampling_600x'

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# --- 循环处理文件 ---
for i in range(1, 25):
    # 先构建原始文件的完整路径
    file_name_only = f'relaxation_battery{i}.csv' # 这是不带路径的文件名
    input_path = os.path.join(input_dir, file_name_only)

    # 使用 os.path.basename() 或刚才的文件名来构建正确的输出路径
    output_path = os.path.join(output_dir, file_name_only)

    try:
        # 读取CSV文件 (保留您的gbk编码)
        df = pd.read_csv(input_path, encoding='utf-8-sig')

        # 以10倍降采样数据
        df_downsampled = df.iloc[::600, :]

        # 将降采样后的数据保存到新文件夹 (保留您的gbk编码)
        df_downsampled.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f'成功处理并降采样文件：{input_path}')

    except FileNotFoundError:
        print(f"文件未找到: {input_path}。已跳过。")
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")

print("\n所有文件处理完毕！")