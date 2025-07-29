import os
import pandas as pd
import warnings

# 忽略一些Pandas的警告，让输出更整洁
warnings.filterwarnings('ignore', category=FutureWarning)


# --- 1. 复用原项目的配置 ---
# 我们只保留数据加载所需的部分，让这个分析脚本保持简洁
class Config:
    def __init__(self):
        # --- 数据路径和特征定义 ---
        # !!! 请确保这里的路径和您主项目中的路径一致 !!!
        self.path_C_features = r'D:\任务归档\电池\研究\data\selected_feature\statistic'

        # --- 数据集划分 (包含所有电池，以加载全部数据) ---

        self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22]
        self.val_batteries = [5, 11, 17, 23]
        self.test_batteries = [6, 12, 18, 24]

        # self.train_batteries = [1, 2, 3, 4]
        # self.val_batteries = [5]
        # self.test_batteries = [6]

        # self.train_batteries = [7, 8, 9, 10]
        # self.val_batteries = [11]
        # self.test_batteries = [12]

        # self.train_batteries = [13, 14, 15, 16]
        # self.val_batteries = [17]
        # self.test_batteries = [18]

        # self.train_batteries = [19, 20, 21, 22]
        # self.val_batteries = [23]
        # self.test_batteries = [24]
        # 如果您想分析所有24个电池，可以直接用下面的列表
        # self.all_batteries = list(range(1, 25))

        # --- 您要分析的特征和目标 ---
        # 文件C中的特征列名
        self.features_from_C = [
            'ICA峰值位置(V)',
            '恒流充电时间(s)',
            '恒压充电时间(s)',
            '恒流与恒压时间比值',
            '2.8~3.4V放电时间(s)',
            '3.3~3.6V充电时间(s)'
        ]
        # 目标列名
        self.target_col = '最大容量(Ah)'


# --- 2. 数据加载函数 ---
def load_all_c_files(config: Config) -> pd.DataFrame:
    """
    加载并合并所有指定电池的文件C数据。
    """
    all_battery_data = []

    # 获取所有电池的ID
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries
    # 如果您使用了 all_batteries, 就用下面这行
    # all_ids = config.all_batteries

    print(f"开始加载以下电池的数据: {sorted(list(set(all_ids)))}")

    for battery_id in sorted(list(set(all_ids))):
        try:
            # 拼接文件路径
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')

            # 加载数据，使用 'gbk' 编码
            df_c = pd.read_csv(path_c, sep=',', encoding='gbk')

            # 清理列名的前后空格，这是一个好习惯
            df_c.rename(columns=lambda x: x.strip(), inplace=True)

            all_battery_data.append(df_c)

        except FileNotFoundError:
            print(f"  -> 警告: 电池 {battery_id} 的文件未找到，已跳过。")
            continue
        except Exception as e:
            print(f"  -> 处理电池 {battery_id} 时出错: {e}")
            continue

    if not all_battery_data:
        print("错误：未能成功加载任何电池数据。请检查路径和文件名。")
        return pd.DataFrame()

    # 将所有数据合并成一个DataFrame
    full_df = pd.concat(all_battery_data, ignore_index=True)
    print("所有文件数据加载并合并完成。")
    return full_df


# --- 3. 相关性分析函数 ---
def analyze(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    计算并打印特征与目标之间的皮尔逊和斯皮尔曼相关系数。
    """
    print("\n--- 特征与最大容量的相关性分析 ---")

    # 检查所有需要的列是否存在
    required_cols = feature_cols + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误：数据中缺少以下列，无法进行分析: {missing_cols}")
        return

    # 仅保留需要分析的列
    analysis_df = df[required_cols]

    # --- 计算皮尔逊相关系数 ---
    # 皮尔逊相关性：衡量线性关系的强度和方向。范围从-1到+1。
    pearson_corr = analysis_df.corr(method='pearson')
    pearson_results = pearson_corr[target_col].drop(target_col)  # 提取与目标的相关性，并移除自身

    # --- 计算斯皮尔曼相关系数 ---
    # 斯皮尔曼相关性：衡量单调关系的强度和方向（不一定是线性的）。它基于数据的排序。
    spearman_corr = analysis_df.corr(method='spearman')
    spearman_results = spearman_corr[target_col].drop(target_col)

    # --- 整理并展示结果 ---
    results_df = pd.DataFrame({
        '皮尔逊(Pearson)': pearson_results,
        '斯皮尔曼(Spearman)': spearman_results
    })

    # 根据皮尔逊相关系数的绝对值进行降序排序，更容易看出关键特征
    sorted_results = results_df.reindex(results_df['皮尔逊(Pearson)'].abs().sort_values(ascending=False).index)

    print(sorted_results)
    print("\n说明:")
    print(" - 数值越接近 1 或 -1，表示相关性越强。")
    print(" - 正值表示正相关（特征增加，容量也倾向于增加）。")
    print(" - 负值表示负相关（特征增加，容量倾向于减少）。")
    print(" - 数值接近 0 表示相关性很弱。")


# --- 4. 主执行函数 ---
if __name__ == '__main__':
    # 初始化配置
    config = Config()

    # 加载数据
    all_data = load_all_c_files(config)

    # 确保数据已成功加载再进行分析
    if not all_data.empty:
        # 执行分析
        analyze(all_data, config.features_from_C, config.target_col)