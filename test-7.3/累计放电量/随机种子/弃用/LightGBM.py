import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import shutil


# --- 1. 配置参数 (保持不变) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/lightgbm_direct/6'  # 修改了保存路径以作区分

        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 11, 13, 14, 15, 18, 21, 22, 23, 24]
        # self.val_batteries = [5, 10, 17, 19]
        # self.test_batteries = [6, 12, 16, 20]

        self.train_batteries = [1, 2, 3, 4]
        self.val_batteries = [5]
        self.test_batteries = [6]

        # self.train_batteries = [7, 8, 9, 11]
        # self.val_batteries = [10]
        # self.test_batteries = [12]

        # self.train_batteries = [15, 13, 18, 14]
        # self.val_batteries = [17]
        # self.test_batteries = [16]
        #
        # self.train_batteries = [21, 22, 23, 24]
        # self.val_batteries = [19]
        # self.test_batteries = [20]
        # --- 特征选择 ---
        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7
        self.sequence_length = 1

        # --- LightGBM 模型超参数 ---
        self.lgbm_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 2025,
            'boosting_type': 'gbdt',
        }
        self.early_stopping_rounds = 50

        # --- 实验设置 ---
        self.num_runs = 5


# --- 2. 固定随机种子 (保持不变) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# --- 3. 数据加载和预处理 (已修改) ---
def load_and_preprocess_data(config):
    """加载数据，合并，并为GBDT模型进行预处理（直接预测累计容量）"""
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries

    for battery_id in sorted(list(set(all_ids))):
        try:
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')

            df_a = pd.read_csv(path_a)
            df_c = pd.read_csv(path_c)
            df_c.rename(columns=lambda x: x.strip(), inplace=True)

            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values.flatten()).reset_index(
                name='voltage_sequence')

            final_df = pd.merge(sequence_df, df_c, on='循环号')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)
        except Exception as e:
            print(f"处理电池 {battery_id} 时出错: {e}")
            continue

    if not all_battery_data:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_battery_data, ignore_index=True).sort_values(by=['battery_id', '循环号'])

    # --- 核心改动：直接使用累计容量作为目标 ---
    target_col = '累计放电容量(Ah)'

    # --- 特征合并 (保持不变) ---
    seq_features = pd.DataFrame(full_df['voltage_sequence'].to_list(),
                                columns=[f'v_seq_{i}' for i in
                                         range(config.sequence_feature_dim * config.sequence_length)])
    scalar_feature_cols = config.features_from_C
    all_feature_cols = scalar_feature_cols + seq_features.columns.tolist()
    features_df = pd.concat([full_df[['battery_id', '循环号']], full_df[scalar_feature_cols], seq_features], axis=1)

    # 划分数据集
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # --- 特征和目标缩放 ---
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()  # <--- 新增对目标的缩放器

    # 训练特征缩放器
    X_train_features = features_df.loc[train_df.index, all_feature_cols]
    scaler_features.fit(X_train_features)

    # 训练目标缩放器
    scaler_target.fit(train_df[[target_col]])

    # 应用特征缩放
    scaled_features = pd.DataFrame(scaler_features.transform(features_df[all_feature_cols]), columns=all_feature_cols,
                                   index=features_df.index)

    # 准备 X
    X_train = scaled_features.loc[train_df.index]
    X_val = scaled_features.loc[val_df.index]
    X_test = scaled_features.loc[test_df.index]

    # 应用目标缩放并准备 y
    y_train = scaler_target.transform(train_df[[target_col]]).flatten()
    y_val = scaler_target.transform(val_df[[target_col]]).flatten()
    # y_test 我们直接从原始 test_df 中获取，以便评估

    # 将缩放器打包返回
    scalers = {'features': scaler_features, 'target': scaler_target}

    return X_train, y_train, X_val, y_val, X_test, test_df, scalers


# --- 4. 可视化函数 (保持不变) ---
def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predictions', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Cumulative Discharge Capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


def plot_diagonal_results(labels, preds, title, save_path):
    plt.figure(figsize=(8, 8))
    min_val = min(np.min(labels), np.min(preds)) * 0.98
    max_val = max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('True Cumulative Discharge Capacity (Ah)', fontsize=12)
    plt.ylabel('Predicted Cumulative Discharge Capacity (Ah)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 5. 主执行函数 (已修改) ---
def main():
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    config = Config()

    os.makedirs(config.save_path, exist_ok=True)
    print(f"所有实验的总保存路径: {config.save_path}")

    all_runs_metrics = []
    best_run_mae = float('inf')
    best_run_dir = None
    best_run_number = -1

    for run_number in range(1, config.num_runs + 1):
        current_seed = random.randint(0, 99999)
        set_seed(current_seed)
        config.lgbm_params['seed'] = current_seed

        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)

        print(f"\n{'=' * 30}")
        print(f" 开始第 {run_number}/{config.num_runs} 次实验 | 随机种子: {current_seed} ")
        print(f" 本次实验结果将保存到: {run_save_path}")
        print(f"{'=' * 30}")

        # --- 1. 数据加载 ---
        try:
            X_train, y_train, X_val, y_val, X_test, test_df_orig, scalers = load_and_preprocess_data(config)
            joblib.dump(scalers, os.path.join(run_save_path, 'scalers.pkl'))
        except Exception as e:
            print(f"数据加载失败: {e}")
            continue

        print(f"数据加载完成。训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

        # --- 2. 模型训练 ---
        model = lgb.LGBMRegressor(**config.lgbm_params)

        print("\n开始训练 LightGBM 模型...")
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='mae',
                  callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=True)])

        joblib.dump(model, os.path.join(run_save_path, 'best_model.pkl'))
        print("\n训练完成。")

        # --- 3. 评估模型 (已修改) ---
        print('\n加载最佳模型进行最终评估...')
        # 直接预测缩放后的累计容量
        test_preds_scaled = model.predict(X_test)

        # --- 核心改动：对预测结果进行反归一化 ---
        scaler_target = scalers['target']
        test_preds_orig = scaler_target.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

        # 获取真实的累计容量
        test_labels_orig = test_df_orig['累计放电容量(Ah)'].values

        # 钳制负值预测
        test_preds_orig = np.clip(test_preds_orig, a_min=0.0, a_max=None)

        # --- 4. 分电池计算指标和绘图 (逻辑微调) ---
        print("\n--- 本轮评估结果 (按单电池) ---")

        # 将预测结果添加到原始的 test_df 中，方便按电池ID分组
        eval_df = test_df_orig.copy()
        eval_df['prediction'] = test_preds_orig

        per_battery_metrics_list = []
        for batt_id in config.test_batteries:
            batt_df = eval_df[eval_df['battery_id'] == batt_id]
            if batt_df.empty:
                continue

            batt_true = batt_df['累计放电容量(Ah)'].values
            batt_pred = batt_df['prediction'].values

            batt_metrics = {
                'Battery_ID': batt_id,
                'MAE': mean_absolute_error(batt_true, batt_pred),
                'MAPE': mean_absolute_percentage_error(batt_true, batt_pred),
                'MSE': mean_squared_error(batt_true, batt_pred),
                'RMSE': np.sqrt(mean_squared_error(batt_true, batt_pred)),
                'R2': r2_score(batt_true, batt_pred)
            }
            per_battery_metrics_list.append(batt_metrics)
            print(
                f"  - 电池 {batt_id}: MAE={batt_metrics['MAE']:.4f}, RMSE={batt_metrics['RMSE']:.4f}, R2={batt_metrics['R2']:.4f}")

            plot_results(batt_true, batt_pred, f'Run {run_number} Battery {batt_id}',
                         os.path.join(run_save_path, f'test_plot_battery_{batt_id}.png'))
            plot_diagonal_results(batt_true, batt_pred, f'Run {run_number} Battery {batt_id}',
                                  os.path.join(run_save_path, f'test_diagonal_plot_battery_{batt_id}.png'))

        pd.DataFrame(per_battery_metrics_list).to_csv(os.path.join(run_save_path, 'test_per_battery_metrics.csv'),
                                                      index=False)
        print(f"  -> 单独指标和图表已保存至: {run_save_path}")

        # --- 5. 计算总体指标 (保持不变) ---
        print("\n--- 本轮评估结果 (所有测试电池汇总) ---")
        overall_metrics = {
            'MAE': mean_absolute_error(test_labels_orig, test_preds_orig),
            'MAPE': mean_absolute_percentage_error(test_labels_orig, test_preds_orig),
            'MSE': mean_squared_error(test_labels_orig, test_preds_orig),
            'RMSE': np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig)),
            'R2': r2_score(test_labels_orig, test_preds_orig)
        }
        all_runs_metrics.append({'run': run_number, 'seed': current_seed, **overall_metrics})
        print(
            f"测试集(汇总): MAE={overall_metrics['MAE']:.4f}, RMSE={overall_metrics['RMSE']:.4f}, R2={overall_metrics['R2']:.4f}")

        # --- 6. 检查是否为最佳轮次 (保持不变) ---
        if overall_metrics['MAE'] < best_run_mae:
            best_run_mae = overall_metrics['MAE']
            best_run_dir = run_save_path
            best_run_number = run_number
            print(f"*** 新的最佳表现！测试集 MAE: {best_run_mae:.4f} ***")

    # --- 循环结束后 (保持不变) ---
    print(f"\n\n{'=' * 50}\n 所有实验均已完成。\n{'=' * 50}")

    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 多次实验性能汇总 ---")
        print(summary_df.to_string())
        print(f"\n汇总指标已保存到: {summary_path}")

    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (测试集 MAE 最低: {best_run_mae:.4f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")
        for filename in os.listdir(best_run_dir):
            shutil.copy2(os.path.join(best_run_dir, filename), os.path.join(config.save_path, filename))
        print("最佳结果复制完成。")


if __name__ == '__main__':
    main()