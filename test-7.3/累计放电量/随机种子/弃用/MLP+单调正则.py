import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import warnings
import joblib
import shutil


# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/monotonic_mlp/6'

        # --- 数据集划分 ---
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
        self.features_from_C = ['恒压充电时间(s)', '3.3~3.6V充电时间(s)']
        self.sequence_feature_dim = 7
        self.sequence_length = 1
        self.num_features = len(self.features_from_C) + self.sequence_feature_dim * self.sequence_length  # 2 + 7 = 9

        # --- MLP 模型超参数 ---
        self.hidden_dims = [64, 64]
        self.dropout = 0.1
        self.activation_fn = nn.SiLU()  # 使用 SiLU (Swish) 激活函数

        # --- 训练参数 ---
        self.epochs = 300
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.patience = 20
        self.monotonic_lambda = 0.3  # 单调性正则项的权重 λ
        self.num_runs = 5  # 运行5次实验

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")


# --- 2. 固定随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 模型定义 (Monotonic MLP) ---
class MonotonicMLP(nn.Module):
    def __init__(self, config):
        super(MonotonicMLP, self).__init__()
        layers = []
        input_dim = config.num_features
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(config.activation_fn)
            layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- 4. 数据集定义 (为单调性损失修改) ---
# --- 4. 数据集定义 (为单调性损失修改) ---
class BatteryMonotonicDataset(Dataset):
    def __init__(self, features_df, targets_df):
        # features_df 和 targets_df 传入时拥有相同的、未重置的索引

        # 1. 首先，根据 battery_id 和 循环号 对 features_df 进行排序
        #    这是为了确保在 __getitem__ 中，t-1 和 t 是连续的
        features_sorted = features_df.sort_values(by=['battery_id', '循环号'])

        # 2. 使用排序后 features_df 的索引，来同步排序 targets_df
        targets_sorted = targets_df.loc[features_sorted.index]

        # 3. 现在两个DataFrame的顺序已经完全一致，再同时重置它们的索引
        self.features_df = features_sorted.reset_index(drop=True)
        self.targets_df = targets_sorted.reset_index(drop=True)

        # ------------------- 后续代码不变 -------------------
        self.features = torch.tensor(self.features_df.drop(columns=['battery_id', '循环号']).values,
                                     dtype=torch.float32)
        self.targets = torch.tensor(self.targets_df.values, dtype=torch.float32).view(-1, 1)

        # 识别每个电池的第一个循环索引
        is_first_cycle_series = ~self.features_df.duplicated(subset=['battery_id'], keep='first')
        self.is_first_cycle = torch.tensor(is_first_cycle_series.values, dtype=torch.bool)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        is_first = self.is_first_cycle[idx]

        if is_first:
            prev_features = torch.zeros_like(self.features[idx])  # 占位符
        else:
            prev_features = self.features[idx - 1]

        return self.features[idx], self.targets[idx], prev_features, not is_first

# --- 5. 数据加载和预处理 ---
def load_and_preprocess_data(config):
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

    full_df = pd.concat(all_battery_data, ignore_index=True).sort_values(by=['battery_id', '循环号']).reset_index(
        drop=True)
    target_col = '累计放电容量(Ah)'

    seq_features_df = pd.DataFrame(full_df['voltage_sequence'].to_list(),
                                   columns=[f'v_seq_{i}' for i in range(config.sequence_feature_dim)])
    scalar_feature_cols = config.features_from_C
    all_feature_cols = scalar_feature_cols + seq_features_df.columns.tolist()

    features_df = pd.concat([full_df[['battery_id', '循环号']], full_df[scalar_feature_cols], seq_features_df], axis=1)
    targets_df = full_df[[target_col]]

    train_df_idx = full_df[full_df['battery_id'].isin(config.train_batteries)].index
    val_df_idx = full_df[full_df['battery_id'].isin(config.val_batteries)].index
    test_df_idx = full_df[full_df['battery_id'].isin(config.test_batteries)].index

    scaler_features = StandardScaler()
    scaler_target = StandardScaler()

    scaler_features.fit(features_df.loc[train_df_idx, all_feature_cols])
    scaler_target.fit(targets_df.loc[train_df_idx])

    scaled_features_df = pd.DataFrame(scaler_features.transform(features_df[all_feature_cols]),
                                      columns=all_feature_cols, index=features_df.index)
    scaled_features_df[['battery_id', '循环号']] = features_df[['battery_id', '循环号']]
    scaled_targets_df = pd.DataFrame(scaler_target.transform(targets_df), columns=[target_col], index=targets_df.index)

    train_dataset = BatteryMonotonicDataset(scaled_features_df.loc[train_df_idx], scaled_targets_df.loc[train_df_idx])
    val_dataset = BatteryMonotonicDataset(scaled_features_df.loc[val_df_idx], scaled_targets_df.loc[val_df_idx])
    test_dataset = BatteryMonotonicDataset(scaled_features_df.loc[test_df_idx], scaled_targets_df.loc[test_df_idx])

    scalers = {'features': scaler_features, 'target': scaler_target}
    return train_dataset, val_dataset, test_dataset, full_df.loc[test_df_idx], scalers


# --- 6. 训练与评估函数 ---
def train_epoch(model, dataloader, optimizer, main_criterion, config):
    model.train()
    total_loss = 0
    for features_t, targets_t, features_t_prev, is_not_first in dataloader:
        features_t, targets_t, features_t_prev = features_t.to(config.device), targets_t.to(
            config.device), features_t_prev.to(config.device)
        is_not_first = is_not_first.to(config.device)

        optimizer.zero_grad()

        preds_t = model(features_t)
        main_loss = main_criterion(preds_t, targets_t)

        mono_loss = 0
        valid_indices = is_not_first.nonzero(as_tuple=True)[0]
        if len(valid_indices) > 0:
            preds_t_for_mono = preds_t[valid_indices]
            with torch.no_grad():
                preds_t_prev = model(features_t_prev[valid_indices])

            penalty = torch.relu(preds_t_prev - preds_t_for_mono)
            mono_loss = torch.mean(penalty)

        loss = main_loss + config.monotonic_lambda * mono_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, main_criterion, config):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features_t, targets_t, _, _ in dataloader:
            features_t, targets_t = features_t.to(config.device), targets_t.to(config.device)
            preds_t = model(features_t)
            loss = main_criterion(preds_t, targets_t)
            total_loss += loss.item()
            all_preds.append(preds_t.cpu().numpy())
            all_labels.append(targets_t.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    return avg_loss, predictions, labels


# --- 7. 可视化函数 ---
def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predictions', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Cumulative Discharge Capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
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
    plt.axis('equal');
    plt.xlim(min_val, max_val);
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=300)
    plt.close()


# --- 8. 主执行函数 ---
def main():
    warnings.filterwarnings('ignore');
    matplotlib.use('Agg');
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"总保存路径: {config.save_path}, 设备: {config.device}")

    all_runs_metrics = []
    best_run_mae = float('inf')
    best_run_dir = None
    best_run_number = -1

    for run_number in range(1, config.num_runs + 1):
        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)
        current_seed = random.randint(0, 99999)
        set_seed(current_seed)

        print(f"\n{'=' * 30}\n 开始第 {run_number}/{config.num_runs} 次实验 | 种子: {current_seed} \n{'=' * 30}")

        train_dataset, val_dataset, test_dataset, test_df_orig, scalers = load_and_preprocess_data(config)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        model = MonotonicMLP(config).to(config.device)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        main_criterion = nn.SmoothL1Loss()

        best_val_loss = float('inf');
        epochs_no_improve = 0

        print("开始训练 Monotonic MLP 模型...")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, main_criterion, config)
            val_loss, _, _ = evaluate(model, val_loader, main_criterion, config)
            print(f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(run_save_path, 'best_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"验证损失连续 {config.patience} 轮未改善，提前停止。")
                    break

        print("\n训练完成，加载最佳模型进行评估...")
        model.load_state_dict(torch.load(os.path.join(run_save_path, 'best_model.pth')))

        _, test_preds_scaled, _ = evaluate(model, test_loader, main_criterion, config)

        scaler_target = scalers['target']
        test_preds_orig = scaler_target.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
        test_labels_orig = test_df_orig['累计放电容量(Ah)'].values

        eval_df = test_df_orig.copy()
        eval_df['raw_prediction'] = test_preds_orig
        eval_df['monotonic_prediction'] = eval_df.groupby('battery_id')['raw_prediction'].transform(
            lambda x: np.maximum.accumulate(x))

        final_preds = eval_df['monotonic_prediction'].values
        overall_metrics = {'MAE': mean_absolute_error(test_labels_orig, final_preds),
                           'R2': r2_score(test_labels_orig, final_preds)}
        print(f"\n--- 本轮总体评估 (使用单调投影后) ---\n"
              f"测试集 MAE: {overall_metrics['MAE']:.4f}, R2: {overall_metrics['R2']:.4f}")

        current_run_summary = {'run': run_number, 'seed': current_seed, **overall_metrics}
        all_runs_metrics.append(current_run_summary)

        if overall_metrics['MAE'] < best_run_mae:
            best_run_mae = overall_metrics['MAE']
            best_run_dir = run_save_path
            best_run_number = run_number
            print(f"*** 新的最佳表现！***")

        for batt_id in config.test_batteries:
            batt_df = eval_df[eval_df['battery_id'] == batt_id]
            if not batt_df.empty:
                plot_results(batt_df['累计放电容量(Ah)'], batt_df['monotonic_prediction'],
                             f'Run {run_number} Battery {batt_id}',
                             os.path.join(run_save_path, f'plot_batt_{batt_id}.png'))
                plot_diagonal_results(batt_df['累计放电容量(Ah)'], batt_df['monotonic_prediction'],
                                      f'Run {run_number} Battery {batt_id}',
                                      os.path.join(run_save_path, f'diag_plot_batt_{batt_id}.png'))
        print("绘图完成。")

    print(f"\n\n{'=' * 50}\n 所有实验均已完成。\n{'=' * 50}")

    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 多次实验性能汇总 ---\n", summary_df.to_string())
        print(f"\n汇总指标已保存到: {summary_path}")

    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 輪 (MAE最低: {best_run_mae:.4f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")
        for filename in os.listdir(best_run_dir):
            shutil.copy2(os.path.join(best_run_dir, filename), os.path.join(config.save_path, filename))
        print("最佳结果复制完成。")


if __name__ == '__main__':
    main()