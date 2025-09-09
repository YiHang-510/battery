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
import matplotlib
import warnings
import joblib
from torch.cuda.amp import autocast, GradScaler
import itertools
import multiprocessing
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- 1. 配置参数 (保留基础配置) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        # !! 注意: 总的保存路径，每个实验会在此路径下创建子文件夹
        self.base_save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/cyclenet/hyperparam_search'

        # --- 数据集划分 ---
        self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 11, 15, 17, 18, 19, 21, 22, 23, 24]
        self.val_batteries = [5, 10, 13, 19]
        self.test_batteries = [6, 12, 14, 20]

        self.features_from_C = ['恒压充电时间(s)', '3.3~3.6V充电时间(s)']
        self.sequence_feature_dim = 7
        self.sequence_length = 1

        # --- 模型超参数 (将被搜索的参数会在这里被覆盖) ---
        self.meta_cycle_len = 7
        self.d_model = 256
        self.d_ff = 1024
        self.cycle_len = 2000
        self.dropout = 0.2
        self.weight_decay = 0.0001

        # --- 训练参数 (将被搜索的参数会在这里被覆盖) ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.002
        self.patience = 15

        self.seed = 2025
        self.mode = 'both'

        # --- 设备设置 (将在 run_trial 中动态设置) ---
        self.use_gpu = True
        self.device = None  # 初始化为 None


# --- 2. 固定随机种子 ---
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 新的多模态模型定义 ---
class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class CycleNetForSOH(nn.Module):
    def __init__(self, configs):
        super(CycleNetForSOH, self).__init__()
        self.configs = configs
        self.sequence_encoder = nn.Linear(configs.sequence_length * configs.sequence_feature_dim, configs.d_model // 2)
        self.scalar_encoder = nn.Linear(configs.scalar_feature_dim, configs.d_model // 2)
        self.combined_feature_dim = configs.d_model
        self.cycle_queue = RecurrentCycle(
            cycle_len=configs.meta_cycle_len,
            channel_size=self.combined_feature_dim
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        x_seq_flat = x_seq.view(x_seq.size(0), -1)
        seq_embedding = self.sequence_encoder(x_seq_flat)
        scalar_embedding = self.scalar_encoder(x_scalar)
        combined_features = torch.cat((seq_embedding, scalar_embedding), dim=1)
        cycle_index = cycle_number % self.configs.meta_cycle_len
        decycled_features = combined_features - self.cycle_queue(cycle_index, length=1).squeeze(1)
        prediction = self.prediction_head(decycled_features)
        return prediction


# --- 4. 数据集定义 ---
class BatteryMultimodalDataset(Dataset):
    def __init__(self, dataframe, sequence_col, scalar_cols, target_col):
        self.df = dataframe.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.scalar_cols = scalar_cols
        self.target_col = target_col
        self.sequences = np.array(self.df[self.sequence_col].tolist(), dtype=np.float32)
        self.scalars = self.df[self.scalar_cols].values.astype(np.float32)
        self.targets = self.df[self.target_col].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)
        return x_seq, x_scalar, cycle_idx, y


# --- 5. 数据加载和预处理 ---
def load_and_preprocess_data(config):
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries

    for battery_id in sorted(list(set(all_ids))):
        try:
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')
            df_a = pd.read_csv(path_a, sep=',')
            df_c = pd.read_csv(path_c, sep=',')
            df_c.rename(columns=lambda x: x.strip(), inplace=True)
            scalar_df = df_c
            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.sequence_length]
            final_df = pd.merge(sequence_df, scalar_df, on='循环号')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)
        except Exception as e:
            print(f"警告: 处理电池 {battery_id} 时出错: {e}")
            continue

    if not all_battery_data:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '累计放电容量(Ah)'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C

    for col in scalar_feature_cols:
        if col not in full_df.columns:
            raise ValueError(f"特征 '{col}' 不存在。")

    config.scalar_feature_dim = len(scalar_feature_cols)
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_target = StandardScaler()

    all_train_sequences = np.vstack(train_df[sequence_col].values)
    scaler_seq.fit(all_train_sequences)
    scaler_scalar.fit(train_df[scalar_feature_cols])
    scaler_target.fit(train_df[[target_col]])

    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])
        df.loc[:, [target_col]] = scaler_target.transform(df[[target_col]])

    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col)

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar, 'target': scaler_target}
    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练函数 ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        batch_seq = batch_seq.to(device)
        batch_scalar = batch_scalar.to(device)
        batch_cycle_idx = batch_cycle_idx.to(device)
        batch_y = batch_y.to(device).unsqueeze(-1)
        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
                loss = criterion(outputs, batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 7. 验证/测试函数 ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_cycle_indices = [], [], []
    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
            batch_seq = batch_seq.to(device)
            batch_scalar = batch_scalar.to(device)
            batch_cycle_idx = batch_cycle_idx.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_cycle_indices.append(batch_cycle_idx.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    cycle_indices = np.concatenate(all_cycle_indices).flatten()

    metrics = {
        'MSE': mean_squared_error(labels, predictions),
        'MAE': mean_absolute_error(labels, predictions),
        'RMSE': np.sqrt(mean_squared_error(labels, predictions)),
        'R2': r2_score(labels, predictions)
    }
    return avg_loss, metrics, predictions, labels, cycle_indices


# --- 9. 单次实验运行函数 (简化Print) ---
def run_trial(trial_params, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = Config()
    for key, value in trial_params.items():
        setattr(config, key, value)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_gpu else "cpu")

    trial_name = "_".join([f"{k}{v}" for k, v in trial_params.items()]).replace(".", "p")
    config.save_path = os.path.join(config.base_save_path, trial_name)
    os.makedirs(config.save_path, exist_ok=True)

    print(f"[GPU {gpu_id}] 开始: {trial_name}")
    set_seed(config.seed)

    try:
        # --- 修正 1：正确接收 scalers 字典 ---
        train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
        scaler_target = scalers['target']
    except Exception as e:
        print(f"[GPU {gpu_id}] 数据加载失败: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CycleNetForSOH(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    grad_scaler = GradScaler() if 'cuda' in config.device.type else None

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(config.epochs):
        train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)

        # --- 修正 2：用5个变量接收 evaluate 的返回值 ---
        val_loss, _, _, _, _ = evaluate(model, val_loader, criterion, config.device)

        print(f"\r[GPU {gpu_id}] 训练中... Epoch {epoch + 1}/{config.epochs} | Val Loss: {val_loss:.5f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print()
                print(f"[GPU {gpu_id}] 提前停止于 Epoch {epoch + 1}")
                break
    print()

    if best_model_state is None:
        print(f"[GPU {gpu_id}] 训练失败，跳过评估。")
        return

    model.load_state_dict(best_model_state)

    # --- 修正 2 (续)：用5个变量接收 evaluate 的返回值 ---
    _, _, test_preds, test_labels, _ = evaluate(model, test_loader, criterion, config.device)

    test_preds_orig = scaler_target.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    test_labels_orig = scaler_target.inverse_transform(test_labels.reshape(-1, 1)).flatten()

    test_metrics_orig = {
        'MSE': mean_squared_error(test_labels_orig, test_preds_orig),
        'MAE': mean_absolute_error(test_labels_orig, test_preds_orig),
        'RMSE': np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig)),
        'R2': r2_score(test_labels_orig, test_preds_orig)
    }

    result_data = {**trial_params, **test_metrics_orig}
    final_metrics_df = pd.DataFrame([result_data])
    metrics_path = os.path.join(config.save_path, 'final_metrics.csv')
    final_metrics_df.to_csv(metrics_path, index=False)

    print(f"[GPU {gpu_id}] 完成: {trial_name} | R2: {test_metrics_orig['R2']:.4f}")


# --- 10. 主控制脚本和超参数搜索空间 ---
if __name__ == '__main__':
    # -- 1. 定义超参数搜索空间 --
    param_grid = {
        'd_model': [128, 256, 512],
        'd_ff': [256, 512, 1024],
        'dropout': [0.1, 0.2, 0.3],
        'weight_decay': [1e-4, 1e-5, 5e-5],
        'batch_size': [64, 128, 256, 512],
        'learning_rate': [0.001, 0.002, 0.005],
        'patience': [15, 25]
    }

    # -- 2. 生成所有参数组合 --
    keys, values = zip(*param_grid.items())
    trial_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"总共将要进行 {len(trial_list)} 次实验。")

    # -- 3. 设置并行计算 --
    NUM_GPUS = 3
    tasks_to_run = list(enumerate(trial_list))
    available_gpus = list(range(NUM_GPUS))
    running_processes = {}
    base_path = Config().base_save_path
    os.makedirs(base_path, exist_ok=True)

    # -- 4. 任务调度循环 --
    while tasks_to_run or running_processes:
        completed_gpus = []
        for gpu_id, process in running_processes.items():
            if not process.is_alive():
                completed_gpus.append(gpu_id)

        for gpu_id in completed_gpus:
            del running_processes[gpu_id]
            available_gpus.append(gpu_id)

        while available_gpus and tasks_to_run:
            gpu_id = available_gpus.pop(0)
            task_id, params = tasks_to_run.pop(0)
            print(f"调度任务 #{task_id} -> GPU {gpu_id}")
            process = multiprocessing.Process(target=run_trial, args=(params, gpu_id))
            process.start()
            running_processes[gpu_id] = process

        time.sleep(5)

    print("\n所有超参数搜索任务已完成！")

    # -- 5. 汇总所有结果 --
    print("开始汇总所有实验结果...")
    all_results = []

    for trial_dir in os.listdir(base_path):
        metrics_file = os.path.join(base_path, trial_dir, 'final_metrics.csv')
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                all_results.append(df)
            except Exception as e:
                print(f"读取文件 {metrics_file} 失败: {e}")

    if not all_results:
        print("未能收集到任何实验结果。")
    else:
        summary_df = pd.concat(all_results, ignore_index=True)
        summary_df = summary_df.sort_values(by='R2', ascending=False)

        summary_path = os.path.join(base_path, 'hyperparameter_search_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n结果汇总已保存到: {summary_path}")

        best_params_row = summary_df.iloc[0]
        print("\n--- 最佳参数组合 ---")
        print(best_params_row)

        best_params_df = pd.DataFrame([best_params_row])
        best_params_path = os.path.join(base_path, 'best_hyperparameters.csv')
        best_params_df.to_csv(best_params_path, index=False)

        # --- 修正 3：使用正确的变量名打印路径 ---
        print(f"\n最佳参数已单独保存到: {best_params_path}")