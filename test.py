import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import warnings
import joblib
from torch.cuda.amp import GradScaler
# autocast 将通过 torch.amp 直接调用
import itertools
import time
import torch.multiprocessing as mp  # MODIFIED: Use torch.multiprocessing
warnings.filterwarnings("ignore", message=".*`torch.cuda.amp.GradScaler.*is deprecated.*")

# --- 1. 配置参数 (已修改) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (修改为三个输入路径) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-400x'  # A文件: 弛豫段电压序列 (1200点/循环)
        self.path_B_scalar = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/End'  # B文件: 弛豫末端电压 (1点/循环)
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'  # C文件: 其他特征和目标 (1行/循环)
        self.save_path = '/home/scuee_user06/myh/电池/data/cyclenet_result-3demision'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        # 这里手动分配电池编号
        self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22]
        self.val_batteries = [5, 11, 17, 23]
        self.test_batteries = [6, 12, 18, 24]

        self.features_from_C = [
            '恒流充电时间(s)',
            '恒压充电时间(s)',
            '恒流与恒压时间比值',
            '3.3~3.6V充电时间(s)'
        ]

        self.sequence_feature_dim = 3

        # --- 模型超参数 (将被网格搜索覆盖) ---
        self.meta_cycle_len = 100
        self.sequence_length = 1
        self.scalar_feature_dim = len(self.features_from_C) + 1
        self.d_model = 256
        self.d_ff = 1024
        self.cycle_len = 2000
        self.dropout = 0.2
        self.use_revin = False
        self.weight_decay = 0.0001

        # --- 训练参数 (将被网格搜索覆盖) ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.002
        self.patience = 15
        self.seed = 2025

        # --- 设备设置 ---
        self.use_gpu = True
        # MODIFIED: Device will be set dynamically in the worker process
        self.device = None


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


# --- 5. 数据加载和预处理 (完全重写) ---
def load_and_preprocess_data(config):
    """加载来自三个文件夹的数据，进行合并、预处理和划分"""
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries

    for battery_id in sorted(list(set(all_ids))):
        try:
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            path_b = os.path.join(config.path_B_scalar, f'EndVrlx_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')
            df_a = pd.read_csv(path_a, sep=',', encoding='gbk')
            df_b = pd.read_csv(path_b, sep=',', encoding='gbk')
            df_c = pd.read_csv(path_c, sep=',', encoding='gbk')

            df_b.rename(columns=lambda x: x.strip(), inplace=True)
            df_c.rename(columns=lambda x: x.strip(), inplace=True)
            scalar_df = pd.merge(df_b, df_c, on='循环号')

            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.sequence_length]

            final_df = pd.merge(sequence_df, scalar_df, on='循环号')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)

        except FileNotFoundError as e:
            print(f"警告: 电池 {battery_id} 的文件未找到，已跳过。错误: {e}")
            continue
        except Exception as e:
            print(f"处理电池 {battery_id} 时出错: {e}")
            continue

    if not all_battery_data:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '最大容量(Ah)'
    sequence_col = 'voltage_sequence'

    sample_b_path = os.path.join(config.path_B_scalar, f'EndVrlx_battery{config.train_batteries[0]}.csv')
    sample_b_df = pd.read_csv(sample_b_path, sep=',', encoding='gbk')
    features_from_B = [col.strip() for col in sample_b_df.columns if col.strip() != '循环号']
    features_from_C = config.features_from_C
    scalar_feature_cols = features_from_B + features_from_C

    for col in scalar_feature_cols:
        if col not in full_df.columns:
            raise ValueError(f"您手动选择的特征 '{col}' 不存在于加载的数据中。")

    config.scalar_feature_dim = len(scalar_feature_cols)
    print(f"已手动选择 {config.scalar_feature_dim} 个标量特征: {scalar_feature_cols}")

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()

    all_train_sequences = np.vstack(train_df[sequence_col].values)
    scaler_seq.fit(all_train_sequences)
    scaler_scalar.fit(train_df[scalar_feature_cols])

    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])

    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)

    # We don't need test_dataset for grid search
    return train_dataset, val_dataset


# --- 6. 训练函数 (已修改) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
            batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device).unsqueeze(-1)

        optimizer.zero_grad()
        # in train_epoch function
        if grad_scaler:
            # 使用新的API，并明确指定设备类型
            with torch.amp.autocast(device_type='cuda'):
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


# --- 7. 验证/测试函数 (已修改) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
            batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
                batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device).unsqueeze(
                    -1)

            outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    r2 = r2_score(labels, predictions)
    return avg_loss, r2


# MODIFIED: This is the new worker function for parallel execution
def run_and_log_experiment(params, base_config, train_dataset, val_dataset, device_id):
    """
    Runs a single experiment on a specified device and returns the results.
    This function is designed to be called by a multiprocessing pool.
    """
    start_time = time.time()

    # --- Create a config instance for this specific job ---
    config = Config()
    # Copy base settings
    config.save_path = base_config.save_path
    config.seed = base_config.seed
    # Apply hyperparameters from the current grid search combination
    for key, value in params.items():
        setattr(config, key, value)

    # --- Set the device for this worker process ---
    config.device = torch.device(f"cuda:{device_id}")

    # --- Set seed for reproducibility in this process ---
    set_seed(config.seed)

    print(f"[Device cuda:{device_id}] Starting experiment with params: {params}")

    # --- DataLoader ---
    # MODIFIED: 将 num_workers 设置为 0 来避免嵌套多进程错误
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = CycleNetForSOH(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

    # --- Training Loop ---
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    epochs_no_improve = 0

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)
        val_loss, val_r2 = evaluate(model, val_loader, criterion, config.device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            epochs_no_improve = 0
            # Note: Saving the model in a parallel grid search can lead to race conditions
            # or excessive disk usage. It's often disabled.
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                # print(f"[Device cuda:{device_id}] Early stopping at epoch {epoch + 1}.")
                break

    end_time = time.time()
    duration = end_time - start_time

    print(
        f"[Device cuda:{device_id}] Finished. Duration: {duration:.2f}s. Best Val Loss: {best_val_loss:.6f}, Best Val R2: {best_val_r2:.4f}")

    # --- Prepare results for logging ---
    result_entry = params.copy()
    result_entry['val_loss'] = best_val_loss
    result_entry['val_r2'] = best_val_r2
    result_entry['duration_seconds'] = duration

    return result_entry


def main():
    # --- STEP 1: Define hyperparameter grid ---
    param_grid = {
        'meta_cycle_len': [30, 50, 70, 100],
        'd_model': [64, 128, 256, 512],
        'd_ff': [128, 256, 512, 1024],
        'dropout': [0.1, 0.2, 0.3],
        'weight_decay': [1e-4, 1e-5],
        'batch_size': [64, 128, 256, 512],
        'learning_rate': [0.0002, 0.0005, 0.001]
    }

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Generated {len(param_combinations)} hyperparameter combinations for grid search.")

    # MODIFIED: Parallel execution setup
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs detected. This script requires GPUs for parallel execution. Exiting.")
        return
    print(f"Found {num_gpus} GPUs. Starting parallel grid search.")

    # --- STEP 2: Pre-load data to share with worker processes ---
    base_config = Config()
    os.makedirs(base_config.save_path, exist_ok=True)

    print("Pre-loading and preprocessing data once...")
    try:
        # Load data once before starting the parallel pool
        train_dataset, val_dataset = load_and_preprocess_data(base_config)
        print("Data pre-loading complete.")
    except Exception as e:
        print(f"Failed to pre-load data, please check paths and files: {e}")
        return

    # --- STEP 3: Prepare arguments for all parallel tasks ---
    task_args = []
    for i, params in enumerate(param_combinations):
        # Assign a GPU to each task in a round-robin fashion
        device_id = i % num_gpus
        task_args.append((params, base_config, train_dataset, val_dataset, device_id))

    # --- STEP 4: Run experiments in parallel using a process pool ---
    # The 'with' statement ensures the pool is properly closed
    with mp.Pool(processes=num_gpus) as pool:
        # starmap distributes the arguments in task_args to the worker function
        results_log = pool.starmap(run_and_log_experiment, task_args)

    # --- STEP 5: Process and save the final results ---
    print("\n\n--- Grid Search Completed ---")
    if not results_log:
        print("No experiments were successfully completed.")
        return

    results_df = pd.DataFrame(results_log)
    results_df = results_df.sort_values(by='val_loss', ascending=True)

    print("Top 10 best parameter combinations (sorted by validation loss):")
    print(results_df.head(10))

    results_csv_path = os.path.join(base_config.save_path, 'grid_search_results_parallel.csv')
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nDetailed results for all {len(results_df)} combinations saved to: {results_csv_path}")
    print("\nBest parameter combination found:")
    print(results_df.iloc[0])


if __name__ == '__main__':
    # MODIFIED: Set multiprocessing start method to 'spawn' for CUDA safety
    # This should be done once at the beginning of the main execution block
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    warnings.filterwarnings('ignore')
    warnings.filterwarnings("ignore", message=".*`torch.cuda.amp.GradScaler.*is deprecated.*")
    matplotlib.use('Agg')  # Use a non-interactive backend for plotting
    main()