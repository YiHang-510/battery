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
import warnings
import joblib
from torch.cuda.amp import autocast, GradScaler

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')


# --- 1. 配置参数 (MODIFIED for single sequence input) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (MODIFIED: Simplified to two paths) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-Downsampling_400x'  # A文件: 弛豫段电压序列 (1200点/循环)
        # --- MODIFIED: We still need a file for the target value (SOH) and cycle number ---
        self.path_C_target = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'  # C文件: For '循环号' and '最大容量(Ah)'
        self.save_path = '/home/scuee_user06/myh/电池/data/cyclenet_result_single_input'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        self.train_batteries = [1, 2, 3, 4]
        self.val_batteries = [5]
        self.test_batteries = [6]

        # --- 模型超参数 (MODIFIED: Restored to original CycleNet params) ---
        self.seq_len = 3  # A文件的序列长度 (弛豫段电压的点数)
        self.pred_len = 1  # 预测长度 (for SOH, this is 1)
        self.enc_in = 1  # 输入特征维度 (only voltage, so 1)
        self.cycle_len = 2000  # 最大循环次数
        self.d_model = 128  # 隐藏层维度
        self.model_type = 'mlp'  # Can be 'linear' or 'mlp'
        self.use_revin = True  # 是否使用可逆实例归一化
        self.weight_decay = 0.0001  # L2正则化

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.005
        self.patience = 40
        self.seed = 2025
        self.mode = 'both'  # 可选 'train', 'validate', 'both'

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")


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


# --- 3. 模型定义 (MODIFIED: Replaced with original CycleNet) ---
class RecurrentCycle(torch.nn.Module):
    # This helper module remains the same.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    """
    Original CycleNet Architecture restored from CycleNet.py
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle_len
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # RevIN: instance normalization
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Remove the cycle of the input data
        x = x - self.cycleQueue(cycle_index, self.seq_len)

        # Forecasting with channel independence (parameters-sharing)
        # Permute for channel-wise linear layer
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Add back the cycle of the output data
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # Instance denormalization
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y


# --- 4. 数据集定义 (MODIFIED: Simplified for sequence data only) ---
class BatterySequenceDataset(Dataset):
    def __init__(self, dataframe, sequence_col, target_col):
        self.df = dataframe.reset_index(drop=True)
        # Pre-convert data to numpy arrays for efficiency
        self.sequences = np.array(self.df[sequence_col].tolist(), dtype=np.float32)
        self.targets = self.df[target_col].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Add a channel dimension to the sequence -> (sequence_length, 1)
        x_seq = torch.from_numpy(self.sequences[idx]).unsqueeze(-1)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)

        # Return only the necessary data
        return x_seq, cycle_idx, y


# --- 5. 数据加载和预处理 (MODIFIED: Simplified for single input) ---
def load_and_preprocess_data(config):
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries

    for battery_id in sorted(list(set(all_ids))):
        try:
            print(f"正在处理电池 {battery_id}...")
            # Path for sequence data (File A)
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            # Path for target/cycle data (File C)
            path_c = os.path.join(config.path_C_target, f'battery{battery_id}_SOH健康特征提取结果.csv')

            # Load data
            df_a = pd.read_csv(path_a, sep=',', encoding='gbk')
            df_c = pd.read_csv(path_c, sep=',', encoding='gbk')
            df_c.rename(columns=lambda x: x.strip(), inplace=True)

            # Keep only cycle number and target from df_c
            df_c = df_c[['循环号', '最大容量(Ah)']]

            # Group sequence data by cycle number
            sequence_df = df_a.groupby('循环号')['弛豫段电压'].apply(lambda x: x.values).reset_index()
            sequence_df.rename(columns={'弛豫段电压': 'voltage_sequence'}, inplace=True)

            # Filter out sequences that don't have the correct length
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.seq_len]

            # Merge sequence data with target data on cycle number
            final_df = pd.merge(sequence_df, df_c, on='循环号')
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

    # Define feature and target
    target_col = '最大容量(Ah)'
    sequence_col = 'voltage_sequence'

    # Split dataset
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # --- Feature Scaling ---
    scaler = StandardScaler()
    # Fit scaler only on the training data sequences
    all_train_sequences = np.vstack(train_df[sequence_col].values).reshape(-1, 1)
    scaler.fit(all_train_sequences)

    # Apply scaling to all datasets
    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler.transform(x.reshape(-1, 1)).flatten())

    # Create Dataset objects
    train_dataset = BatterySequenceDataset(train_df, sequence_col, target_col)
    val_dataset = BatterySequenceDataset(val_df, sequence_col, target_col)
    test_dataset = BatterySequenceDataset(test_df, sequence_col, target_col)

    # Return the single scaler for inverse transform if needed later
    return train_dataset, val_dataset, test_dataset, scaler


# --- 6. 训练函数 (MODIFIED for new data format) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    # Modified data unpacking
    for batch_seq, batch_cycle_idx, batch_y in dataloader:
        batch_seq = batch_seq.to(device)
        batch_cycle_idx = batch_cycle_idx.to(device)
        # Reshape y to match model output: (batch_size, pred_len, enc_in)
        batch_y = batch_y.to(device).unsqueeze(-1).unsqueeze(-1)

        optimizer.zero_grad()

        # --- MODIFIED: Model call with two arguments ---
        if grad_scaler:
            with autocast():
                outputs = model(batch_seq, batch_cycle_idx)
                loss = criterion(outputs, batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs = model(batch_seq, batch_cycle_idx)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# --- 7. 验证/测试函数 (MODIFIED for new data format) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Modified data unpacking
        for batch_seq, batch_cycle_idx, batch_y in dataloader:
            batch_seq = batch_seq.to(device)
            batch_cycle_idx = batch_cycle_idx.to(device)
            # Reshape y to match model output
            batch_y_reshaped = batch_y.to(device).unsqueeze(-1).unsqueeze(-1)

            # --- MODIFIED: Model call with two arguments ---
            outputs = model(batch_seq, batch_cycle_idx)
            loss = criterion(outputs, batch_y_reshaped)

            total_loss += loss.item()
            # Squeeze model output to get predictions
            all_preds.append(outputs.squeeze().cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)

    metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    return avg_loss, metrics, predictions, labels


# --- 8. 可视化和工具函数 (No changes needed) ---
def get_exp_tag(config):
    train_ids = '-'.join([str(i) for i in config.train_batteries])
    val_ids = '-'.join([str(i) for i in config.val_batteries])
    test_ids = '-'.join([str(i) for i in config.test_batteries])
    tag = (
        f"train{train_ids}_val{val_ids}_test{test_ids}_"
        f"ep{config.epochs}_bs{config.batch_size}_lr{config.learning_rate}_"
        f"dm{config.d_model}_type{config.model_type}"
    )
    return tag


def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predictions', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index (Cycle)', fontsize=12)
    plt.ylabel('Max Capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 9. 主执行函数 (MODIFIED to handle new return values) ---
def main():
    config = Config()
    set_seed(config.seed)

    exp_tag = get_exp_tag(config)
    config.save_path = os.path.join(config.save_path, exp_tag)
    os.makedirs(config.save_path, exist_ok=True)
    print(f"本次实验结果将保存到: {config.save_path}")
    print(f"使用设备: {config.device}")

    try:
        # --- MODIFIED: load function now returns a single scaler ---
        train_dataset, val_dataset, test_dataset, scaler = load_and_preprocess_data(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载失败: {e}")
        return
    # Save the single scaler
    joblib.dump(scaler, os.path.join(config.save_path, 'scaler.pkl'))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(f"数据加载完成。训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    model = Model(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

    metrics_log = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if config.mode in ['both', 'train']:
        print("\n开始训练模型...")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, config.device)
            print(
                f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")

            log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                         **{'val_' + k: v for k, v in val_metrics.items()}}
            metrics_log.append(log_entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.save_path, f'best_model_{exp_tag}.pth'))
                print(f"  - 验证损失降低，保存模型。")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break
        print("\n训练完成。")
        pd.DataFrame(metrics_log).to_csv(os.path.join(config.save_path, 'training_log.csv'), index=False)
        if config.mode == 'train':
            return

    if config.mode in ['both', 'validate']:
        print('\n加载最佳模型进行最终评估...')
        model_path = os.path.join(config.save_path, f'best_model_{exp_tag}.pth')
        if not os.path.exists(model_path):
            print(f"错误: 找不到已训练的模型 '{model_path}'。请先在 'train' 或 'both' 模式下运行。")
            return

        model.load_state_dict(torch.load(model_path, map_location=config.device))

        _, val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, config.device)
        _, test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, config.device)

        print("\n--- 评估结果 ---")
        final_metrics = pd.DataFrame([{'set': 'validation', **val_metrics}, {'set': 'test', **test_metrics}])
        final_metrics.to_csv(os.path.join(config.save_path, 'final_evaluation_metrics.csv'), index=False)
        print(
            f"最终验证集指标: MSE={val_metrics['MSE']:.6f}, MAE={val_metrics['MAE']:.6f}, RMSE={val_metrics['RMSE']:.6f}, R2={val_metrics['R2']:.4f}")
        print(
            f"最终测试集指标: MSE={test_metrics['MSE']:.6f}, MAE={test_metrics['MAE']:.6f}, RMSE={test_metrics['RMSE']:.6f}, R2={test_metrics['R2']:.4f}")

        pd.DataFrame({'True_Capacity': val_labels, 'Predicted_Capacity': val_preds}).to_csv(
            os.path.join(config.save_path, 'validation_predictions.csv'), index=False)
        pd.DataFrame({'True_Capacity': test_labels, 'Predicted_Capacity': test_preds}).to_csv(
            os.path.join(config.save_path, 'test_predictions.csv'), index=False)
        print(f"\n验证集和测试集的预测值已保存。")

        plot_results(val_labels, val_preds, 'Validation Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, 'validation_plot.png'))
        plot_results(test_labels, test_preds, 'Test Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, 'test_plot.png'))
        print(f"结果对比图已保存。")


if __name__ == '__main__':
    main()