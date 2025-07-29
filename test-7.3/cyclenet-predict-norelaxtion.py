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

warnings.filterwarnings('ignore')


# --- 1. 配置参数 (MODIFIED: Re-introducing sequence params for the new approach) ---
class Config:
    def __init__(self):
        # 数据路径仍然是文件C
        self.data_path = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/data/cyclenet_on_tabular_result'

        self.train_batteries = [1, 2, 3, 4]
        self.val_batteries = [5]
        self.test_batteries = [6]

        # --- 模型超参数 (MODIFIED: CycleNet params on top of tabular data) ---
        # seq_len 现在代表我们用多少个历史循环数据来构建一个序列
        self.seq_len = 10
        # enc_in 是文件C中的特征数量 (将自动更新)
        self.enc_in = 10
        self.cycle_len = 2000  # 最大循环次数
        self.d_model = 128  # 隐藏层维度
        self.model_type = 'mlp'  # CycleNet内部的处理器类型
        self.use_revin = True  # 是否使用可逆实例归一化
        self.weight_decay = 0.0001

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.005
        self.patience = 40
        self.seed = 2025
        self.mode = 'both'

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")


# --- 2. 固定随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 模型定义 (MODIFIED: CycleNet with a prediction head) ---
class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle_len
        self.use_revin = configs.use_revin

        # Part 1: CycleNet Core
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # Part 2: Prediction Head
        # CycleNet本身不改变数据维度，所以我们需要一个预测头将序列输出转换为单个值
        # 输入维度是 seq_len * enc_in (将处理后的序列展平)
        self.prediction_head = nn.Sequential(
            nn.Flatten(),  # 将 [batch, seq_len, enc_in] 展平为 [batch, seq_len * enc_in]
            nn.Linear(configs.seq_len * configs.enc_in, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # 输出最终的SOH值
        )

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)

        # --- CycleNet核心操作 ---
        # RevIN: 实例归一化
        if self.use_revin:
            # 在序列长度维度上进行归一化
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 减去周期性分量
        # 注意：这里的cycle_index应该是序列的起始点的循环号
        processed_x = x - self.cycleQueue(cycle_index, self.seq_len)

        # --- 预测 ---
        # 将处理后的序列送入预测头
        prediction = self.prediction_head(processed_x)

        return prediction


# --- 4. 数据集定义 (MODIFIED: Creates sequences from tabular data) ---
class SlidingWindowDataset(Dataset):
    def __init__(self, features, targets, cycle_numbers, window_size):
        self.features = features
        self.targets = targets
        self.cycle_numbers = cycle_numbers
        self.window_size = window_size

    def __len__(self):
        # 总长度减去窗口大小，因为前几个点没有足够的历史数据
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        # 真实的数据索引
        real_idx = idx + self.window_size

        # 提取窗口大小的序列作为输入特征
        start_idx = real_idx - self.window_size
        end_idx = real_idx
        x_seq = self.features[start_idx:end_idx]

        # 序列的起始循环号，用于CycleNet
        cycle_idx_start = self.cycle_numbers[start_idx]

        # 目标值是窗口末端的值
        y = self.targets[real_idx]

        return torch.tensor(x_seq, dtype=torch.float32), \
            torch.tensor(cycle_idx_start, dtype=torch.long), \
            torch.tensor(y, dtype=torch.float32)


# --- 5. 数据加载和预处理 (MODIFIED: Heavily adapted for sequence creation) ---
def load_and_preprocess_data(config):
    full_df_list = []
    for battery_id in sorted(list(set(config.train_batteries + config.val_batteries + config.test_batteries))):
        try:
            path_c = os.path.join(config.data_path, f'battery{battery_id}_SOH健康特征提取结果.csv')
            df = pd.read_csv(path_c, sep=',', encoding='gbk')
            df['battery_id'] = battery_id
            # 必须按循环号排序，以保证序列的连续性
            df = df.sort_values(by='循环号').reset_index(drop=True)
            full_df_list.append(df)
        except FileNotFoundError as e:
            print(f"警告: 电池 {battery_id} 的文件未找到，已跳过。错误: {e}")

    if not full_df_list:
        raise ValueError("未能加载任何电池数据。")

    # 定义特征和目标
    target_col = '最大容量(Ah)'
    # 找到所有电池数据共有的特征列
    feature_cols = [col for col in full_df_list[0].columns if col not in [target_col, 'battery_id']]
    config.enc_in = len(feature_cols)
    print(f"检测到 {config.enc_in} 个输入特征: {feature_cols}")

    # --- 特征缩放 ---
    # 合并所有训练电池的数据来拟合scaler
    train_dfs_for_scaling = [df for df in full_df_list if df['battery_id'].iloc[0] in config.train_batteries]
    combined_train_df = pd.concat(train_dfs_for_scaling, ignore_index=True)
    scaler = StandardScaler().fit(combined_train_df[feature_cols])

    # 创建数据集
    all_datasets = {}
    for battery_df in full_df_list:
        battery_id = battery_df['battery_id'].iloc[0]

        # 复制以避免SettingWithCopyWarning
        df_copy = battery_df.copy()

        # 应用缩放
        df_copy.loc[:, feature_cols] = scaler.transform(df_copy[feature_cols])

        # 提取为Numpy数组
        features = df_copy[feature_cols].values
        targets = df_copy[target_col].values
        cycle_numbers = df_copy['循环号'].values

        # 创建滑动窗口数据集
        dataset = SlidingWindowDataset(features, targets, cycle_numbers, config.seq_len)
        all_datasets[battery_id] = dataset

    # 将属于同一集合的电池数据集合并
    train_datasets = [all_datasets[bid] for bid in config.train_batteries if bid in all_datasets]
    val_datasets = [all_datasets[bid] for bid in config.val_batteries if bid in all_datasets]
    test_datasets = [all_datasets[bid] for bid in config.test_batteries if bid in all_datasets]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    return train_dataset, val_dataset, test_dataset, scaler


# --- 6. 训练和评估函数 (MODIFIED: to handle new data format) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    # 数据解包: x_seq, cycle_idx, y
    for batch_seq, batch_cycle_idx, batch_y in dataloader:
        batch_seq = batch_seq.to(device)
        batch_cycle_idx = batch_cycle_idx.to(device)
        batch_y = batch_y.to(device).unsqueeze(-1)

        optimizer.zero_grad()

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


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_seq, batch_cycle_idx, batch_y in dataloader:
            batch_seq = batch_seq.to(device)
            batch_cycle_idx = batch_cycle_idx.to(device)

            outputs = model(batch_seq, batch_cycle_idx)
            loss = criterion(outputs, batch_y.to(device).unsqueeze(-1))

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    metrics = {
        'MSE': mean_squared_error(labels, predictions),
        'MAE': mean_absolute_error(labels, predictions),
        'RMSE': np.sqrt(mean_squared_error(labels, predictions)),
        'R2': r2_score(labels, predictions)
    }
    return avg_loss, metrics, predictions, labels


# --- 8. 可视化和工具函数 (基本不变) ---
def get_exp_tag(config):
    return f"CycleNet_On_Tabular_seq{config.seq_len}_bs{config.batch_size}_lr{config.learning_rate}"


def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='.', linestyle='None')
    plt.plot(preds, label='Predictions', marker='.', linestyle='None', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Max Capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 9. 主执行函数 ---
def main():
    config = Config()
    set_seed(config.seed)

    exp_tag = get_exp_tag(config)
    config.save_path = os.path.join(config.save_path, exp_tag)
    os.makedirs(config.save_path, exist_ok=True)
    print(f"实验结果将保存到: {config.save_path}")
    print(f"使用设备: {config.device}")

    try:
        train_dataset, val_dataset, test_dataset, scaler = load_and_preprocess_data(config)
    except (ValueError, FileNotFoundError) as e:
        print(f"数据加载失败: {e}")
        return
    joblib.dump(scaler, os.path.join(config.save_path, 'scaler.pkl'))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"数据加载完成。训练集样本数: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    model = Model(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

    # ... (训练和评估循环与之前版本类似, 此处省略以保持简洁, 实际代码与上方函数一致) ...
    # 为了完整性，在此处粘贴完整的训练/评估主循环
    metrics_log = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if config.mode in ['train', 'both']:
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
                torch.save(model.state_dict(), os.path.join(config.save_path, 'best_model.pth'))
                print(f"  - 验证损失降低，保存模型。")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break
        print("\n训练完成。")
        pd.DataFrame(metrics_log).to_csv(os.path.join(config.save_path, 'training_log.csv'), index=False)

    if config.mode in ['validate', 'both']:
        print('\n加载最佳模型进行最终评估...')
        model_path = os.path.join(config.save_path, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"错误: 找不到已训练的模型 '{model_path}'。")
            return

        model.load_state_dict(torch.load(model_path, map_location=config.device))

        _, val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, config.device)
        _, test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, config.device)

        print("\n--- 评估结果 ---")
        print(f"最终验证集指标: R2={val_metrics['R2']:.4f}, RMSE={val_metrics['RMSE']:.6f}")
        print(f"最终测试集指标: R2={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.6f}")

        plot_results(val_labels, val_preds, 'Validation Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, 'validation_plot.png'))
        plot_results(test_labels, test_preds, 'Test Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, 'test_plot.png'))
        print("评估图表已保存。")


if __name__ == '__main__':
    main()