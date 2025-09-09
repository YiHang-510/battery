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
from torch.cuda.amp import autocast, GradScaler
import math  # Added for Positional Encoding


# --- 1. 配置参数 (已修改 for Transformer) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (No change) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/data/transformer_result-forcyclenum/all'  # Changed save path for new model

        # --- 数据集划分 (No change) ---
        self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 11, 15, 17, 18, 19, 21, 22, 23, 24]
        self.val_batteries = [5, 10, 13, 19]
        self.test_batteries = [6, 12, 14, 20]

        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
            '弛豫末端电压'
        ]
        self.sequence_feature_dim = 6

        # --- 模型超参数 (MODIFIED for Transformer) ---
        self.sequence_length = 1
        self.scalar_feature_dim = len(self.features_from_C)  # Will be updated in data loading
        self.d_model = 256  # Hidden dimension for the Transformer
        self.d_ff = 1024  # Feed-forward layer dimension in Transformer
        self.nhead = 8  # Number of attention heads in the Transformer
        self.num_layers = 3  # Number of layers in the Transformer encoder
        self.cycle_len = 2000  # Maximum cycle number for cycle embedding
        self.dropout = 0.2
        self.weight_decay = 0.0001

        # --- 训练参数 (No change) ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.002
        self.patience = 15
        self.seed = 2025
        self.mode = 'both'

        # --- 设备设置 (No change) ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")


# --- 2. 固定随机种子 (No change) ---
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


# --- 3. NEW: PositionalEncoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- 4. NEW: Transformer Model Definition ---
class TransformerForSOH(nn.Module):
    def __init__(self, configs):
        super(TransformerForSOH, self).__init__()
        self.configs = configs

        # 1. Input Encoders
        self.sequence_encoder = nn.Linear(configs.sequence_length * configs.sequence_feature_dim, configs.d_model)
        self.scalar_encoder = nn.Linear(configs.scalar_feature_dim, configs.d_model)

        # 2. Positional and Cycle Embeddings
        self.pos_encoder = PositionalEncoding(configs.d_model, configs.dropout,
                                              max_len=10)  # max_len is small as our sequence is fixed at length 2
        self.cycle_embedding = nn.Embedding(configs.cycle_len, configs.d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.nhead,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            batch_first=True  # Crucial for (batch, seq, feature) input shape
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=configs.num_layers
        )

        # 4. Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        # --- 1. Encode and Combine Inputs ---
        x_seq_flat = x_seq.view(x_seq.size(0), -1)
        seq_embedding = self.sequence_encoder(x_seq_flat)
        scalar_embedding = self.scalar_encoder(x_scalar)

        # Combine into a sequence of length 2: [scalar_token, sequence_token]
        src = torch.stack([scalar_embedding, seq_embedding], dim=1)  # Shape: (batch, 2, d_model)

        # --- 2. Add Positional and Cycle Information ---
        cycle_emb = self.cycle_embedding(cycle_number)
        src = src + cycle_emb.unsqueeze(1)  # Add cycle info to both tokens
        src = self.pos_encoder(src)  # Add relative position info

        # --- 3. Transformer Processing ---
        transformer_output = self.transformer_encoder(src)  # Shape: (batch, 2, d_model)

        # --- 4. Prediction ---
        # Use the output of the first token (scalar token) for prediction
        prediction_input = transformer_output[:, 0, :]  # Shape: (batch, d_model)
        prediction = self.prediction_head(prediction_input)

        return prediction


# --- 5. 数据集定义 (No change) ---
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


# --- 6. 数据加载和预处理 (No change) ---
def load_and_preprocess_data(config):
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries
    for battery_id in sorted(list(set(all_ids))):
        try:
            print(f"正在处理电池 {battery_id}...")
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
        except FileNotFoundError as e:
            print(f"警告: 电池 {battery_id} 的文件未找到，已跳过。错误: {e}")
            continue
        except Exception as e:
            print(f"处理电池 {battery_id} 时出错: {e}")
            continue
    if not all_battery_data:
        raise ValueError("未能成功加载任何电池数据。")
    full_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '循环号'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C
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
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col)
    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar}
    return train_dataset, val_dataset, test_dataset, scalers


# --- 7. 训练函数 (No change) ---
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


# --- 8. 验证/测试函数 (No change) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_cycle_indices = [], [], []
    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
            batch_seq, batch_scalar, batch_cycle_idx = batch_seq.to(device), batch_scalar.to(
                device), batch_cycle_idx.to(device)
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


# --- 9. 可视化和工具函数 (No change) ---
def get_exp_tag(config):
    # ... (code is identical to original)
    train_ids = '-'.join([str(i) for i in config.train_batteries])
    val_ids = '-'.join([str(i) for i in config.val_batteries])
    test_ids = '-'.join([str(i) for i in config.test_batteries])
    tag = (
        f"train{train_ids}_val{val_ids}_test{test_ids}_"
        f"ep{config.epochs}_bs{config.batch_size}_lr{config.learning_rate}_dp{config.dropout}"
        f"dm{config.d_model}_nh{config.nhead}_nl{config.num_layers}"  # Added transformer params to tag
    )
    return tag


def plot_results(labels, preds, title, save_path):
    # ... (code is identical to original)
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predictions', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Cycle Number', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


def plot_diagonal_results(labels, preds, title, save_path):
    # ... (code is identical to original)
    plt.figure(figsize=(8, 8))
    min_val = min(np.min(labels), np.min(preds)) * 0.98
    max_val = max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('True Cycle Number', fontsize=12)
    plt.ylabel('Predicted Cycle Number', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 10. 主执行函数 (MODIFIED to use Transformer) ---
def main():
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    config = Config()
    set_seed(config.seed)

    exp_tag = get_exp_tag(config)
    config.save_path = os.path.join(config.save_path, exp_tag)
    os.makedirs(config.save_path, exist_ok=True)
    print(f"本次实验结果将保存到: {config.save_path}")
    print(f"使用设备: {config.device}")

    try:
        train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载失败: {e}")
        return
    joblib.dump(scalers, os.path.join(config.save_path, 'scalers.pkl'))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(
        f"数据加载完成。训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

    # --- MODIFICATION: Instantiate the new Transformer model ---
    model = TransformerForSOH(config).to(config.device)
    # --- End Modification ---

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
            val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, config.device)
            print(
                f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")

            log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                         **{'val_' + k: v for k, v in val_metrics.items()}}
            metrics_log.append(log_entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.save_path, f'best_model_{exp_tag}.pth'))
                print(f"  - 验证损失降低，保存模型到 {os.path.join(config.save_path, f'best_model_{exp_tag}.pth')}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break
        print("\n训练完成。")
        pd.DataFrame(metrics_log).to_csv(os.path.join(config.save_path, f'training_metrics_log_{exp_tag}.csv'),
                                         index=False)

    if config.mode in ['both', 'validate']:
        print('\n加载最佳模型进行最终评估...')
        model_path = os.path.join(config.save_path, f'best_model_{exp_tag}.pth')
        if not os.path.exists(model_path):
            print(f"错误: 找不到已训练的模型 '{model_path}'。请先在 'train' 或 'both' 模式下运行。")
            return

        model.load_state_dict(torch.load(model_path, map_location=config.device))

        # The evaluate function and subsequent code do not need changes
        _, val_metrics, val_preds, val_labels, val_cycle_nums = evaluate(model, val_loader, criterion, config.device)
        _, test_metrics, test_preds, test_labels, test_cycle_nums = evaluate(model, test_loader, criterion,
                                                                             config.device)

        print("\n--- 评估结果 ---")
        final_metrics = pd.DataFrame([{'set': 'validation', **val_metrics}, {'set': 'test', **test_metrics}])
        final_metrics.to_csv(os.path.join(config.save_path, 'final_evaluation_metrics.csv'), index=False)
        print(
            f"最终验证集指标: MSE={val_metrics['MSE']:.6f}, MAE={val_metrics['MAE']:.6f}, RMSE={val_metrics['RMSE']:.6f}, R2={val_metrics['R2']:.4f}")
        print(
            f"最终测试集指标: MSE={test_metrics['MSE']:.6f}, MAE={test_metrics['MAE']:.6f}, RMSE={test_metrics['RMSE']:.6f}, R2={test_metrics['R2']:.4f}")

        val_results_df = pd.DataFrame(
            {'Original_Cycle_Index': val_cycle_nums, 'True_Cycle': val_labels, 'Predicted_Cycle': val_preds})
        test_results_df = pd.DataFrame(
            {'Original_Cycle_Index': test_cycle_nums, 'True_Cycle': test_labels, 'Predicted_Cycle': test_preds})
        val_results_df.to_csv(os.path.join(config.save_path, f'validation_predictions_{exp_tag}.csv'), index=False)
        test_results_df.to_csv(os.path.join(config.save_path, f'test_predictions_{exp_tag}.csv'), index=False)
        print(f"\n验证集和测试集的预测值已保存。")

        plot_results(val_labels, val_preds, 'Validation Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, f'validation_plot_{exp_tag}.png'))
        plot_results(test_labels, test_preds, 'Test Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, f'test_plot_{exp_tag}.png'))
        plot_diagonal_results(val_labels, val_preds, 'Validation Set: Diagonal Plot',
                              os.path.join(config.save_path, f'validation_diagonal_plot_{exp_tag}.png'))
        plot_diagonal_results(test_labels, test_preds, 'Test Set: Diagonal Plot',
                              os.path.join(config.save_path, f'test_diagonal_plot_{exp_tag}.png'))
        print(f"结果图和对角图已保存。")


if __name__ == '__main__':
    main()