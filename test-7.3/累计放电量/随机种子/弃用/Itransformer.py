import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import warnings
import shutil


# --- 1. 配置参数 (已修改为iTransformer) ---
class Config:
    def __init__(self):
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/iTransformer-seq10/1'  # 修改了保存路径

        # --- 数据集划分 ---
        self.train_batteries = [1, 2, 3, 4]
        self.val_batteries = [5]
        self.test_batteries = [6]

        self.features_from_C = ['恒压充电时间(s)', '3.3~3.6V充电时间(s)']
        self.sequence_feature_dim = 7  # 原始序列特征维度
        self.scalar_feature_dim = len(self.features_from_C)  # 标量特征维度

        # --- 模型超参数 (为 iTransformer 调整) ---
        self.seq_len = 100  # 输入序列长度
        self.pred_len = 1  # 预测长度，对于回归任务设为1
        self.c_in = self.sequence_feature_dim + self.scalar_feature_dim  # 总输入特征维度
        self.d_model = 256  # 隐藏层维度
        self.n_heads = 8  # 多头注意力头数
        self.e_layers = 3  # Encoder层数
        self.d_ff = 1024  # FFN中间层维度
        self.dropout = 0.2
        self.activation = 'gelu'  # 激活函数
        self.output_attention = False  # 是否输出注意力权重
        self.use_norm = True  # 是否在模型中使用Instance Normalization

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.0005
        self.patience = 30
        self.weight_decay = 0.0001
        self.num_runs = 5
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
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 模型定义 (iTransformer) ---
# 注意: 由于未提供 'layers' 目录中的代码，以下是根据 iTransformer 论文和标准 Transformer 实现的必要模块。
# 如果您有原始的 'layers' 文件，请替换掉这部分。

class DataEmbedding_inverted(nn.Module):
    # FIX: Changed 'c_in' to 'seq_len' in the constructor arguments for clarity and correctness
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # FIX: The Linear layer now correctly maps the sequence length (features of an inverted token)
        # to the model's hidden dimension (d_model).
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [Batch, Seq_len, D_in]
        # Permute to treat features as tokens: [Batch, D_in, Seq_len]
        x = x.permute(0, 2, 1)
        # Apply the linear layer to the last dimension (Seq_len)
        x = self.value_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def forward(self, x, attn_mask=None):
        B, N, _ = x.shape  # Batch, Num_Variates, d_model
        q = self.query(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out(context)

        if self.output_attention:
            return out, attn_weights
        return out, None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.attention = attention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_layer = AttentionLayer(FullAttention, d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, N, E]
        new_x, attn = self.attention_layer(x, attn_mask=attn_mask)
        x = new_x
        y = x

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Model(nn.Module):
    """
    iTransformer adapted for regression on mixed sequence/scalar data.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.use_norm = configs.use_norm

        # Embedding
        # FIX: Pass configs.seq_len to the embedding layer instead of configs.c_in
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head for Regression
        self.prediction_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configs.c_in * configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

    def forward(self, x_seq, x_scalar):
        # x_seq: [B, L, D_seq], x_scalar: [B, D_scalar]

        # 1. Combine sequence and scalar features
        x_scalar_rep = x_scalar.unsqueeze(1).repeat(1, self.configs.seq_len, 1)
        x_enc = torch.cat([x_seq, x_scalar_rep], dim=-1)  # [B, L, D_seq + D_scalar]

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # 2. Embedding
        enc_out = self.enc_embedding(x_enc, None)

        # 3. Encoder
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # 4. Prediction Head
        prediction = self.prediction_head(enc_out)

        return prediction

# --- 4. 数据集定义 (与原版相同) ---
class BatterySequenceDataset(Dataset):
    def __init__(self, df, sequence_col, scalar_cols, target_col, sequence_length):
        self.sequence_length = sequence_length
        self.sequences = np.array(df[sequence_col].tolist(), dtype=np.float32)
        self.scalars = df[scalar_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.battery_ids = df['battery_id'].values

        self.valid_indices = []
        unique_batteries = df['battery_id'].unique()
        for batt_id in unique_batteries:
            batt_indices = np.where(self.battery_ids == batt_id)[0]
            if len(batt_indices) < self.sequence_length:
                continue
            for i in range(len(batt_indices) - self.sequence_length + 1):
                self.valid_indices.append(batt_indices[i])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length

        x_seq_window = self.sequences[start_idx:end_idx]
        x_scalar_last_step = self.scalars[end_idx - 1]
        y_target = self.targets[end_idx - 1]

        return (torch.from_numpy(x_seq_window),
                torch.from_numpy(x_scalar_last_step),
                torch.tensor(y_target, dtype=torch.float32))


# --- 5. 数据加载和预处理 (与原版相同) ---
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
            sequence_df = df_a[['循环号'] + feature_cols]
            sequence_df['voltage_sequence'] = sequence_df[feature_cols].values.tolist()
            sequence_df = sequence_df[['循环号', 'voltage_sequence']]

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
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_target = StandardScaler()

    all_train_sequences = np.vstack(train_df[sequence_col].values)
    scaler_seq.fit(all_train_sequences)
    scaler_scalar.fit(train_df[scalar_feature_cols])
    scaler_target.fit(train_df[[target_col]])

    full_df[sequence_col] = full_df[sequence_col].apply(lambda x: scaler_seq.transform([x])[0])
    full_df.loc[:, scalar_feature_cols] = scaler_scalar.transform(full_df[scalar_feature_cols])
    full_df.loc[:, [target_col]] = scaler_target.transform(full_df[[target_col]])

    train_df_scaled = full_df[full_df['battery_id'].isin(config.train_batteries)]
    val_df_scaled = full_df[full_df['battery_id'].isin(config.val_batteries)]
    test_df_scaled = full_df[full_df['battery_id'].isin(config.test_batteries)]

    train_dataset = BatterySequenceDataset(train_df_scaled, sequence_col, scalar_feature_cols, target_col,
                                           config.seq_len)
    val_dataset = BatterySequenceDataset(val_df_scaled, sequence_col, scalar_feature_cols, target_col,
                                         config.seq_len)
    test_dataset = BatterySequenceDataset(test_df_scaled, sequence_col, scalar_feature_cols, target_col,
                                          config.seq_len)

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar, 'target': scaler_target}
    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练函数 (与原版相同) ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_y in dataloader:
        batch_seq, batch_scalar, batch_y = batch_seq.to(device), batch_scalar.to(device), batch_y.to(device).unsqueeze(
            -1)

        optimizer.zero_grad()
        outputs = model(batch_seq, batch_scalar)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 7. 验证/测试函数 (与原版相同) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_seq, batch_scalar, batch_y in dataloader:
            batch_seq, batch_scalar, batch_y = batch_seq.to(device), batch_scalar.to(device), batch_y.to(
                device).unsqueeze(-1)
            outputs = model(batch_seq, batch_scalar)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    metrics = {
        'MSE': mean_squared_error(labels, predictions),
        'MAE': mean_absolute_error(labels, predictions),
        'R2': r2_score(labels, predictions)
    }
    return avg_loss, metrics, predictions, labels


# --- 8. 可视化和工具函数 (与原版相同) ---
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
    plt.axis('equal');
    plt.xlim(min_val, max_val);
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 9. 主执行函数 (已修改以使用新模型) ---
def main():
    warnings.filterwarnings('ignore');
    matplotlib.use('Agg');
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"总保存路径: {config.save_path}, 设备: {config.device}")

    best_run_mae = float('inf')
    best_run_dir = None
    for run_number in range(1, config.num_runs + 1):
        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)
        current_seed = random.randint(0, 99999);
        set_seed(current_seed)
        print(f"\n{'=' * 30}\n 开始第 {run_number}/{config.num_runs} 次实验 | 种子: {current_seed} \n{'=' * 30}")

        train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        print(
            f"数据加载完成。训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}, 测试样本数: {len(test_dataset)}")

        # --- 实例化新模型 ---
        model = Model(config).to(config.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        best_val_loss_this_run = float('inf');
        epochs_no_improve = 0
        for epoch in range(config.epochs):
            # 移除了 GradScaler, 如有需要可以加回
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, config.device)
            print(
                f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")
            if val_loss < best_val_loss_this_run:
                best_val_loss_this_run = val_loss
                torch.save(model.state_dict(), os.path.join(run_save_path, 'best_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"验证损失连续 {config.patience} 轮未改善，提前停止。")
                    break

        print("\n加载本轮最佳模型进行评估...")
        model.load_state_dict(torch.load(os.path.join(run_save_path, 'best_model.pth')))
        _, test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, config.device)

        scaler_target = scalers['target']
        test_preds_orig = scaler_target.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        test_labels_orig = scaler_target.inverse_transform(test_labels.reshape(-1, 1)).flatten()
        test_preds_orig = np.clip(test_preds_orig, a_min=0, a_max=None)

        final_mae = mean_absolute_error(test_labels_orig, test_preds_orig)
        final_r2 = r2_score(test_labels_orig, test_preds_orig)
        print(f"测试集(原始尺度) MAE: {final_mae:.4f}, R2: {final_r2:.4f}")

        if final_mae < best_run_mae:
            best_run_mae = final_mae
            best_run_dir = run_save_path

        plot_results(test_labels_orig, test_preds_orig, f'Run {run_number} Test Set Predictions',
                     os.path.join(run_save_path, 'test_plot_overall.png'))
        plot_diagonal_results(test_labels_orig, test_preds_orig, f'Run {run_number} Test Set Diagonal',
                              os.path.join(run_save_path, 'test_diagonal_overall.png'))

    if best_run_dir:
        print(f"\n表现最佳的实验保存在: {best_run_dir} (MAE: {best_run_mae:.4f})")
        print(f"正在将最佳结果复制到主目录 {config.save_path} ...")
        for filename in os.listdir(best_run_dir):
            shutil.copy2(os.path.join(best_run_dir, filename), config.save_path)
        print("复制完成。")


if __name__ == '__main__':
    main()