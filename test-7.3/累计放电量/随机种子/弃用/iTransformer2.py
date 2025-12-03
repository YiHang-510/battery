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
from torch.cuda.amp import autocast, GradScaler
import shutil


# --- 1. 配置参数 (新增 Transformer 超参) ---
class Config:
    def __init__(self):
        # 路径
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/iTransformer-seq10/6'

        # 数据集划分（示例）
        self.train_batteries = [1, 2, 3, 4]
        self.val_batteries = [5]
        self.test_batteries = [6]

        # 特征维度
        self.features_from_C = ['恒压充电时间(s)', '3.3~3.6V充电时间(s)']
        self.sequence_feature_dim = 7  # 序列每步 7 维（弛豫电压7点）
        self.scalar_feature_dim = len(self.features_from_C)  # 2 维标量

        # 序列窗口长度
        self.sequence_length = 10       # L

        # --- 模型超参数：倒置Transformer（变量为token） ---
        self.d_model = 256              # token 表示维度
        self.n_heads = 4
        self.e_layers = 3
        self.d_ff = 1024
        self.dropout = 0.2
        self.activation = 'gelu'

        # 训练参数
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 5e-4
        self.weight_decay = 1e-4
        self.patience = 30
        self.num_runs = 3

        # 设备
        self.use_gpu = True
        self.device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')


# --- 2. 固定随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 倒置Transformer回归模型 ---
class ITransformerRegressor(nn.Module):
    """
    将时间维(L)作为每个变量token的局部特征，先对 BxNxL 做线性投影到 d_model，
    再以 N(变量数=7+标量2) 为序列长度做 TransformerEncoder，最后池化得到样本表示 -> 回归到标量 y。
    输入：
        x_seq:    [B, L, 7]
        x_scalar: [B, 2]  (在模型内部沿 L 复制并与 x_seq 拼接，得到 N=9 个 token)
    输出：
        y_hat:    [B, 1]
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        L = cfg.sequence_length
        self.N = cfg.sequence_feature_dim + cfg.scalar_feature_dim  # 7 + 2 = 9

        # 将每个 token 的长度 L 投影到 d_model（等价 DataEmbedding_inverted 的最小实现）
        self.token_proj = nn.Sequential(
            nn.LayerNorm(L),
            nn.Linear(L, cfg.d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=False,  # PyTorch Transformer 期望 [S, B, E]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.e_layers)
        self.enc_out_norm = nn.LayerNorm(cfg.d_model)

        # 回归头
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, 1)
        )

    def forward(self, x_seq: torch.Tensor, x_scalar: torch.Tensor):
        # x_seq: [B, L, 7]; x_scalar: [B, 2]
        B, L, Dseq = x_seq.shape
        assert L == self.cfg.sequence_length, f"输入序列长度 L={L} 与配置 {self.cfg.sequence_length} 不一致"
        assert Dseq == self.cfg.sequence_feature_dim, f"序列特征维 {Dseq} 与配置 {self.cfg.sequence_feature_dim} 不一致"

        # 将标量特征复制到时间维，拼接为额外 token
        x_scalar_t = x_scalar.unsqueeze(1).expand(B, L, x_scalar.shape[-1])  # [B, L, 2]
        x = torch.cat([x_seq, x_scalar_t], dim=-1)                            # [B, L, 9]

        # 倒置：变量为 token -> [B, N, L]
        x = x.permute(0, 2, 1).contiguous()                                   # [B, N, L]
        # 对每个 token 的时间片进行线性嵌入 -> [B, N, d_model]
        x = self.token_proj(x)

        # 以 N 作为序列长度做自注意力：Transformer 需要 [S, B, E]
        x = x.permute(1, 0, 2).contiguous()                                   # [N, B, d_model]
        x = self.encoder(x)                                                   # [N, B, d_model]
        x = x.permute(1, 0, 2).contiguous()                                   # [B, N, d_model]
        x = self.enc_out_norm(x)

        # token 池化（均值，也可改为可学习 CLS）
        x = x.mean(dim=1)                                                     # [B, d_model]

        # 回归
        y = self.head(x)                                                      # [B, 1]
        return y


# --- 4. 数据集定义（窗口化，不变） ---
class BatterySequenceDataset(Dataset):
    def __init__(self, df, sequence_col, scalar_cols, target_col, sequence_length):
        self.sequence_length = sequence_length
        self.sequences = np.array(df[sequence_col].tolist(), dtype=np.float32)
        self.scalars = df[scalar_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        self.battery_ids = df['battery_id'].values

        self.valid_indices = []
        for batt_id in df['battery_id'].unique():
            idxs = np.where(self.battery_ids == batt_id)[0]
            if len(idxs) < self.sequence_length:
                continue
            for i in range(len(idxs) - self.sequence_length + 1):
                self.valid_indices.append(idxs[i])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.sequence_length
        x_seq_window = self.sequences[start:end]               # [L, 7]
        x_scalar_last = self.scalars[end - 1]                  # [2]
        y_target = self.targets[end - 1]                       # []
        return (
            torch.from_numpy(x_seq_window),
            torch.from_numpy(x_scalar_last),
            torch.tensor(y_target, dtype=torch.float32)
        )


# --- 5. 数据加载和预处理（与原脚本一致） ---
def load_and_preprocess_data(config: Config):
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

    full_df = pd.concat(all_battery_data, ignore_index=True).sort_values(by=['battery_id', '循环号']).reset_index(drop=True)

    target_col = '累计放电容量(Ah)'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # 标准化
    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_target = StandardScaler()

    all_train_sequences = np.vstack(train_df[sequence_col].values)  # [num_train, 7]
    scaler_seq.fit(all_train_sequences)
    scaler_scalar.fit(train_df[scalar_feature_cols])
    scaler_target.fit(train_df[[target_col]])

    full_df[sequence_col] = full_df[sequence_col].apply(lambda x: scaler_seq.transform([x])[0])
    full_df.loc[:, scalar_feature_cols] = scaler_scalar.transform(full_df[scalar_feature_cols])
    full_df.loc[:, [target_col]] = scaler_target.transform(full_df[[target_col]])

    train_df_scaled = full_df[full_df['battery_id'].isin(config.train_batteries)]
    val_df_scaled = full_df[full_df['battery_id'].isin(config.val_batteries)]
    test_df_scaled = full_df[full_df['battery_id'].isin(config.test_batteries)]

    train_dataset = BatterySequenceDataset(train_df_scaled, sequence_col, scalar_feature_cols, target_col, config.sequence_length)
    val_dataset = BatterySequenceDataset(val_df_scaled, sequence_col, scalar_feature_cols, target_col, config.sequence_length)
    test_dataset = BatterySequenceDataset(test_df_scaled, sequence_col, scalar_feature_cols, target_col, config.sequence_length)

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar, 'target': scaler_target}
    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练 / 验证 ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_y in dataloader:
        batch_seq = batch_seq.to(device)                # [B, L, 7]
        batch_scalar = batch_scalar.to(device)          # [B, 2]
        batch_y = batch_y.to(device).unsqueeze(-1)      # [B, 1]
        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                outputs = model(batch_seq, batch_scalar)
                loss = criterion(outputs, batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs = model(batch_seq, batch_scalar)
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
        for batch_seq, batch_scalar, batch_y in dataloader:
            batch_seq = batch_seq.to(device)
            batch_scalar = batch_scalar.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)
            outputs = model(batch_seq, batch_scalar)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    metrics = {
        'MSE': mean_squared_error(labels, preds),
        'MAE': mean_absolute_error(labels, preds),
        'R2': r2_score(labels, preds),
    }
    return avg_loss, metrics, preds, labels


# --- 7. 可视化 ---
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
    plt.axis('equal'); plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 8. 主函数 ---
def main():
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    cfg = Config()
    os.makedirs(cfg.save_path, exist_ok=True)
    print(f"保存路径: {cfg.save_path}, 设备: {cfg.device}")

    best_run_mae, best_run_dir = float('inf'), None

    for run in range(1, cfg.num_runs + 1):
        run_dir = os.path.join(cfg.save_path, f'run_{run}')
        os.makedirs(run_dir, exist_ok=True)
        seed = random.randint(0, 99999)
        set_seed(seed)
        print(f"\n{'='*28}\n 第 {run}/{cfg.num_runs} 次实验 | 种子: {seed} \n{'='*28}")

        train_ds, val_ds, test_ds, scalers = load_and_preprocess_data(cfg)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}, 测试样本: {len(test_ds)}")

        model = ITransformerRegressor(cfg).to(cfg.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scaler = GradScaler(enabled=(cfg.use_gpu and cfg.device.type == 'cuda'))

        best_val, no_improve = float('inf'), 0
        for epoch in range(cfg.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg.device, scaler)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, cfg.device)
            print(f"Epoch {epoch+1}/{cfg.epochs} | 训练: {train_loss:.6f} | 验证: {val_loss:.6f} | R2: {val_metrics['R2']:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    print(f"验证集 {cfg.patience} 轮未提升，提前停止。")
                    break

        print("\n加载本轮最佳模型评估测试集...")
        model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
        _, test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, cfg.device)

        # 反归一化
        scaler_target = scalers['target']
        test_preds_orig = scaler_target.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        test_labels_orig = scaler_target.inverse_transform(test_labels.reshape(-1, 1)).flatten()
        test_preds_orig = np.clip(test_preds_orig, a_min=0, a_max=None)

        final_mae = mean_absolute_error(test_labels_orig, test_preds_orig)
        final_r2 = r2_score(test_labels_orig, test_preds_orig)
        print(f"测试集(原始尺度) MAE: {final_mae:.4f}, R2: {final_r2:.4f}")

        if final_mae < best_run_mae:
            best_run_mae = final_mae
            best_run_dir = run_dir

        plot_results(test_labels_orig, test_preds_orig, f'Run {run} Test Predictions', os.path.join(run_dir, 'test_plot_overall.png'))
        plot_diagonal_results(test_labels_orig, test_preds_orig, f'Run {run} Test Diagonal', os.path.join(run_dir, 'test_diagonal_overall.png'))

    if best_run_dir:
        print(f"\n最佳实验目录: {best_run_dir} (MAE: {best_run_mae:.4f})")
        print(f"复制最佳结果到主目录 {cfg.save_path}...")
        for fname in os.listdir(best_run_dir):
            shutil.copy2(os.path.join(best_run_dir, fname), cfg.save_path)
        print("复制完成。")


if __name__ == '__main__':
    main()
