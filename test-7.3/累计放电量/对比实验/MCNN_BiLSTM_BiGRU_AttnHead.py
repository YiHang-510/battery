"""
A novel hybrid neural network-based SOH and RUL estimation method for lithium-ion batteries
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib
import warnings
import joblib
from torch.cuda.amp import autocast, GradScaler
import shutil
from typing import Tuple


# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/MCNN_BiRNN_AM/4'  # 建议为新模型创建一个新的保存路径

        self.train_batteries = [1, 2, 3, 6]
        self.val_batteries = [5]
        self.test_batteries = [4]
        #
        # self.train_batteries = [7, 8, 9, 11]
        # self.val_batteries = [10]
        # self.test_batteries = [12]

        # self.train_batteries = [15, 16, 17, 18]
        # self.val_batteries = [13]
        # self.test_batteries = [14]
        #
        # self.train_batteries = [21, 22, 23, 24]
        # self.val_batteries = [19]
        # self.test_batteries = [20]

        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7  # A文件的输入特征维度
        self.sequence_length = 1  # 序列长度
        self.cap_norm = 3.5  # 最大容量的固定归一化系数

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.patience = 10
        self.seed = 2025
        self.mode = 'both'
        self.dropout = 0.2  # 为新模型设置一个dropout率

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

        # --- 时间(循环号)归一化 (min-max to [0,1]) ---
        self.cycle_norm_min = None
        self.cycle_norm_max = None

        # --- 辅助参数 ---
        self.scalar_feature_dim = len(self.features_from_C)


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


# ==============================================================================
# ======================== 3. 基于论文的新模型架构 ============================
# ==============================================================================

# --- 辅助模块: 卷积块 ---
class ConvBlock(nn.Module):
    """一个标准的卷积块，包含两个卷积层。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: float = 0.1):
        super().__init__()
        pad = (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p),  # 增加Dropout以进行正则化
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# --- 辅助模块: 点积注意力机制 ---
class DotAttention(nn.Module):
    """简单的点积注意力机制，返回上下文向量和注意力权重。"""

    def __init__(self, d_hidden: int):
        super().__init__()
        # 使用序列的最后一个隐藏状态作为查询(Query)
        self.q_proj = nn.Linear(d_hidden, d_hidden, bias=False)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # H 是RNN的输出，形状为: [Batch, SequenceLength, HiddenDim]
        last_hidden_state = H[:, -1, :]  # Query: [B, D]
        Q = self.q_proj(last_hidden_state)  # 投影后的Query

        # 计算注意力得分: (H * Q^T) / sqrt(d_k)
        attn_scores = torch.einsum('btd,bd->bt', H, Q) / (H.size(-1) ** 0.5)  # [B, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, T]

        # 计算上下文向量
        context = torch.einsum('bt,btd->bd', attn_weights, H)  # [B, D]
        return context, attn_weights


# --- 主模型: MCNN_BiRNN_AM (根据论文实现) ---
class MCNN_BiRNN_AM(nn.Module):
    def __init__(self,
                 seq_channels: int,  # 序列特征的通道数 (例如: 7)
                 scalar_dim: int,  # 标量特征的数量 (例如: 2)
                 cnn_out_channels: int = 64,
                 rnn_hidden: int = 128,
                 rnn_layers: int = 1,
                 dropout: float = 0.2,
                 mlp_hidden: int = 256):
        super().__init__()

        # 总的输入特征维度 = 序列特征 + 标量特征 + 归一化时间
        self.num_health_features = seq_channels + scalar_dim + 1

        # --- 1. 多通道CNN分支 ---
        # 将所有健康特征作为一个序列输入
        self.cnn_branch = nn.Sequential(
            ConvBlock(self.num_health_features, cnn_out_channels, k=3, p=dropout),
            nn.AdaptiveAvgPool1d(1)  # 对序列长度维度进行全局平均池化
        )

        # --- 2. 双向RNN分支 ---
        # BiLSTM -> BiGRU -> Attention
        self.bilstm = nn.LSTM(input_size=self.num_health_features,
                              hidden_size=rnn_hidden,
                              num_layers=rnn_layers,
                              dropout=0.0 if rnn_layers == 1 else dropout,
                              bidirectional=True,
                              batch_first=True)

        self.bigru = nn.GRU(input_size=2 * rnn_hidden,  # 输入来自BiLSTM
                            hidden_size=rnn_hidden,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)

        # 注意力机制，用于处理BiGRU的输出
        self.attention = DotAttention(d_hidden=2 * rnn_hidden)
        self.dropout_rnn = nn.Dropout(dropout)

        # --- 3. 融合与最终预测头 ---
        # 论文中将两个分支的输出进行拼接
        fusion_dim = cnn_out_channels + (2 * rnn_hidden)  # CNN输出 + RNN上下文向量

        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor):
        # v: [B, L, C_v] -> 电压序列
        # s: [B, C_s]    -> 标量特征
        # t: [B]         -> 归一化的循环索引

        B, L, C_v = v.shape

        # --- 准备序列模型的输入 ---
        # 将标量和时间扩展到与序列相同的长度L
        s_expanded = s.unsqueeze(1).expand(-1, L, -1)  # [B, L, C_s]
        t_expanded = t_norm.unsqueeze(1).unsqueeze(2).expand(-1, L, 1)  # [B, L, 1]

        # 拼接所有特征，形成两个分支的共同输入
        x_rnn = torch.cat([v, s_expanded, t_expanded], dim=2)  # 形状: [B, L, D_total]

        # 对于Conv1d，通道维度应为第2维
        x_cnn = x_rnn.permute(0, 2, 1)  # 形状: [B, D_total, L]

        # --- 分支 1: CNN ---
        h_cnn = self.cnn_branch(x_cnn)  # 形状: [B, C_cnn_out, 1]
        h_cnn = h_cnn.squeeze(-1)  # 形状: [B, C_cnn_out]

        # --- 分支 2: RNN ---
        h_lstm, _ = self.bilstm(x_rnn)
        h_gru, _ = self.bigru(h_lstm)
        h_rnn, _ = self.attention(h_gru)  # 形状: [B, 2 * rnn_hidden]
        h_rnn = self.dropout_rnn(h_rnn)

        # --- 融合与预测 ---
        h_fused = torch.cat([h_cnn, h_rnn], dim=1)

        # 得到最终预测结果
        Q = self.prediction_head(h_fused).squeeze(-1)  # 形状: [B]

        # 论文预测的是SOH，通常在[0,1]之间，因此使用sigmoid
        # 您的目标是容量，但也是归一化后的，所以sigmoid是合理的
        Q = torch.sigmoid(Q)

        # 返回一个空的'aux'字典以匹配您训练函数的签名
        return Q, {}


# ==============================================================================
# ======================== 模型定义结束 ======================================
# ==============================================================================


# --- 4. 数据集定义 (不变) ---
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
        # 注意: 您的原始代码中序列长度为1，所以x_seq的形状是 [1, 7]
        # 这对于RNN来说也是一个长度为1的序列，是兼容的
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)
        return x_seq, x_scalar, cycle_idx, y


# --- 5. 数据加载和预处理 (不变) ---
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

            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.sequence_length]

            final_df = pd.merge(sequence_df, df_c, on='循环号')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)

        except FileNotFoundError as e:
            print(f"警告: 电池 {battery_id} 的文件未找到，已跳过。错误: {e}")
            continue
        except Exception as e:
            print(f"处理电池 {battery_id} 时出错: {e}")
            continue

    if not all_battery_data: raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '最大容量(Ah)'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C

    for col in scalar_feature_cols:
        if col not in full_df.columns:
            raise ValueError(f"您选择的特征 '{col}' 不存在于加载的数据中。")

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    config.cycle_norm_min = float(train_df['循环号'].min())
    config.cycle_norm_max = float(train_df['循环号'].max())
    if config.cycle_norm_max <= config.cycle_norm_min:
        config.cycle_norm_max = config.cycle_norm_min + 1.0

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()

    scaler_seq.fit(np.vstack(train_df[sequence_col].values))
    scaler_scalar.fit(train_df[scalar_feature_cols])

    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])
        df[target_col] = df[target_col].astype(float) / config.cap_norm

    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col)
    scalers = {
        'sequence': scaler_seq, 'scalar': scaler_scalar,
        'cycle_norm': {'min': config.cycle_norm_min, 'max': config.cycle_norm_max}
    }
    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练函数 (已修改以适配新模型) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler, config):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        # 新模型 forward 需要的参数是 v, s, t_norm
        # 您的数据加载器已经提供了这三者需要的所有原始数据
        batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
            batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device)

        # 归一化循环号，得到 t_norm
        t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
        t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
        t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min)
        t_norm = torch.clamp(t_norm, 0.0, 1.0)

        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                # 新模型的 forward 函数签名是 (v, s, t_norm)
                outputs, aux = model(batch_seq, batch_scalar, t_norm)
                loss = criterion(outputs, batch_y)
                # --- 修改: 移除旧模型特有的正则化项 ---
                # loss_reg = model.cfg.residual_l2 * (aux["c_h"] ** 2).mean()
                # loss = loss + loss_reg
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs, aux = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs, batch_y)
            # --- 修改: 移除旧模型特有的正则化项 ---
            # loss_reg = model.cfg.residual_l2 * (aux["c_h"] ** 2).mean()
            # loss = loss + loss_reg
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 7. 验证/测试函数 (已修改以适配新模型) ---
def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_cycle_indices = [], [], []
    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
            batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
                batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device)

            # 归一化循环号
            t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
            t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
            t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min)
            t_norm = torch.clamp(t_norm, 0.0, 1.0)

            outputs, aux = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs, batch_y)
            # --- 修改: 移除旧模型特有的正则化项 ---
            # loss_reg = model.cfg.residual_l2 * (aux["c_h"] ** 2).mean()
            # loss = loss + loss_reg

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_cycle_indices.append(batch_cycle_idx.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    cycle_indices = np.concatenate(all_cycle_indices).flatten()
    metrics = {'MSE': mean_squared_error(labels, predictions),
               'MAPE': mean_absolute_percentage_error(labels, predictions),
               'MAE': mean_absolute_error(labels, predictions),
               'RMSE': np.sqrt(mean_squared_error(labels, predictions)),
               'R2': r2_score(labels, predictions)}
    return avg_loss, metrics, predictions, labels, cycle_indices


# --- 8. 可视化和工具函数 (不变) ---
def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predictions', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Capacity', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


def plot_diagonal_results(labels, preds, title, save_path):
    plt.figure(figsize=(8, 8))
    min_val, max_val = min(np.min(labels), np.min(preds)) * 0.98, max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('True Capacity')
    plt.ylabel('Predicted Capacity')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()



# --- 9. 主执行函数 (已更新模型初始化) ---
def main():
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')

    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"所有实验的总保存路径: {config.save_path}")
    print(f"使用设备: {config.device}")

    num_runs = 5
    all_runs_metrics, all_runs_PER_BATTERY_metrics = [], []
    best_run_val_loss = float('inf')
    best_run_dir = None
    best_run_number = -1

    for run_number in range(1, num_runs + 1):
        current_seed = random.randint(0, 99999)
        set_seed(current_seed)
        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)
        print(
            f"\n{'=' * 30}\n 开始第 {run_number}/{num_runs} 次实验 | 随机种子: {current_seed} \n 本次实验结果将保存到: {run_save_path}\n{'=' * 30}")

        try:
            train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
        except (FileNotFoundError, ValueError) as e:
            print(f"数据加载失败: {e}")
            continue

        joblib.dump(scalers, os.path.join(run_save_path, 'scalers.pkl'))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True)
        print(f"数据加载完成。训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        # ====================================================================
        # ==================== 修改: 实例化新的 MCNN_BiRNN_AM 模型 ============
        # ====================================================================
        # 移除旧模型 (TM_PIRes) 的初始化代码
        # tm_config = ...
        # model = TM_PIRes(...)

        # 实例化基于论文的新模型
        model = MCNN_BiRNN_AM(
            seq_channels=config.sequence_feature_dim,
            scalar_dim=config.scalar_feature_dim,
            cnn_out_channels=64,  # 超参数, 可以调整
            rnn_hidden=128,  # 超参数, 可以调整
            rnn_layers=1,  # 超参数, 可以调整
            dropout=config.dropout,  # 使用config中定义的dropout率
            mlp_hidden=256  # 超参数, 可以调整
        ).to(config.device)
        # ====================================================================
        # ======================== 模型初始化修改结束 ========================
        # ====================================================================

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

        metrics_log = []
        best_val_loss_this_run = float('inf')
        epochs_no_improve = 0

        if config.mode in ['both', 'train']:
            print("\n开始训练模型...")
            for epoch in range(config.epochs):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler, config)
                val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, config.device, config)

                print(
                    f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")
                log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                             **{'val_' + k: v for k, v in val_metrics.items()}}
                metrics_log.append(log_entry)

                if val_loss < best_val_loss_this_run:
                    best_val_loss_this_run = val_loss
                    torch.save(model.state_dict(), os.path.join(run_save_path, 'best_model.pth'))
                    print(f"  - 验证损失降低，保存模型到 {os.path.join(run_save_path, 'best_model.pth')}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= config.patience:
                        print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                        break
            pd.DataFrame(metrics_log).to_csv(os.path.join(run_save_path, 'training_metrics_log.csv'), index=False)

        if config.mode in ['both', 'validate']:
            print('\n加载本轮最佳模型进行最终评估...')
            model_path = os.path.join(run_save_path, 'best_model.pth')
            if not os.path.exists(model_path):
                print(f"错误: 找不到已训练的模型 '{model_path}'。")
                continue

            model.load_state_dict(torch.load(model_path, map_location=config.device))

            _, _, test_preds, test_labels, test_cycle_nums = evaluate(model, test_loader, criterion, config.device,
                                                                      config)

            # 由于新模型的输出是sigmoid，范围在(0,1)，所以反归一化只需乘以cap_norm
            # 注意：您的原始代码中反归一化部分被注释了，这里也保持一致，直接使用归一化后的值进行评估
            test_preds_orig = test_preds
            test_labels_orig = test_labels
            test_preds_orig = np.clip(test_preds_orig, a_min=0.0, a_max=None)

            print("\n--- 本轮评估结果 (按单电池) ---")
            eval_df = pd.DataFrame(
                {'battery_id': test_dataset.df['battery_id'].values, 'cycle': test_cycle_nums, 'true': test_labels_orig,
                 'pred': test_preds_orig})
            per_battery_metrics_list = []
            for batt_id in config.test_batteries:
                batt_df = eval_df[eval_df['battery_id'] == batt_id]
                if batt_df.empty: continue
                batt_true, batt_pred = batt_df['true'].values, batt_df['pred'].values
                batt_metrics_dict = {'Battery_ID': batt_id, 'MAE': mean_absolute_error(batt_true, batt_pred),
                                     'MAPE': mean_absolute_percentage_error(batt_true, batt_pred),
                                     'MSE': mean_squared_error(batt_true, batt_pred),
                                     'RMSE': np.sqrt(mean_squared_error(batt_true, batt_pred)),
                                     'R2': r2_score(batt_true, batt_pred)}
                per_battery_metrics_list.append(batt_metrics_dict)
                print(
                    f"  - 电池 {batt_id}: MAE={batt_metrics_dict['MAE']:.6f}, RMSE={batt_metrics_dict['RMSE']:.6f}, R2={batt_metrics_dict['R2']:.4f}")
                all_runs_PER_BATTERY_metrics.append({**batt_metrics_dict, 'run': run_number, 'seed': current_seed})
                plot_results(batt_true, batt_pred, f'Battery {batt_id}: True vs Predicted Capacity',
                             os.path.join(run_save_path, f'test_plot_battery_{batt_id}.png'))
                plot_diagonal_results(batt_true, batt_pred, f'Battery {batt_id}: Diagonal Plot',
                                      os.path.join(run_save_path, f'test_diagonal_plot_battery_{batt_id}.png'))
            pd.DataFrame(per_battery_metrics_list).to_csv(os.path.join(run_save_path, 'test_per_battery_metrics.csv'),
                                                          index=False)

            print("\n--- 本轮评估结果 (所有测试电池汇总) ---")
            final_test_metrics = {'MAE': mean_absolute_error(test_labels_orig, test_preds_orig),
                                  'MAPE': mean_absolute_percentage_error(test_labels_orig, test_preds_orig),
                                  'MSE': mean_squared_error(test_labels_orig, test_preds_orig),
                                  'RMSE': np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig)),
                                  'R2': r2_score(test_labels_orig, test_preds_orig)}
            pd.DataFrame([final_test_metrics]).to_csv(os.path.join(run_save_path, 'test_overall_metrics.csv'),
                                                      index=False)
            all_runs_metrics.append({'run': run_number, 'seed': current_seed, **final_test_metrics})
            print(
                f"测试集(汇总): MSE={final_test_metrics['MSE']:.6f}, MAE={final_test_metrics['MAE']:.6f}, RMSE={final_test_metrics['RMSE']:.6f}, R2={final_test_metrics['R2']:.4f}")

            if best_val_loss_this_run < best_run_val_loss:
                best_run_val_loss, best_run_dir, best_run_number = best_val_loss_this_run, run_save_path, run_number
                print(f"*** 新的最佳表现！验证集损失: {best_val_loss_this_run:.6f} ***")

    print(f"\n\n{'=' * 50}\n 所有实验均已完成。\n{'=' * 50}")
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 ---\n", summary_df)
        print(f"\n汇总指标已保存到: {summary_path}")

    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (验证集损失最低: {best_run_val_loss:.6f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")
        for filename in os.listdir(best_run_dir):
            shutil.copy2(os.path.join(best_run_dir, filename), os.path.join(config.save_path, filename))
        print("最佳结果复制完成。")

    if all_runs_PER_BATTERY_metrics:
        per_batt_summary_df = pd.DataFrame(all_runs_PER_BATTERY_metrics)
        core_cols = ['Battery_ID', 'run', 'seed', 'MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
        ordered_cols = [col for col in core_cols if col in per_batt_summary_df.columns] + [col for col in
                                                                                           per_batt_summary_df.columns
                                                                                           if col not in core_cols]
        per_batt_summary_df = per_batt_summary_df[ordered_cols].sort_values(by=['Battery_ID', 'run'])
        summary_path_per_batt = os.path.join(config.save_path, 'all_runs_per_battery_summary.csv')
        per_batt_summary_df.to_csv(summary_path_per_batt, index=False)
        print(f"“分电池”详细汇总报告已保存到: {summary_path_per_batt}")


if __name__ == '__main__':
    main()