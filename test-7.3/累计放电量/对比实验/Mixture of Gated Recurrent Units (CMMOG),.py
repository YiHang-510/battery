"""
A CMMOG-based lithium-battery SOH estimation method using multi-task learning framework
"""

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib
import warnings
import joblib
from torch.cuda.amp import autocast, GradScaler
import shutil
import math
from typing import Tuple, Dict



# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量V2.0_correct/MoE_Model/vv'  # 建议为新模型创建一个新的保存路径

        # self.train_batteries = [1, 2, 3, 6]
        # self.val_batteries = [5]
        # self.test_batteries = [4]
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


        self.train_batteries = [1, 2, 9, 10]
        self.val_batteries = [18]
        self.test_batteries = [17]

        # self.train_batteries = [3, 12, 19, 11]
        # self.val_batteries = [20]
        # self.test_batteries = [4]

        # self.train_batteries = [13, 14, 21, 22]
        # self.val_batteries = [5]
        # self.test_batteries = [6]

        # self.train_batteries = [16, 8, 15, 7]
        # self.val_batteries = [23]
        # self.test_batteries = [24]

        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7  # A文件的输入特征维度 (电压序列的通道数)
        self.sequence_length = 1  # 您的数据加载器每个样本产生一个时间步
        self.cap_norm = 3.5  # 最大容量的固定归一化系数

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.005
        self.weight_decay = 0.0001
        self.patience = 15  # MoE模型可能需要更多耐心
        self.seed = 2025
        self.mode = 'both'
        self.dropout = 0.2

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
# ======================== 3. 新的 Mixture of Experts 模型架构 =================
# ==============================================================================

# --- 辅助模块: 卷积块 (可复用) ---
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        pad = (k - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# --- 辅助模块: 单个专家网络 ---
# 每个专家接收CNN提取出的序列特征，并给出自己的判断
class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, SeqLen, InputDim]
        _, h_n = self.bigru(x)  # h_n shape: [2*num_layers, B, H]
        # 拼接前向和后向的最后一个隐藏状态
        # h_n is (D*num_layers, N, H) -> (2, B, H)
        # Concatenate the final forward and backward hidden states
        h_n = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1)  # Shape: [B, 2*H]
        return h_n


# --- 主模型: Mixture of Experts (MoE) ---
class MoE_Model(nn.Module):
    def __init__(self,
                 # 输入维度
                 seq_channels: int,
                 scalar_dim: int,
                 # 模型超参数
                 num_experts: int = 4,
                 cnn_channels: int = 32,
                 expert_hidden_dim: int = 64,
                 mlp_hidden_dim: int = 128,
                 out_activation: str = 'sigmoid',  # 'sigmoid' | 'softplus' | 'none'
                 use_homoscedastic: bool = False
                 ):
        super().__init__()
        self.out_activation = out_activation
        self.use_homoscedastic = use_homoscedastic

        # 输入CNN的总通道数 = 电压(1) + 标量特征(2) + 时间(1) = 4
        cnn_input_channels = 1 + scalar_dim + 1

        # --- 共享的CNN主干网络 ---
        self.conv = ConvBlock(cnn_input_channels, cnn_channels)
        self.conv_proj_step = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=1)
        self.conv_pool = nn.AdaptiveAvgPool1d(1)

        # --- 专家网络列表 ---
        self.experts = nn.ModuleList([
            Expert(input_dim=cnn_channels, hidden_dim=expert_hidden_dim) for _ in range(num_experts)
        ])

        # --- 门控网络 ---
        # 接收池化后的特征，为每个专家打分
        self.gate = nn.Sequential(
            nn.Linear(cnn_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_experts)
        )

        # --- 最终的预测头 (Tower) ---
        self.tower = nn.Sequential(
            nn.Linear(2 * expert_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )

        # --- 不确定性学习 (可选) ---
        self.log_var = nn.Parameter(torch.zeros(1), requires_grad=True) if use_homoscedastic else None

        # --- 初始化预测头的权重 ---
        for m in self.tower.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def _build_cnn_input(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        # 您的数据加载器输出 v: [B, 1, 7], s: [B, 2]
        # forward中会将 t_norm unsqueeze 成 [B, 1]
        # 该函数将它们组合成适用于Conv1d的格式

        # 获取序列长度 T (这里是7)
        B, _, T = v.shape
        # 将标量特征 s [B,2] -> [B,2,1] -> [B,2,7]
        s_exp = s.unsqueeze(-1).repeat(1, 1, T)
        # 将时间 t_norm [B,1] -> [B,1,1] -> [B,1,7]
        t_exp = t_norm.unsqueeze(-1).repeat(1, 1, T)

        # 沿通道维度拼接
        x = torch.cat([v, s_exp, t_exp], dim=1)  # [B, 1+2+1, 7] -> [B, 4, 7]
        return x

    def forward(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 确保输入张量维度正确
        if t_norm.ndim == 1:
            t_norm = t_norm.unsqueeze(-1)  # [B] -> [B,1]

        B, _, T = v.shape
        # Conv1d 输入要求 [B, C, L], 您的数据 [B, 1, 7] 正好符合 (1个通道，长度为7的序列)
        assert T == 7, f"Expected v length 7, got {T}"

        # ---- 1. 共享CNN ----
        x = self._build_cnn_input(v, s, t_norm)  # [B, 4, 7]
        feat_step = self.conv(x)  # [B, Cc, 7]
        feat_step = self.conv_proj_step(feat_step)  # [B, Cc, 7]
        feat_pool = self.conv_pool(feat_step).squeeze(-1)  # [B, Cc]

        # 为专家网络准备序列输入: [B, 7, Cc]
        step_seq = feat_step.transpose(1, 2)

        # ---- 2. 专家网络 ----
        expert_outputs = []
        for exp in self.experts:
            expert_outputs.append(exp(step_seq))  # 每个输出: [B, 2*H]
        E = torch.stack(expert_outputs, dim=1)  # [B, K, 2*H]

        # ---- 3. 门控网络 ----
        logits = self.gate(feat_pool)  # [B, K]
        gates = torch.softmax(logits, dim=-1)  # [B, K]

        # ---- 4. 专家输出加权融合 ----
        fused = torch.einsum('bk,bkd->bd', gates, E)  # [B, 2*H]

        # ---- 5. 预测头 ----
        Q = self.tower(fused)  # [B, 1]
        if self.out_activation == 'sigmoid':
            Q = torch.sigmoid(Q)
        elif self.out_activation == 'softplus':
            Q = F.softplus(Q)

        # 最终输出需要展平为 [B]
        Q = Q.squeeze(-1)

        aux: Dict[str, torch.Tensor] = {
            'gates': gates,
        }
        if self.use_homoscedastic and self.log_var is not None:
            aux['log_var'] = self.log_var

        return Q, aux


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
        # .tolist() 会将 (1,7) 的numpy数组变成list, np.array再变回来
        self.sequences = np.array(self.df[sequence_col].tolist(), dtype=np.float32)
        # MoE 模型将7个电压读数作为序列，所以输入形状应为 [B, 1, 7]
        # 我们需要确保数据加载时维度正确
        if self.sequences.ndim == 3 and self.sequences.shape[1] > self.sequences.shape[2]:
            # 如果形状是 [N, 7, 1]，则转置为 [N, 1, 7]
            self.sequences = self.sequences.transpose(0, 2, 1)

        self.scalars = self.df[self.scalar_cols].values.astype(np.float32)
        self.targets = self.df[self.target_col].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 确保序列输出形状是 (1, 7)，以匹配Conv1d的 (C, L)
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)
        return x_seq, x_scalar, cycle_idx, y


# --- 5. 数据加载和预处理 (不变) ---
def load_and_preprocess_data(config):
    # ... (此函数无需修改，与之前完全相同)
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

            # 关键：x.values 会是 (1, 7) 的数组，这正是我们需要的
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

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    config.cycle_norm_min = float(train_df['循环号'].min())
    config.cycle_norm_max = float(train_df['循环号'].max())
    if config.cycle_norm_max <= config.cycle_norm_min:
        config.cycle_norm_max = config.cycle_norm_min + 1.0

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()

    # scaler_seq 需要一个2D数组 [N_samples, N_features]
    # np.vstack 将 (1,7) 的数组列表堆叠成 [Total_samples, 7]
    all_sequences_for_fit = np.vstack(train_df[sequence_col].values)
    scaler_seq.fit(all_sequences_for_fit)

    scaler_scalar.fit(train_df[scalar_feature_cols])

    for df in [train_df, val_df, test_df]:
        # transform也需要 [N, 7] 的输入，所以我们需要先vstack
        sequences_to_transform = np.vstack(df[sequence_col].values)
        transformed_sequences = scaler_seq.transform(sequences_to_transform)
        # 再将变换后的 [N, 7] 分配回DataFrame的每一行
        df[sequence_col] = [row.reshape(1, -1) for row in transformed_sequences]

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


# --- 6. 训练函数 (无需修改) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler, config):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
            batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device)

        t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
        t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
        t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min)
        t_norm = torch.clamp(t_norm, 0.0, 1.0)

        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                outputs, aux = model(batch_seq, batch_scalar, t_norm)
                loss = criterion(outputs, batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs, aux = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 7. 验证/测试函数 (无需修改) ---
def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_cycle_indices = [], [], []
    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
            batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
                batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device)

            t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
            t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
            t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min)
            t_norm = torch.clamp(t_norm, 0.0, 1.0)

            outputs, aux = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs, batch_y)

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
        # ==================== 修改: 实例化新的 MoE_Model =====================
        # ====================================================================
        model = MoE_Model(
            seq_channels=config.sequence_feature_dim,
            scalar_dim=config.scalar_feature_dim,
            num_experts=4,  # 超参数, 可以调整
            cnn_channels=32,  # 超参数, 可以调整
            expert_hidden_dim=64,  # 超参数, 可以调整
            mlp_hidden_dim=128,  # 超参数, 可以调整
            out_activation='sigmoid'
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