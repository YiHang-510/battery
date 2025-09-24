import os
import random
from typing import Dict, Tuple

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
import torch.nn.functional as F


# --- 1. 配置参数 (已更新以适应新模型) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        # 更新保存路径以反映新模型
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/消融-LSTMRFR2-最大容量/20'

        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 15, 16, 17, 18, 21, 22, 23, 24]
        # self.val_batteries = [5, 11, 13, 19]
        # self.test_batteries = [6, 12, 14, 20]

        self.train_batteries = [21, 22, 23, 24]
        self.val_batteries = [19]
        self.test_batteries = [20]

        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7  # A文件的输入特征维度
        self.sequence_length = 1  # 序列长度
        self.cap_norm = 3.5  # 最大容量的固定归一化系数

        # --- 模型超参数 (CrossScaleLSTM) ---
        self.lstm_hidden = 16
        self.mlp_hidden = 64
        self.dropout = 0.1
        self.out_activation = 'sigmoid'  # 'sigmoid' or 'softplus' or None

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.patience = 10
        self.seed = 2025
        self.mode = 'both'

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

        # --- 时间(循环号)归一化 (min-max to [0,1]) ---
        self.cycle_norm_min = None
        self.cycle_norm_max = None

        # --- 辅助参数 ---
        self.scalar_feature_dim = len(self.features_from_C)


# --- 2. 固定随机种子 (不变) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 新的预测网络 (CrossScaleLSTM) 及其组件 ---

class TinyLSTM(nn.Module):
    """A small LSTM to process a single feature sequence."""

    def __init__(self, in_dim: int, hid: int):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True)
        self.out = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Sequence, Feature]
        _, (h, _) = self.lstm(x)
        # h shape: [1, Batch, Hidden], squeeze to [Batch, Hidden]
        return self.out(h.squeeze(0))


class RFR2Surrogate(nn.Module):
    """
    A surrogate module for fusing features from different groups and context.
    'RFR2' likely implies a complex fusion, here implemented with an MLP.
    """

    def __init__(self, n_single: list, mlp_hidden: int, dropout: float):
        super().__init__()
        # n_single defines features per group, e.g., [5, 3, 4, 2]
        # Context dim is scalar features (2) + time (1) = 3
        in_dim = sum(n_single) + 3
        self.fuser_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, singles_out: Dict[str, torch.Tensor], context: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # singles_out contains tensors for 'PVF', 'LVF', 'GSF', 'EPF'
        fused_singles = torch.cat([
            singles_out['PVF'],
            singles_out['LVF'],
            singles_out['GSF'],
            singles_out['EPF']
        ], dim=1)

        full_input = torch.cat([fused_singles, context], dim=1)
        output = self.fuser_mlp(full_input)

        aux = {'fused_input': full_input}
        return output, aux


class CrossScaleFeatures(nn.Module):
    """
    A placeholder feature engineering module that extracts 14 scalar features
    from the input voltage sequence, mimicking the expected behavior.
    """

    def __init__(self):
        super().__init__()

    def forward(self, v: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Input v is expected to be [B, 7]
        # Basic statistics
        v_mean = v.mean(dim=1, keepdim=True)
        v_std = v.std(dim=1, keepdim=True)
        v_max = v.max(dim=1, keepdim=True)[0]
        v_min = v.min(dim=1, keepdim=True)[0]
        v_range = v_max - v_min

        # Point-based and difference features
        f1 = v_mean
        f2 = v_std
        f3 = v_range
        f4 = v[:, 0:1]  # First point
        f5 = v[:, -1:]  # Last point
        f6 = v[:, -1:] - v[:, 0:1]  # Total drop
        f7 = v[:, 1:2] - v[:, 0:2].mean(dim=1, keepdim=True)
        f8 = v[:, 2:3] - v[:, 1:3].mean(dim=1, keepdim=True)
        f9 = v[:, 3:4] - v[:, 2:4].mean(dim=1, keepdim=True)
        f10 = v[:, 4:5] - v[:, 3:5].mean(dim=1, keepdim=True)
        f11 = v[:, 5:6] - v[:, 4:6].mean(dim=1, keepdim=True)
        f12 = v[:, 6:7] - v[:, 5:7].mean(dim=1, keepdim=True)
        f13 = v[:, 3:4]  # Middle point
        f14 = v_max

        # Concatenate all 14 features
        feats = torch.cat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14], dim=1)

        # Group features as specified
        groups = {
            'PVF': feats[:, 0:5],  # 5 features
            'LVF': feats[:, 5:8],  # 3 features
            'GSF': feats[:, 8:12],  # 4 features
            'EPF': feats[:, 12:14]  # 2 features
        }
        return feats, groups


class Model(nn.Module):
    def __init__(self,
                 lstm_hidden: int = 16,
                 mlp_hidden: int = 64,
                 dropout: float = 0.1,
                 out_activation: str = 'sigmoid'):
        super().__init__()
        self.out_activation = out_activation
        self.fe = CrossScaleFeatures()
        # 14 parallel tiny LSTMs for 14 engineered features
        self.single_lstms = nn.ModuleList([TinyLSTM(in_dim=1, hid=lstm_hidden) for _ in range(14)])
        # Two-level fusion surrogate
        self.fuser = RFR2Surrogate(n_single=[5, 3, 4, 2], mlp_hidden=mlp_hidden, dropout=dropout)

    @staticmethod
    def _to_seq(x: torch.Tensor, T: int = 3) -> torch.Tensor:
        # Expand scalar feature to a tiny sequence for LSTM input
        return x.unsqueeze(-1).repeat(1, T, 1)  # [B, T, 1]

    def forward(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # v arrives as [B, 1, 7], squeeze to [B, 7] for feature extraction
        v_squeezed = v.squeeze(1)

        feats, groups = self.fe(v_squeezed)  # feats: [B,14]

        # Split 14 features into categories
        singles_cat = {
            'PVF': groups['PVF'],  # [B, 5]
            'LVF': groups['LVF'],  # [B, 3]
            'GSF': groups['GSF'],  # [B, 4]
            'EPF': groups['EPF'],  # [B, 2]
        }

        # Per-single LSTM pre-estimation
        singles_out: Dict[str, torch.Tensor] = {}
        idx = 0
        for cat in ['PVF', 'LVF', 'GSF', 'EPF']:
            X = singles_cat[cat]
            outs = []
            for j in range(X.size(1)):
                seq = self._to_seq(X[:, j:j + 1], T=3)
                yj = self.single_lstms[idx](seq)
                outs.append(yj)
                idx += 1
            singles_out[cat] = torch.cat(outs, dim=1)

        # Context for final fusion: [s(2), t_norm(1)]
        context = torch.cat([s, t_norm], dim=1)

        y, aux_fuse = self.fuser(singles_out, context)
        Q = y.squeeze(-1)  # Ensure output is [B]

        if self.out_activation == 'sigmoid':
            Q = torch.sigmoid(Q)
        elif self.out_activation == 'softplus':
            Q = F.softplus(Q)

        # Collect auxiliary info for potential analysis
        aux = {}
        aux.update(aux_fuse)
        aux['features'] = feats
        return Q, aux


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
        batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
            batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device)

        # 归一化循环号
        t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
        t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
        t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min).clamp(min=1e-8)  # Added epsilon for safety
        t_norm = torch.clamp(t_norm, 0.0, 1.0).unsqueeze(1)  # Reshape to [B, 1] for model

        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                outputs, _ = model(batch_seq, batch_scalar, t_norm)
                loss = criterion(outputs, batch_y)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs, _ = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
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
            t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min).clamp(min=1e-8)  # Added epsilon for safety
            t_norm = torch.clamp(t_norm, 0.0, 1.0).unsqueeze(1)  # Reshape to [B, 1] for model

            outputs, _ = model(batch_seq, batch_scalar, t_norm)
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
    plt.ylabel('Normalized Capacity', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_diagonal_results(labels, preds, title, save_path):
    plt.figure(figsize=(8, 8))
    min_val, max_val = min(np.min(labels), np.min(preds)) * 0.98, max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('True Capacity (Normalized)')
    plt.ylabel('Predicted Capacity (Normalized)')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=300)
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
        print(f"\n{'=' * 30}\n 开始第 {run_number}/{num_runs} 次实验 | 随机种子: {current_seed} \n{'=' * 30}")

        try:
            train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
        except (FileNotFoundError, ValueError) as e:
            print(f"数据加载失败: {e}")
            continue

        joblib.dump(scalers, os.path.join(run_save_path, 'scalers.pkl'))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4,
                                 pin_memory=True)
        print(f"数据加载完成。训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        # --- 修改: 初始化新模型 ---
        model = Model(
            lstm_hidden=config.lstm_hidden,
            mlp_hidden=config.mlp_hidden,
            dropout=config.dropout,
            out_activation=config.out_activation
        ).to(config.device)

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
                    f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val R2: {val_metrics['R2']:.4f}")
                log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                             **{'val_' + k: v for k, v in val_metrics.items()}}
                metrics_log.append(log_entry)

                if val_loss < best_val_loss_this_run:
                    best_val_loss_this_run = val_loss
                    torch.save(model.state_dict(), os.path.join(run_save_path, 'best_model.pth'))
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= config.patience:
                        print(f"\nEarly stopping after {config.patience} epochs with no improvement.")
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

            # Clip predictions to be non-negative
            test_preds_orig = np.clip(test_preds, a_min=0.0, a_max=None)
            test_labels_orig = test_labels

            print("\n--- 本轮评估结果 (按单电池) ---")
            eval_df = pd.DataFrame(
                {'battery_id': test_dataset.df['battery_id'].values, 'cycle': test_cycle_nums, 'true': test_labels_orig,
                 'pred': test_preds_orig})
            per_battery_metrics_list = []
            for batt_id in sorted(list(set(config.test_batteries))):
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
                plot_results(batt_true, batt_pred, f'Battery {batt_id}: True vs Predicted',
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
            print(f"测试集(汇总): RMSE={final_test_metrics['RMSE']:.6f}, R2={final_test_metrics['R2']:.4f}")

            if best_val_loss_this_run < best_run_val_loss:
                best_run_val_loss, best_run_dir, best_run_number = best_val_loss_this_run, run_save_path, run_number
                print(f"*** 新的最佳表现！验证集损失: {best_val_loss_this_run:.6f} ***")

    print(f"\n\n{'=' * 50}\n 所有实验均已完成。\n{'=' * 50}")
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 ---\n", summary_df.round(4))
        print(f"\n汇总指标已保存到: {summary_path}")

    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (验证集损失最低: {best_run_val_loss:.6f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")
        for filename in os.listdir(best_run_dir):
            shutil.copy2(os.path.join(best_run_dir, filename), os.path.join(config.save_path, filename))
        print("最佳结果复制完成。")

    if all_runs_PER_BATTERY_metrics:
        per_batt_summary_df = pd.DataFrame(all_runs_PER_BATTERY_metrics)
        summary_path_per_batt = os.path.join(config.save_path, 'all_runs_per_battery_summary.csv')
        per_batt_summary_df.to_csv(summary_path_per_batt, index=False)
        print(f"“分电池”详细汇总报告已保存到: {summary_path_per_batt}")


if __name__ == '__main__':
    main()

