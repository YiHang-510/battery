"""
A physics-guided approach for accurate battery SOH estimation using RCMHCRE and BatteryPINN
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
import math
from dataclasses import dataclass
from typing import Tuple
import torch.nn.functional as F


# --- 1. 配置参数 (已修改为新模型) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        # --- 修改: 更新保存路径以反映新模型 ---
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/RCMHCRE/4'

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

        # self.train_batteries = [21, 22, 23, 24]
        # self.val_batteries = [19]
        # self.test_batteries = [20]

        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7  # A文件的输入特征维度 (RCMHCRE要求长度为7)
        self.sequence_length = 1  # 序列长度
        self.cap_norm = 3.5  # 最大容量的固定归一化系数

        # --- 模型超参数 (RCMHCRE_PIRes) ---
        self.entropy_max_scale = 6  # RCMHCRE的最大尺度
        self.d_model = 128  # 编码器的隐藏维度
        self.dropout = 0.1
        self.n_basis = 10  # I-spline 基函数数
        self.degree = 3  # I-spline/B-spline 阶数
        self.n_grid = 512  # I-spline 预计算网格
        self.residual_l2 = 1e-4  # 残差系数 L2 正则

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

        # --- 辅助参数 (自动计算) ---
        self.scalar_feature_dim = len(self.features_from_C) + self.entropy_max_scale


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


# --- 3. 新的预测网络 (RCMHCRE_PIRes) 及其组件 ---

# ------------------------------------------------------------
# 新增: RCMHCRE 特征提取器 (已补全)
# ------------------------------------------------------------
class RCMHCREExtractor(nn.Module):
    """
    计算每个样本（长度为7的一维信号）的6尺度熵特征。
    返回每个尺度的归一化CRE熵，值在[0, ~1.5]范围内（非严格上界）。
    步骤遵循论文大纲：复合多尺度平均 -> 通过FFT进行希尔伯特变换 ->
    功率归一化 -> 累积剩余熵（经验CDF）。
    """

    def __init__(self, max_scale: int = 6):
        super().__init__()
        self.max_scale = max_scale

    @staticmethod
    def _composite_multiscale(x: torch.Tensor, s: int) -> torch.Tensor:
        """对整数尺度s进行复合多尺度平均"""
        B, T = x.shape
        L = (T // s) * s
        if L <= 0: return x.unsqueeze(1)
        xs = x[:, :L]
        ys = torch.stack([xs[:, e:L:s] for e in range(s)], dim=1)  # [B, s, L/s]
        return ys.mean(dim=1)  # [B, L/s]

    @staticmethod
    def _hilbert_transform_fft(y: torch.Tensor) -> torch.Tensor:
        """通过FFT希尔伯特滤波器近似解析信号"""
        Y = torch.fft.rfft(y, dim=-1)
        H = torch.zeros_like(Y.real)  # Create H with real type
        H[..., 0] = 1.0
        if y.size(-1) > 1:
            if y.size(-1) % 2 == 0:
                H[..., 1:-1] = 2.0
                H[..., -1] = 1.0
            else:
                H[..., 1:] = 2.0
        H = H.to(Y.device)
        return torch.fft.irfft(Y * H, n=y.size(-1), dim=-1)

    @staticmethod
    def _power_norm(z: torch.Tensor) -> torch.Tensor:
        """功率归一化"""
        inst_amp = torch.abs(z)
        inst_pow = inst_amp ** 2
        total_pow = torch.sum(inst_pow, dim=-1, keepdim=True)
        # 加上一个小的epsilon防止除以零
        return inst_pow / (total_pow + 1e-9)

    @staticmethod
    def _cre(p: torch.Tensor) -> torch.Tensor:
        """计算累积剩余熵"""
        p_sorted, _ = torch.sort(p, dim=-1, descending=True)
        cdf = torch.cumsum(p_sorted, dim=-1)
        # 确保CDF的最后一个元素为1
        cdf = torch.clamp(cdf, 0.0, 1.0)
        # CRE是对 -log(CDF) 的求和
        entropy = -torch.sum(torch.log(cdf + 1e-9), dim=-1)
        return entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T=7] -> returns [B, max_scale]"""
        B, T = x.shape
        entropies = []
        for s in range(1, self.max_scale + 1):
            y_s = self._composite_multiscale(x, s)
            z_s = self._hilbert_transform_fft(y_s)
            p_s = self._power_norm(z_s)
            h_s = self._cre(p_s)
            entropies.append(h_s)
        return torch.stack(entropies, dim=1)  # [B, max_scale]


# ------------------------------------------------------------
# B-spline / I-spline 基函数 (不变)
# ------------------------------------------------------------
class ISplineBasis(nn.Module):
    def __init__(self, n_basis: int = 10, degree: int = 3, n_grid: int = 512, device=None, dtype=None):
        super().__init__()
        assert n_basis > degree, "n_basis must be > degree."
        self.n_basis, self.degree, self.n_grid = n_basis, degree, n_grid
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        n_knots = n_basis + degree + 1
        interior = n_knots - 2 * (degree + 1)
        interior = max(0, interior)
        interior_knots = torch.linspace(0.0, 1.0, interior + 2, dtype=dtype, device=device)[
                         1:-1] if interior > 0 else torch.empty(0, dtype=dtype, device=device)
        knots = torch.cat([torch.zeros(degree + 1, dtype=dtype, device=device), interior_knots,
                           torch.ones(degree + 1, dtype=dtype, device=device)], dim=0)

        grid = torch.linspace(0.0, 1.0, n_grid, dtype=dtype, device=device)
        B_grid = _bspline_basis_grid(knots, degree, grid)
        denom = (knots[degree + 1: degree + 1 + n_basis] - knots[:n_basis]).clamp_min(1e-8)
        M_grid = (degree + 1) * B_grid / denom

        dx = 1.0 / (n_grid - 1)
        I_grid = torch.zeros_like(M_grid)
        I_grid[1:, :] = torch.cumsum(0.5 * (M_grid[1:, :] + M_grid[:-1, :]) * dx, dim=0)
        I_grid = I_grid / I_grid[-1, :].clamp_min(1e-8)

        self.register_buffer("knots", knots, persistent=False)
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("I_grid", I_grid, persistent=False)

    def eval(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        u = (t * (self.n_grid - 1)).clamp(0, self.n_grid - 1)
        i0 = torch.floor(u).long()
        i1 = (i0 + 1).clamp(max=self.n_grid - 1)
        w = u - i0.float()
        I0 = self.I_grid[i0.squeeze(-1), :]
        I1 = self.I_grid[i1.squeeze(-1), :]
        return (1.0 - w) * I0 + w * I1


def _bspline_basis_grid(knots: torch.Tensor, degree: int, grid: torch.Tensor) -> torch.Tensor:
    device, dtype = knots.device, knots.dtype
    G, M = grid.numel(), knots.numel() - degree - 1
    assert M > 0
    Bk = torch.zeros(G, M, device=device, dtype=dtype)
    for i in range(M):
        left, right = knots[i], knots[i + 1]
        cond = (grid >= left) & (grid < right) if i < M - 1 else (grid >= left) & (grid <= right)
        Bk[:, i] = cond.to(dtype)
    for k in range(1, degree + 1):
        Bk_next = torch.zeros_like(Bk)
        for i in range(M):
            term1 = ((grid - knots[i]) / (knots[i + k] - knots[i]).clamp_min(1e-8)) * Bk[:, i] if (knots[i + k] - knots[
                i]) > 0 else 0.0
            term2 = ((knots[i + k + 1] - grid) / (knots[i + k + 1] - knots[i + 1]).clamp_min(1e-8)) * Bk[:,
                                                                                                      i + 1] if i + 1 < M and (
                        knots[i + k + 1] - knots[i + 1]) > 0 else 0.0
            Bk_next[:, i] = term1 + term2
        Bk = Bk_next
    return Bk


# ------------------------------------------------------------
# 主模型 RCMHCRE_PIRes
# ------------------------------------------------------------
@dataclass
class ModelConfig:
    d_model: int = 128
    dropout: float = 0.1
    n_basis: int = 10
    degree: int = 3
    n_grid: int = 512
    residual_l2: float = 1e-4
    entropy_max_scale: int = 6


class RCMHCRE_PIRes(nn.Module):
    def __init__(self, scalar_dim: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # 1. 熵特征提取器
        self.entropy_extractor = RCMHCREExtractor(cfg.entropy_max_scale)

        # 2. 标量编码器
        self.enc_s = nn.Sequential(
            nn.Linear(scalar_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d)
        )

        # 3. PIRes 头部
        self.basis = ISplineBasis(cfg.n_basis, cfg.degree, cfg.n_grid)
        self.c0 = nn.Parameter(torch.randn(cfg.n_basis))
        self.res_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_basis)
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor):
        # 1) 编码
        # v: [B, 1, 7], s: [B, C], t_norm: [B]
        v_squeezed = v.squeeze(1)  # -> [B, 7]
        entropy_feats = self.entropy_extractor(v_squeezed)  # -> [B, max_scale]

        # 拼接熵特征和原始标量特征
        combined_scalars = torch.cat([s, entropy_feats], dim=1)
        h = self.enc_s(combined_scalars)

        # 2) 时间规范化与基函数（递增基！）
        t_norm = t_norm.clamp(0.0, 1.0)
        B_inc = self.basis.eval(t_norm)

        # 3) 累计量 S(t)
        c0_pos = F.softplus(self.c0)
        S_main = (B_inc @ c0_pos) / self.cfg.n_basis

        c_h = F.softplus(self.res_head(h))
        S_res = (B_inc * c_h).sum(dim=-1) / self.cfg.n_basis

        gamma = 1.0
        S = S_main + gamma * S_res

        # 4) 有界且单调下降的最大容量（归一化域）
        Q = torch.exp(-S)

        return Q, {"S_main": S_main, "S_res": S_res, "gamma": gamma, "c_h": c_h}


# --- 4. 数据集定义 (不变) ---
class BatteryMultimodalDataset(Dataset):
    def __init__(self, dataframe, sequence_col, scalar_cols, target_col):
        self.df = dataframe.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.scalar_cols = scalar_cols
        self.target_col = target_col
        self.sequences = np.array(self.df[self.sequence_col].tolist(), dtype=np.float32)
        # --- 修改: 只选择Config中定义的标量特征 ---
        self.scalars = self.df[scalar_cols].values.astype(np.float32)
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
    # --- 注意: scalar_feature_cols现在只包含来自C文件的特征 ---
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
    # --- 只对C文件中的特征进行fit ---
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


# --- 6. 训练/验证/测试函数 (与之前版本完全兼容) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler, config):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        batch_seq, batch_scalar, batch_cycle_idx, batch_y = \
            batch_seq.to(device), batch_scalar.to(device), batch_cycle_idx.to(device), batch_y.to(device)

        t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
        t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
        t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min).clamp_min(1.0)
        t_norm = torch.clamp(t_norm, 0.0, 1.0)

        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                outputs, aux = model(batch_seq, batch_scalar, t_norm)
                loss = criterion(outputs, batch_y)
                loss_reg = model.cfg.residual_l2 * (aux["c_h"] ** 2).mean()
                loss = loss + loss_reg
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs, aux = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs, batch_y)
            loss_reg = model.cfg.residual_l2 * (aux["c_h"] ** 2).mean()
            loss = loss + loss_reg
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


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
            t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min).clamp_min(1.0)
            t_norm = torch.clamp(t_norm, 0.0, 1.0)

            outputs, aux = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs, batch_y)
            loss_reg = model.cfg.residual_l2 * (aux["c_h"] ** 2).mean()
            loss = loss + loss_reg

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


# --- 7. 可视化和工具函数 (不变) ---
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


# --- 8. 主执行函数 (已更新模型初始化) ---
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

        # --- 修改: 初始化新模型 ---
        model_cfg = ModelConfig(
            d_model=config.d_model, dropout=config.dropout,
            n_basis=config.n_basis, degree=config.degree,
            n_grid=config.n_grid, residual_l2=config.residual_l2,
            entropy_max_scale=config.entropy_max_scale
        )
        model = RCMHCRE_PIRes(
            scalar_dim=config.scalar_feature_dim,
            cfg=model_cfg
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

            # --- 反归一化 ---
            test_preds_orig = test_preds * config.cap_norm
            test_labels_orig = test_labels * config.cap_norm
            test_preds_orig = np.clip(test_preds_orig, a_min=0.0, a_max=None)

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
