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
from typing import Tuple, Optional
import torch.nn.functional as F


# --- 1. 配置参数 (已更新以适应新模型) ---
# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        # ... (原有路径设置保持不变) ...
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量V2.0_correct/TM_PIRes-消融-wo_scalars/vv'

        # ... (原有电池划分保持不变) ...
        # self.train_batteries = [1, 2, 9, 10]
        # self.val_batteries = [17]
        # self.test_batteries = [18]

        # self.train_batteries = [3, 12, 19, 11]
        # self.val_batteries = [20]
        # self.test_batteries = [4]

        # self.train_batteries = [13, 14, 21, 22]
        # self.val_batteries = [5]
        # self.test_batteries = [6]

        self.train_batteries = [16, 8, 15, 7]
        self.val_batteries = [23]
        self.test_batteries = [24]

        # 原始连续标量特征
        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7  
        self.sequence_length = 1

        # === [新增] 消融实验配置 ===
        # 可选: "full", "wo_tau336", "wo_scalars"
        self.ablation_case = "wo_scalars"  
        self.tau336_col_name = '3.3~3.6V充电时间(s)'
        self.tau336_mask_col = 'mask_tau_3p3_3p6'
        # =========================

        # ... (其余参数 d_model, epochs 等保持不变) ...
        self.d_model = 128
        self.n_blocks = 3
        self.kernel_sizes = (3, 5, 7)
        self.dropout = 0.1
        self.n_basis = 10
        self.degree = 3
        self.n_grid = 512
        self.residual_l2 = 1e-3
        
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.patience = 15
        self.seed = 2025
        self.mode = 'both'
        self.task_weights = {"q": 1.0, "soh": 10000000}
        
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        
        self.cycle_norm_min = None
        self.cycle_norm_max = None

        # 注意：self.scalar_feature_dim 将在 load_and_preprocess_data 中根据消融情况动态更新
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


# --- 3. 新的预测网络 (TM_PIRes) 及其组件 ---

# ------------------------------------------------------------
# B-spline / I-spline 基函数
# ------------------------------------------------------------
def _bspline_basis_grid(knots: torch.Tensor, degree: int, grid: torch.Tensor) -> torch.Tensor:
    device = knots.device
    dtype = knots.dtype
    G = grid.numel()
    M = knots.numel() - degree - 1
    assert M > 0, "n_basis must be > degree."

    Bk = torch.zeros(G, M, device=device, dtype=dtype)
    for i in range(M):
        left, right = knots[i], knots[i + 1]
        cond = (grid >= left) & (grid < right)
        if i == M - 1:
            cond = cond | (grid == right)
        Bk[:, i] = cond.to(dtype)

    for k in range(1, degree + 1):
        Bk_next = torch.zeros_like(Bk)
        for i in range(M):
            term1 = 0.0
            denom1 = knots[i + k] - knots[i]
            if denom1 > 0:
                term1 = ((grid - knots[i]) / denom1) * Bk[:, i]

            term2 = 0.0
            if (i + 1) < M:
                denom2 = knots[i + k + 1] - knots[i + 1]
                if denom2 > 0:
                    term2 = ((knots[i + k + 1] - grid) / denom2) * Bk[:, i + 1]
            Bk_next[:, i] = term1 + term2
        Bk = Bk_next
    return Bk


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


# ------------------------------------------------------------
# TimeMixer
# ------------------------------------------------------------
class DepthwiseTemporalMix(nn.Module):
    def __init__(self, d_model: int, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model) for k in kernel_sizes])
        self.proj = nn.Conv1d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.zeros(len(kernel_sizes)))
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        feats = [self.act(conv(x_t)) for conv in self.convs]
        a = F.softmax(self.alpha, dim=0)
        y = sum(a[i] * feats[i] for i in range(len(feats)))
        return self.drop(self.proj(y)).transpose(1, 2)


class ChannelMix(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout)
        )

    def forward(self, x): return self.ff(x)


class TimeMixerBlock(nn.Module):
    def __init__(self, d_model: int, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.tmix = DepthwiseTemporalMix(d_model, kernel_sizes, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cmix = ChannelMix(d_model, expansion=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.tmix(self.norm1(x))
        x = x + self.cmix(self.norm2(x))
        return x


class TimeMixerStack(nn.Module):
    def __init__(self, d_model: int, n_blocks: int, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TimeMixerBlock(d_model, kernel_sizes, dropout) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return x


# ------------------------------------------------------------
# TM_PIRes: 主模型
# ------------------------------------------------------------
@dataclass
class TMPIResConfig:
    d_model: int = 128
    n_blocks: int = 3
    kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.1
    n_basis: int = 10
    degree: int = 3
    n_grid: int = 512
    residual_l2: float = 1e-4


class TM_PIRes(nn.Module):
    def __init__(self, seq_channels: int, scalar_dim: int, cfg: TMPIResConfig, n_terms: int = 16):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # --- (1) 序列编码器 ---
        self.embed = nn.Linear(seq_channels, d)
        self.tm = TimeMixerStack(d, cfg.n_blocks, cfg.kernel_sizes, cfg.dropout)
        
        # === [修改] 兼容 scalar_dim=0 ===
        if scalar_dim > 0:
            self.enc_s = nn.Sequential(
                nn.Linear(scalar_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d)
            )
        else:
            self.enc_s = None
        # ==============================

        # --- (2) Q 预测头 ---
        self.basis = ISplineBasis(cfg.n_basis, cfg.degree, cfg.n_grid)
        self.c0 = nn.Parameter(torch.randn(cfg.n_basis))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.res_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_basis)
        )

        # --- (3) SOH 预测头 ---
        self.n_terms = n_terms
        self.soh_param_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_terms * 3)
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # --- (A) 特征提取 ---
        x = self.embed(v)
        H = self.tm(x)
        h_seq = H.mean(dim=1)
        
        # === [修改] 仅当 encoder 存在时融合标量特征 ===
        if self.enc_s is not None:
            h = h_seq + self.enc_s(s)
        else:
            h = h_seq
        # ===========================================

        # --- (B) Q 预测 ---
        B_I = self.basis.eval(t_norm)
        c0_pos = F.softplus(self.c0)
        m = (B_I @ c0_pos) + self.b0
        c_h = F.softplus(self.res_head(h))
        R = (B_I * c_h).sum(dim=-1)
        Q = m + R

        # --- (C) SOH 预测 ---
        params = self.soh_param_head(h)
        a, b, d = torch.split(params, self.n_terms, dim=1)

        a = torch.sigmoid(a)
        b = -F.softplus(b) - 1e-6
        d = torch.sigmoid(d) * 0.5

        t = t_norm.view(-1, 1)
        out_terms = a * torch.exp(b * t) + d
        out_sum = out_terms.sum(dim=1)
        SOH_pred = torch.sigmoid(out_sum)

        return (Q, SOH_pred), {"c_h": c_h}

# --- 4. 数据集定义 (不变) ---
class BatteryMultimodalDataset(Dataset):
    def __init__(self, dataframe, sequence_col, scalar_cols, target_q_col, target_soh_col):
        self.df = dataframe.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.scalar_cols = scalar_cols
        self.target_q_col = target_q_col
        self.target_soh_col = target_soh_col

        self.sequences = np.array(self.df[self.sequence_col].tolist(), dtype=np.float32)
        
        # === [修改] 兼容 scalar_cols 为空的情况 ===
        if len(self.scalar_cols) == 0:
            # 创建 shape=(N, 0) 的空数组
            self.scalars = np.zeros((len(self.df), 0), dtype=np.float32)
        else:
            self.scalars = self.df[self.scalar_cols].values.astype(np.float32)
        # ========================================
            
        self.targets_q = self.df[self.target_q_col].values.astype(np.float32)
        self.targets_s = self.df[self.target_soh_col].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        x_seq    = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        y_q      = torch.tensor(self.targets_q[idx], dtype=torch.float32)
        y_s      = torch.tensor(self.targets_s[idx], dtype=torch.float32)
        cycle_idx= torch.tensor(self.cycle_indices[idx], dtype=torch.long)
        return x_seq, x_scalar, cycle_idx, y_q, y_s


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

    # 依据每个电池首循环的最大容量构造 SOH
    full_df = full_df.sort_values(["battery_id", "循环号"])
    first_cap = full_df.groupby("battery_id")["最大容量(Ah)"].transform("first")
    full_df["SOH"] = (full_df["最大容量(Ah)"] / first_cap).clip(lower=0.0, upper=1.0)

    target_q_col = '累计放电容量(Ah)'
    target_soh_col = 'SOH'
    sequence_col = 'voltage_sequence'
    
    # === [核心修改] 缺失值处理与 Mask 生成 ===
    # 1. 打印原始缺失率
    if config.tau336_col_name in full_df.columns:
        missing_count = full_df[config.tau336_col_name].isna().sum()
        total_count = len(full_df)
        print(f"Dataset Info: '{config.tau336_col_name}' missing ratio: {missing_count/total_count:.2%}")
        
        # 2. 生成 mask: 1=可用, 0=缺失
        full_df[config.tau336_mask_col] = full_df[config.tau336_col_name].notna().astype(np.float32)
        
        # 3. 填充缺失值为 0.0 (为了保证 tensor 数值有效，物理上 mask=0 已屏蔽其意义)
        full_df[config.tau336_col_name] = full_df[config.tau336_col_name].fillna(0.0)
    else:
        print(f"Warning: '{config.tau336_col_name}' not found, ablation logic might fail.")
    # ========================================

    # 验证特征存在性 (仅验证 Config 中定义的原始连续特征)
    for col in config.features_from_C:
        if col not in full_df.columns:
            raise ValueError(f"您选择的特征 '{col}' 不存在于加载的数据中。")

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # 循环号归一化
    config.cycle_norm_min = float(train_df['循环号'].min())
    config.cycle_norm_max = float(train_df['循环号'].max())
    if config.cycle_norm_max <= config.cycle_norm_min:
        config.cycle_norm_max = config.cycle_norm_min + 1.0

    # === [核心修改] 仅对连续标量做 StandardScaler ===
    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler() # 仅用于连续特征
    
    continuous_cols = config.features_from_C  # ['恒压', '3.3-3.6']
    
    # Fit (仅连续特征)
    scaler_seq.fit(np.vstack(train_df[sequence_col].values))
    scaler_scalar.fit(train_df[continuous_cols])

    # Transform (仅连续特征)
    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        # 覆写原始列为归一化后的值
        df.loc[:, continuous_cols] = scaler_scalar.transform(df[continuous_cols])
    # ============================================

    # === [核心修改] 根据消融 Case 选择最终输入的 scalar_cols ===
    if config.ablation_case == "full":
        # 完整: [tau_CV, tau_336, mask]
        used_scalar_cols = config.features_from_C + [config.tau336_mask_col]
        
    elif config.ablation_case == "wo_tau336":
        # 去除 3.3-3.6: 仅 [tau_CV]
        # 假设 tau_CV 是列表中的第一个，或者通过名字过滤
        used_scalar_cols = [c for c in config.features_from_C if c != config.tau336_col_name]
        
    elif config.ablation_case == "wo_scalars":
        # 去除所有标量: []
        used_scalar_cols = []
    else:
        raise ValueError(f"Unknown ablation case: {config.ablation_case}")

    print(f"Ablation Case: '{config.ablation_case}'")
    print(f"Used Scalar Cols ({len(used_scalar_cols)}): {used_scalar_cols}")

    # 动态更新 config 中的维度，以便模型初始化时使用正确维度
    config.scalar_feature_dim = len(used_scalar_cols)
    # ========================================================

    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, used_scalar_cols, target_q_col, target_soh_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, used_scalar_cols, target_q_col, target_soh_col)
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, used_scalar_cols, target_q_col, target_soh_col)

    scalers = {
        'sequence': scaler_seq,
        'scalar': scaler_scalar, # 注意：这只包含连续特征的 scaler
        'cycle_norm': {'min': config.cycle_norm_min, 'max': config.cycle_norm_max}
    }
    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练函数 (已修改以适配新模型) ---
def train_epoch(model, dataloader, optimizer, criterion, config):
    """
    训练一个 epoch
    (函数描述不变)
    """
    model.train()

    # --- 修复AMP逻辑 ---
    use_amp = getattr(config, "use_amp", False)
    device_type = config.device.type

    # GradScaler 仅在 CUDA AMP 时启用
    scaler = GradScaler(enabled=(use_amp and device_type == 'cuda'))

    # 为 autocast 选择正确的 dtype
    if device_type == 'cuda':
        amp_dtype = torch.float16 # CUDA 通常使用 float16
    else:
        amp_dtype = torch.bfloat16 # CPU 必须使用 bfloat16
    # --- 修复结束 ---

    total_loss = 0.0

    for batch in dataloader:
        batch_seq, batch_scalar, batch_cycle_idx, batch_y_q, batch_y_s = batch
        batch_seq       = batch_seq.to(config.device)
        batch_scalar    = batch_scalar.to(config.device)
        batch_cycle_idx = batch_cycle_idx.to(config.device)
        batch_y_q       = batch_y_q.to(config.device)
        batch_y_s       = batch_y_s.to(config.device)

        # 归一化循环号 -> t_norm ∈ [0,1]
        t_min  = torch.as_tensor(getattr(config, "cycle_norm_min", 0.0), device=config.device, dtype=torch.float32)
        t_max  = torch.as_tensor(getattr(config, "cycle_norm_max", 1.0), device=config.device, dtype=torch.float32)
        t_norm = torch.clamp((batch_cycle_idx.float() - t_min) / (t_max - t_min + 1e-12), 0.0, 1.0)

        optimizer.zero_grad(set_to_none=True)

        # --- 修复AMP逻辑 ---
        # 统一使用 autocast 上下文
        with autocast(dtype=amp_dtype, enabled=use_amp):
            (outputs, aux) = model(batch_seq, batch_scalar, t_norm)
            pred_q, pred_s = outputs
            loss_q   = criterion(pred_q, batch_y_q)
            loss_soh = criterion(pred_s, batch_y_s)
            if torch.rand(1) < 0.01:  # 仅随机打印1%的批次，避免刷屏
                print(f"\n[Raw Loss Check] Q_loss: {loss_q.item():.6f}, SOH_loss: {loss_soh.item():.6f}")
            loss_reg = getattr(model.cfg, "residual_l2", getattr(config, "residual_l2", 0.0)) * (aux.get("c_h", 0.0) ** 2).mean()
            loss = config.task_weights["q"] * loss_q + config.task_weights["soh"] * loss_soh + loss_reg

        if use_amp and device_type == 'cuda':
            # 仅在 CUDA AMP 时使用 scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 适用于: (1) 非AMP 训练 (2) CPU AMP 训练
            loss.backward()
            optimizer.step()
        # --- 修复结束 ---

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(dataloader))

    return avg_loss

def evaluate(model, dataloader, criterion, device, config):
    """
    验证 / 测试
    (主体逻辑不变，仅在最后计算 metrics 时增加 MAPE 并调整顺序)
    """
    model.eval()
    total_loss = 0.0
    all_q_preds, all_q_labels = [], []
    all_s_preds, all_s_labels = [], []
    all_cycle_indices = []

    with torch.no_grad():
        for batch in dataloader:
            batch_seq, batch_scalar, batch_cycle_idx, batch_y_q, batch_y_s = batch
            batch_seq       = batch_seq.to(device)
            batch_scalar    = batch_scalar.to(device)
            batch_cycle_idx = batch_cycle_idx.to(device)
            batch_y_q       = batch_y_q.to(device)
            batch_y_s       = batch_y_s.to(device)

            # t_norm
            t_min  = torch.as_tensor(getattr(config, "cycle_norm_min", 0.0), device=device, dtype=torch.float32)
            t_max  = torch.as_tensor(getattr(config, "cycle_norm_max", 1.0), device=device, dtype=torch.float32)
            t_norm = torch.clamp((batch_cycle_idx.float() - t_min) / (t_max - t_min + 1e-12), 0.0, 1.0)

            (outputs, aux) = model(batch_seq, batch_scalar, t_norm)
            pred_q, pred_s = outputs

            loss_q   = criterion(pred_q, batch_y_q)
            loss_soh = criterion(pred_s, batch_y_s)
            loss_reg = getattr(model.cfg, "residual_l2", getattr(config, "residual_l2", 0.0)) * (aux.get("c_h", 0.0) ** 2).mean()
            loss = config.task_weights["q"] * loss_q + config.task_weights["soh"] * loss_soh + loss_reg
            total_loss += loss.item()

            all_q_preds.append(pred_q.detach().cpu().numpy());  all_q_labels.append(batch_y_q.detach().cpu().numpy())
            all_s_preds.append(pred_s.detach().cpu().numpy());  all_s_labels.append(batch_y_s.detach().cpu().numpy())
            all_cycle_indices.append(batch_cycle_idx.detach().cpu().numpy())

    # 拼接
    q_preds = np.concatenate(all_q_preds, axis=0).reshape(-1)
    q_lbls  = np.concatenate(all_q_labels, axis=0).reshape(-1)
    s_preds = np.concatenate(all_s_preds, axis=0).reshape(-1)
    s_lbls  = np.concatenate(all_s_labels, axis=0).reshape(-1)
    cycle_indices = np.concatenate(all_cycle_indices, axis=0).reshape(-1)

    # === [修改] 增加 MAPE 并调整字典定义顺序 (尽管字典顺序不决定最终CSV顺序，但保持一致较好) ===
    metrics_q = {
        "Q_MAE":  mean_absolute_error(q_lbls, q_preds),
        "Q_MAPE": mean_absolute_percentage_error(q_lbls, q_preds), # 新增
        "Q_MSE":  mean_squared_error(q_lbls, q_preds),
        "Q_RMSE": np.sqrt(mean_squared_error(q_lbls, q_preds)),
        "Q_R2":   r2_score(q_lbls, q_preds),
    }
    metrics_s = {
        "SOH_MAE":  mean_absolute_error(s_lbls, s_preds),
        "SOH_MAPE": mean_absolute_percentage_error(s_lbls, s_preds), # 新增
        "SOH_MSE":  mean_squared_error(s_lbls, s_preds),
        "SOH_RMSE": np.sqrt(mean_squared_error(s_lbls, s_preds)),
        "SOH_R2":   r2_score(s_lbls, s_preds),
    }
    # ===================================================================================

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, metrics_q, metrics_s, q_preds, q_lbls, cycle_indices, s_preds, s_lbls


# --- 8. 可视化和工具函数  ---

def plot_soh_results(labels, preds, title, save_path):
    """ 专用于SOH的绘图函数 """
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True SOH', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predicted SOH', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('SOH (State of Health)', fontsize=12)  # <--- 修改了Y轴
    plt.legend()
    plt.grid(True)
    plt.ylim(min(0.7, np.min(labels) * 0.95), max(1.0, np.max(labels) * 1.0))  # SOH专用Y轴
    plt.savefig(save_path, dpi=1200)
    plt.close()


def plot_soh_diagonal_results(labels, preds, title, save_path):
    """ 专用于SOH的对角线绘图函数 """
    plt.figure(figsize=(8, 8))
    # SOH范围通常在0.7-1.0之间
    min_val, max_val = min(np.min(labels), np.min(preds)) * 0.98, max(np.max(labels), np.max(preds)) * 1.02
    min_val = max(0.0, min_val)  # SOH不应小于0
    max_val = min(1.2, max_val)  # SOH不应大于1 (稍放宽)

    plt.scatter(labels, preds, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('True SOH', fontsize=12)  # <--- 修改了X轴
    plt.ylabel('Predicted SOH', fontsize=12)  # <--- 修改了Y轴
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()
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
    min_val, max_val = min(np.min(labels), np.min(preds)) * 0.98, max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('True Cumulative Discharge Capacity (Ah)', fontsize=12)
    plt.ylabel('Predicted Cumulative Discharge Capacity (Ah)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 9. 主执行函数 (已更新模型初始化和函数调用) ---
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
        tm_config = TMPIResConfig(
            d_model=config.d_model, n_blocks=config.n_blocks, kernel_sizes=config.kernel_sizes,
            dropout=config.dropout, n_basis=config.n_basis, degree=config.degree,
            n_grid=config.n_grid, residual_l2=config.residual_l2
        )
        model = TM_PIRes(
            seq_channels=config.sequence_feature_dim,
            scalar_dim=config.scalar_feature_dim,
            cfg=tm_config
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
                # --- 修改: 传入 config ---
                train_loss = train_epoch(model, train_loader, optimizer, criterion, config)
                val_loss, val_q, val_s, *_ = evaluate(model, val_loader, criterion, config.device, config)
                print(f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | "
                      f"验证Q_R2: {val_q['Q_R2']:.4f} | 验证SOH_R2: {val_s['SOH_R2']:.4f}")
                log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                             **{'val_' + k: v for k, v in {**val_q, **val_s}.items()}}

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

            # --- 修改 1: 正确解包 evaluate 的 8 个返回值 ---
            # evaluate 返回: avg_loss, metrics_q, metrics_s, q_preds, q_lbls, cycle_indices, s_preds, s_lbls
            _, test_metrics_q, test_metrics_s, test_preds_q, test_labels_q, test_cycle_nums, test_preds_s, test_labels_s = evaluate(
                model, test_loader, criterion, config.device, config
            )

            # --- 修改 2: 分别处理 Q 和 SOH 的预测结果 ---
            # (您的代码中已注释掉反归一化，这里保持一致)
            test_preds_orig_q = np.clip(test_preds_q, a_min=0.0, a_max=None)
            test_labels_orig_q = test_labels_q

            test_preds_orig_s = np.clip(test_preds_s, a_min=0.0, a_max=1.0)  # SOH 裁剪到 [0, 1]
            test_labels_orig_s = test_labels_s

            print("\n--- 本轮评估结果 (按单电池) ---")

            # --- 修改 3: eval_df 包含 Q 和 SOH 的 真值/预测值 ---
            eval_df = pd.DataFrame({
                'battery_id': test_dataset.df['battery_id'].values,
                'cycle': test_cycle_nums,
                'true_q': test_labels_orig_q,
                'pred_q': test_preds_orig_q,
                'true_soh': test_labels_orig_s,
                'pred_soh': test_preds_orig_s
            })

            per_battery_metrics_list = []
            for batt_id in config.test_batteries:
                batt_df = eval_df[eval_df['battery_id'] == batt_id]
                if batt_df.empty: continue

                # --- 修改 4: 分别计算 Q 和 SOH 的指标 ---
                # 容量 (Q) 指标
                batt_true_q, batt_pred_q = batt_df['true_q'].values, batt_df['pred_q'].values
                batt_metrics_q = {
                    'Q_MAE': mean_absolute_error(batt_true_q, batt_pred_q),
                    'Q_MAPE': mean_absolute_percentage_error(batt_true_q, batt_pred_q),
                    'Q_MSE': mean_squared_error(batt_true_q, batt_pred_q),
                    'Q_RMSE': np.sqrt(mean_squared_error(batt_true_q, batt_pred_q)),
                    'Q_R2': r2_score(batt_true_q, batt_pred_q)
                }

                # SOH (S) 指标
                batt_true_s, batt_pred_s = batt_df['true_soh'].values, batt_df['pred_soh'].values
                batt_metrics_s = {
                    'SOH_MAE': mean_absolute_error(batt_true_s, batt_pred_s),
                    'SOH_MAPE': mean_absolute_percentage_error(batt_true_s, batt_pred_s),
                    'SOH_MSE': mean_squared_error(batt_true_s, batt_pred_s),
                    'SOH_RMSE': np.sqrt(mean_squared_error(batt_true_s, batt_pred_s)),
                    'SOH_R2': r2_score(batt_true_s, batt_pred_s)
                }

                # 合并字典
                batt_metrics_dict = {'Battery_ID': batt_id, **batt_metrics_q, **batt_metrics_s}
                per_battery_metrics_list.append(batt_metrics_dict)

                # 更新打印信息
                print(f"  - 电池 {batt_id}: Q_MAE={batt_metrics_q['Q_MAE']:.6f}, Q_R2={batt_metrics_q['Q_R2']:.4f} | "
                      f"SOH_MAE={batt_metrics_s['SOH_MAE']:.6f}, SOH_R2={batt_metrics_s['SOH_R2']:.4f}")

                all_runs_PER_BATTERY_metrics.append({**batt_metrics_dict, 'run': run_number, 'seed': current_seed})

                # --- 修改 5: 分别绘制 Q 和 SOH 的图表 ---
                # 绘制 Q (容量) 图表
                plot_results(batt_true_q, batt_pred_q,
                             f'Run {run_number} Battery {batt_id}: True vs Predicted Capacity',
                             os.path.join(run_save_path, f'test_plot_CAPACITY_battery_{batt_id}.png'))
                plot_diagonal_results(batt_true_q, batt_pred_q,
                                      f'Run {run_number} Battery {batt_id}: Capacity Diagonal Plot',
                                      os.path.join(run_save_path, f'test_diagonal_plot_CAPACITY_battery_{batt_id}.png'))

                # (需要您已添加了第1步中的新函数)
                # 绘制 SOH 图表
                plot_soh_results(batt_true_s, batt_pred_s, f'Run {run_number} Battery {batt_id}: True vs Predicted SOH',
                                 os.path.join(run_save_path, f'test_plot_SOH_battery_{batt_id}.png'))
                plot_soh_diagonal_results(batt_true_s, batt_pred_s,
                                          f'Run {run_number} Battery {batt_id}: SOH Diagonal Plot',
                                          os.path.join(run_save_path, f'test_diagonal_plot_SOH_battery_{batt_id}.png'))

            pd.DataFrame(per_battery_metrics_list).to_csv(os.path.join(run_save_path, 'test_per_battery_metrics.csv'),
                                                          index=False)

            print("\n--- 本轮评估结果 (所有测试电池汇总) ---")

            # --- 修改 6: 合并 Q 和 SOH 的总指标 ---
            # 直接使用 evaluate 返回的指标字典
            final_test_metrics = {**test_metrics_q, **test_metrics_s}

            ordered_metrics = [
                'Q_MAE', 'Q_MAPE', 'Q_MSE', 'Q_RMSE', 'Q_R2',
                'SOH_MAE', 'SOH_MAPE', 'SOH_MSE', 'SOH_RMSE', 'SOH_R2'
            ]

            pd.DataFrame([final_test_metrics]).to_csv(os.path.join(run_save_path, 'test_overall_metrics.csv'),
                                                      index=False)
            all_runs_metrics.append({'run': run_number, 'seed': current_seed, **final_test_metrics})

            # 更新打印信息
            print(f"测试集(汇总): Q_MAE={final_test_metrics['Q_MAE']:.6f}, Q_R2={final_test_metrics['Q_R2']:.4f} | "
                  f"SOH_MAE={final_test_metrics['SOH_MAE']:.6f}, SOH_R2={final_test_metrics['SOH_R2']:.4f}")

            if best_val_loss_this_run < best_run_val_loss:
                best_run_val_loss, best_run_dir, best_run_number = best_val_loss_this_run, run_save_path, run_number
                print(f"*** 新的最佳表现！验证集损失: {best_val_loss_this_run:.6f} ***")

        print(f"\n\n{'=' * 50}\n 所有实验均已完成。\n{'=' * 50}")
        if all_runs_metrics:
            summary_df = pd.DataFrame(all_runs_metrics)
            # === [修改] 保存五次实验汇总时，也强制列顺序 ===
            summary_cols = ['run', 'seed'] + ordered_metrics
            # 确保列存在
            summary_cols = [c for c in summary_cols if c in summary_df.columns]
            summary_df = summary_df[summary_cols]
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

            # --- 修改 7: 更新 all_runs_PER_BATTERY_metrics 的列排序 ---
            # 确保所有 Q 和 SOH 指标都被包含
            core_cols = ['Battery_ID', 'run', 'seed'] + ordered_metrics

            ordered_cols = [col for col in core_cols if col in per_batt_summary_df.columns] + [col for col in
                                                                                               per_batt_summary_df.columns
                                                                                               if col not in core_cols]
            per_batt_summary_df = per_batt_summary_df[ordered_cols].sort_values(by=['Battery_ID', 'run'])
            summary_path_per_batt = os.path.join(config.save_path, 'all_runs_per_battery_summary.csv')
            per_batt_summary_df.to_csv(summary_path_per_batt, index=False)
            print(f"“分电池”详细汇总报告已保存到: {summary_path_per_batt}")


if __name__ == '__main__':
    main()