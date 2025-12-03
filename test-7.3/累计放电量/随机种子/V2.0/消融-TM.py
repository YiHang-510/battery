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


# --- 1. 配置参数 (已修改支持消融实验) ---
class Config:
    def __init__(self):
        # =========================================================
        # [核心修改] 消融实验模式开关
        # 可选值:
        #   'Q'    : 单独训练/评估 累计放电容量 (Capacity)
        #   'SOH'  : 单独训练/评估 健康状态 (SOH)
        #   'BOTH' : 联合训练 (原模式)
        # =========================================================
        self.ablation_mode = 'Q'  # <--- 在这里修改模式！！！

        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'

        # 根据模式自动调整保存路径后缀
        base_save_path = '/home/scuee_user06/myh/电池/result-累计放电容量V2.0_correct/TM_PIRes'
        if self.ablation_mode == 'Q':
            self.save_path = os.path.join(base_save_path, 'Ablation_Only_Q/cc')
        elif self.ablation_mode == 'SOH':
            self.save_path = os.path.join(base_save_path, 'Ablation_Only_SOH/vv')
        else:
            self.save_path = os.path.join(base_save_path, 'Joint_Both')

        self.train_batteries = [1, 2, 9, 10]
        self.val_batteries = [17]
        self.test_batteries = [18]

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
        self.sequence_feature_dim = 7
        self.sequence_length = 1

        # --- 模型超参数 ---
        self.d_model = 128
        self.n_blocks = 3
        self.kernel_sizes = (3, 5, 7)
        self.dropout = 0.1
        self.n_basis = 10
        self.degree = 3
        self.n_grid = 512
        self.residual_l2 = 1e-3

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.patience = 10
        self.seed = 2025
        self.mode = 'both'

        # 任务权重 (将在 train_epoch 中根据 ablation_mode 动态覆盖)
        self.task_weights = {"q": 1.0, "soh": 10000000}

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.cycle_norm_min = None
        self.cycle_norm_max = None
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


# --- 3. 模型定义 (保持不变，结构复用) ---
# ... (_bspline_basis_grid, ISplineBasis, DepthwiseTemporalMix, ChannelMix, TimeMixerBlock, TimeMixerStack 保持不变) ...

def _bspline_basis_grid(knots: torch.Tensor, degree: int, grid: torch.Tensor) -> torch.Tensor:
    device = knots.device
    dtype = knots.dtype
    G = grid.numel()
    M = knots.numel() - degree - 1
    assert M > 0
    Bk = torch.zeros(G, M, device=device, dtype=dtype)
    for i in range(M):
        left, right = knots[i], knots[i + 1]
        cond = (grid >= left) & (grid < right)
        if i == M - 1: cond = cond | (grid == right)
        Bk[:, i] = cond.to(dtype)
    for k in range(1, degree + 1):
        Bk_next = torch.zeros_like(Bk)
        for i in range(M):
            term1 = 0.0
            denom1 = knots[i + k] - knots[i]
            if denom1 > 0: term1 = ((grid - knots[i]) / denom1) * Bk[:, i]
            term2 = 0.0
            if (i + 1) < M:
                denom2 = knots[i + k + 1] - knots[i + 1]
                if denom2 > 0: term2 = ((knots[i + k + 1] - grid) / denom2) * Bk[:, i + 1]
            Bk_next[:, i] = term1 + term2
        Bk = Bk_next
    return Bk


class ISplineBasis(nn.Module):
    def __init__(self, n_basis=10, degree=3, n_grid=512, device=None, dtype=None):
        super().__init__()
        self.n_basis, self.degree, self.n_grid = n_basis, degree, n_grid
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        n_knots = n_basis + degree + 1
        interior = max(0, n_knots - 2 * (degree + 1))
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

    def eval(self, t):
        t = t.view(-1, 1)
        u = (t * (self.n_grid - 1)).clamp(0, self.n_grid - 1)
        i0, i1 = torch.floor(u).long(), (torch.floor(u).long() + 1).clamp(max=self.n_grid - 1)
        w = u - i0.float()
        return (1.0 - w) * self.I_grid[i0.squeeze(-1), :] + w * self.I_grid[i1.squeeze(-1), :]


class DepthwiseTemporalMix(nn.Module):
    def __init__(self, d_model, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model) for k in kernel_sizes])
        self.proj = nn.Conv1d(d_model, d_model, 1)
        self.alpha = nn.Parameter(torch.zeros(len(kernel_sizes)))
        self.act = nn.SiLU();
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        feats = [self.act(conv(x_t)) for conv in self.convs]
        a = F.softmax(self.alpha, dim=0)
        y = sum(a[i] * feats[i] for i in range(len(feats)))
        return self.drop(self.proj(y)).transpose(1, 2)


class ChannelMix(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * expansion), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout))

    def forward(self, x): return self.ff(x)


class TimeMixerBlock(nn.Module):
    def __init__(self, d_model, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model);
        self.tmix = DepthwiseTemporalMix(d_model, kernel_sizes, dropout)
        self.norm2 = nn.LayerNorm(d_model);
        self.cmix = ChannelMix(d_model, expansion=4, dropout=dropout)

    def forward(self, x): return x + self.cmix(self.norm2(x + self.tmix(self.norm1(x))))


class TimeMixerStack(nn.Module):
    def __init__(self, d_model, n_blocks, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TimeMixerBlock(d_model, kernel_sizes, dropout) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return x


@dataclass
class TMPIResConfig:
    d_model: int = 128;
    n_blocks: int = 3;
    kernel_sizes: Tuple[int, ...] = (3, 5, 7)
    dropout: float = 0.1;
    n_basis: int = 10;
    degree: int = 3;
    n_grid: int = 512;
    residual_l2: float = 1e-4


class TM_PIRes(nn.Module):
    def __init__(self, seq_channels: int, scalar_dim: int, cfg: TMPIResConfig, n_terms: int = 16):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Linear(seq_channels, d)
        self.tm = TimeMixerStack(d, cfg.n_blocks, cfg.kernel_sizes, cfg.dropout)
        self.enc_s = nn.Sequential(nn.Linear(scalar_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d))

        # Q Head
        self.basis = ISplineBasis(cfg.n_basis, cfg.degree, cfg.n_grid)
        self.c0 = nn.Parameter(torch.randn(cfg.n_basis))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.res_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_basis))

        # SOH Head
        self.n_terms = n_terms
        self.soh_param_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_terms * 3))

    def forward(self, v, s, t_norm):
        x = self.embed(v)
        H = self.tm(x)
        h = H.mean(dim=1) + self.enc_s(s)

        # Q Prediction
        B_I = self.basis.eval(t_norm)
        m = (B_I @ F.softplus(self.c0)) + self.b0
        c_h = F.softplus(self.res_head(h))
        Q = m + (B_I * c_h).sum(dim=-1)

        # SOH Prediction
        params = self.soh_param_head(h)
        a, b, d_bias = torch.split(params, self.n_terms, dim=1)
        a = torch.sigmoid(a)
        b = -F.softplus(b) - 1e-6
        d_bias = torch.sigmoid(d_bias) * 0.5
        t = t_norm.view(-1, 1)
        SOH_pred = torch.sigmoid((a * torch.exp(b * t) + d_bias).sum(dim=1))

        return (Q, SOH_pred), {"c_h": c_h}


# --- 4. 数据集 (不变) ---
class BatteryMultimodalDataset(Dataset):
    def __init__(self, dataframe, sequence_col, scalar_cols, target_q_col, target_soh_col):
        self.df = dataframe.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.scalar_cols = scalar_cols
        self.targets_q = self.df[target_q_col].values.astype(np.float32)
        self.targets_s = self.df[target_soh_col].values.astype(np.float32)
        self.sequences = np.array(self.df[self.sequence_col].tolist(), dtype=np.float32)
        self.scalars = self.df[self.scalar_cols].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.sequences[idx]), torch.from_numpy(self.scalars[idx]),
                torch.tensor(self.cycle_indices[idx], dtype=torch.long),
                torch.tensor(self.targets_q[idx], dtype=torch.float32),
                torch.tensor(self.targets_s[idx], dtype=torch.float32))


def load_and_preprocess_data(config):
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries
    for battery_id in sorted(list(set(all_ids))):
        try:
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')
            df_a = pd.read_csv(path_a, sep=',');
            df_c = pd.read_csv(path_c, sep=',')
            df_c.rename(columns=lambda x: x.strip(), inplace=True)
            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.sequence_length]
            final_df = pd.merge(sequence_df, df_c, on='循环号')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)
        except Exception as e:
            print(f"Error battery {battery_id}: {e}"); continue

    full_df = pd.concat(all_battery_data, ignore_index=True)
    full_df = full_df.sort_values(["battery_id", "循环号"])
    first_cap = full_df.groupby("battery_id")["最大容量(Ah)"].transform("first")
    full_df["SOH"] = (full_df["最大容量(Ah)"] / first_cap).clip(lower=0.0, upper=1.0)

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    config.cycle_norm_min = float(train_df['循环号'].min())
    config.cycle_norm_max = float(train_df['循环号'].max())
    if config.cycle_norm_max <= config.cycle_norm_min: config.cycle_norm_max = config.cycle_norm_min + 1.0

    scaler_seq = StandardScaler();
    scaler_scalar = StandardScaler()
    scaler_seq.fit(np.vstack(train_df['voltage_sequence'].values))
    scaler_scalar.fit(train_df[config.features_from_C])

    for df in [train_df, val_df, test_df]:
        df['voltage_sequence'] = df['voltage_sequence'].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, config.features_from_C] = scaler_scalar.transform(df[config.features_from_C])

    return (BatteryMultimodalDataset(train_df, 'voltage_sequence', config.features_from_C, '累计放电容量(Ah)', 'SOH'),
            BatteryMultimodalDataset(val_df, 'voltage_sequence', config.features_from_C, '累计放电容量(Ah)', 'SOH'),
            BatteryMultimodalDataset(test_df, 'voltage_sequence', config.features_from_C, '累计放电容量(Ah)', 'SOH'),
            {'sequence': scaler_seq, 'scalar': scaler_scalar})


# --- 5. 训练函数 (修改版) ---
def train_epoch(model, dataloader, optimizer, criterion, config):
    model.train()
    use_amp = getattr(config, "use_amp", False)
    scaler = GradScaler(enabled=(use_amp and config.device.type == 'cuda'))
    total_loss = 0.0

    # --- 根据消融模式设置权重 ---
    # 如果是单任务模式，强制将另一任务权重归零
    w_q = config.task_weights["q"] if config.ablation_mode in ['Q', 'BOTH'] else 0.0
    w_soh = config.task_weights["soh"] if config.ablation_mode in ['SOH', 'BOTH'] else 0.0

    for batch in dataloader:
        seq, scalar, cyc, y_q, y_s = [b.to(config.device) for b in batch]
        t_norm = torch.clamp(
            (cyc.float() - config.cycle_norm_min) / (config.cycle_norm_max - config.cycle_norm_min + 1e-12), 0.0, 1.0)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            (pred_q, pred_s), aux = model(seq, scalar, t_norm)

            # 计算 Q Loss
            loss_q = criterion(pred_q, y_q) if w_q > 0 else torch.tensor(0.0, device=config.device)

            # 计算 SOH Loss
            loss_soh = criterion(pred_s, y_s) if w_soh > 0 else torch.tensor(0.0, device=config.device)

            # 计算正则化 Loss (仅当训练 Q 时才需要 PI-Res 的正则化)
            if w_q > 0:
                loss_reg = config.residual_l2 * (aux.get("c_h", 0.0) ** 2).mean()
            else:
                loss_reg = torch.tensor(0.0, device=config.device)

            # 总损失
            loss = w_q * loss_q + w_soh * loss_soh + loss_reg

        if use_amp and config.device.type == 'cuda':
            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update()
        else:
            loss.backward();
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 6. 评估函数 (修改版) ---
def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_q_preds, all_q_labels = [], []
    all_s_preds, all_s_labels = [], []
    all_cycle_indices = []

    w_q = config.task_weights["q"] if config.ablation_mode in ['Q', 'BOTH'] else 0.0
    w_soh = config.task_weights["soh"] if config.ablation_mode in ['SOH', 'BOTH'] else 0.0

    with torch.no_grad():
        for batch in dataloader:
            seq, scalar, cyc, y_q, y_s = [b.to(device) for b in batch]
            t_norm = torch.clamp(
                (cyc.float() - config.cycle_norm_min) / (config.cycle_norm_max - config.cycle_norm_min + 1e-12), 0.0,
                1.0)

            (pred_q, pred_s), aux = model(seq, scalar, t_norm)

            loss_val = 0.0
            if w_q > 0:
                loss_val += w_q * criterion(pred_q, y_q) + config.residual_l2 * (aux.get("c_h", 0.0) ** 2).mean()
                all_q_preds.append(pred_q.cpu().numpy());
                all_q_labels.append(y_q.cpu().numpy())

            if w_soh > 0:
                loss_val += w_soh * criterion(pred_s, y_s)
                all_s_preds.append(pred_s.cpu().numpy());
                all_s_labels.append(y_s.cpu().numpy())

            total_loss += loss_val.item()
            all_cycle_indices.append(cyc.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    cycle_indices = np.concatenate(all_cycle_indices).reshape(-1)

    # --- 计算 Q 指标 (含 MAPE) ---
    metrics_q, q_preds, q_lbls = {}, None, None
    if w_q > 0 and all_q_preds:
        q_preds = np.concatenate(all_q_preds).reshape(-1)
        q_lbls = np.concatenate(all_q_labels).reshape(-1)
        metrics_q = {
            "Q_MAE": mean_absolute_error(q_lbls, q_preds),
            "Q_MAPE": mean_absolute_percentage_error(q_lbls, q_preds),  # 新增
            "Q_MSE": mean_squared_error(q_lbls, q_preds),
            "Q_RMSE": np.sqrt(mean_squared_error(q_lbls, q_preds)),
            "Q_R2": r2_score(q_lbls, q_preds)
        }

    # --- 计算 SOH 指标 (含 MAPE) ---
    metrics_s, s_preds, s_lbls = {}, None, None
    if w_soh > 0 and all_s_preds:
        s_preds = np.concatenate(all_s_preds).reshape(-1)
        s_lbls = np.concatenate(all_s_labels).reshape(-1)
        metrics_s = {
            "SOH_MAE": mean_absolute_error(s_lbls, s_preds),
            "SOH_MAPE": mean_absolute_percentage_error(s_lbls, s_preds),  # 新增
            "SOH_MSE": mean_squared_error(s_lbls, s_preds),
            "SOH_RMSE": np.sqrt(mean_squared_error(s_lbls, s_preds)),
            "SOH_R2": r2_score(s_lbls, s_preds)
        }

    return avg_loss, metrics_q, metrics_s, q_preds, q_lbls, cycle_indices, s_preds, s_lbls


# --- 7. 绘图工具 ---
def plot_results(labels, preds, title, save_path, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Pred', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title);
    plt.xlabel('Sample');
    plt.ylabel(ylabel);
    plt.legend();
    plt.grid(True)
    plt.savefig(save_path);
    plt.close()


def plot_diagonal(labels, preds, title, save_path, xlabel, ylabel):
    plt.figure(figsize=(8, 8))
    min_v, max_v = min(labels.min(), preds.min()) * 0.98, max(labels.max(), preds.max()) * 1.02
    plt.scatter(labels, preds, alpha=0.6);
    plt.plot([min_v, max_v], [min_v, max_v], 'r--')
    plt.title(title);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.grid(True);
    plt.axis('equal')
    plt.savefig(save_path);
    plt.close()


# --- 8. 主函数 ---
def main():
    warnings.filterwarnings('ignore');
    matplotlib.use('Agg')
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"当前消融模式: {config.ablation_mode}")
    print(f"保存路径: {config.save_path}")

    num_runs = 5
    all_metrics = []

    for run in range(1, num_runs + 1):
        set_seed(2025 + run * 100)
        run_dir = os.path.join(config.save_path, f'run_{run}')
        os.makedirs(run_dir, exist_ok=True)

        try:
            train_ds, val_ds, test_ds, scalers = load_and_preprocess_data(config)
        except Exception as e:
            print(f"Data Error: {e}"); continue

        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

        tm_cfg = TMPIResConfig(d_model=config.d_model, n_blocks=config.n_blocks, dropout=config.dropout,
                               n_basis=config.n_basis, degree=config.degree, n_grid=config.n_grid,
                               residual_l2=config.residual_l2)
        model = TM_PIRes(config.sequence_feature_dim, config.scalar_feature_dim, tm_cfg).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()

        best_loss = float('inf');
        patience = 0

        print(f"\nRun {run}/{num_runs} Start...")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_dl, optimizer, criterion, config)
            val_loss, val_q, val_s, _, _, _, _, _ = evaluate(model, val_dl, criterion, config.device, config)

            # 打印简略日志
            log_str = f"Ep {epoch + 1} | Loss: {train_loss:.4f} | Val: {val_loss:.4f}"
            if config.ablation_mode in ['Q', 'BOTH']: log_str += f" | Q_R2: {val_q.get('Q_R2', 0):.4f}"
            if config.ablation_mode in ['SOH', 'BOTH']: log_str += f" | SOH_R2: {val_s.get('SOH_R2', 0):.4f}"
            print(log_str)

            if val_loss < best_loss:
                best_loss = val_loss;
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
                patience = 0
            else:
                patience += 1
                if patience >= config.patience: break

        # Test
        model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
        _, t_mq, t_ms, pred_q, true_q, _, pred_s, true_s = evaluate(model, test_dl, criterion, config.device, config)

        # 汇总结果 & 绘图
        final_metrics = {'run': run}
        if config.ablation_mode in ['Q', 'BOTH']:
            final_metrics.update(t_mq)  # 包含 Q_MAE, Q_MAPE, Q_MSE, Q_RMSE, Q_R2
            # 绘图 Q
            for bid in config.test_batteries:
                mask = (test_ds.df['battery_id'] == bid).values
                if sum(mask) > 0:
                    p_q = pred_q[mask];
                    t_q = true_q[mask]
                    plot_results(t_q, p_q, f"Batt {bid} Capacity", os.path.join(run_dir, f"Q_Batt{bid}.png"),
                                 "Capacity (Ah)")
                    plot_diagonal(t_q, p_q, f"Batt {bid} Q Diag", os.path.join(run_dir, f"Q_Diag_Batt{bid}.png"),
                                  "True Q", "Pred Q")

        if config.ablation_mode in ['SOH', 'BOTH']:
            final_metrics.update(t_ms)  # 包含 SOH_MAE, SOH_MAPE, SOH_MSE, SOH_RMSE, SOH_R2
            # 绘图 SOH
            for bid in config.test_batteries:
                mask = (test_ds.df['battery_id'] == bid).values
                if sum(mask) > 0:
                    p_s = np.clip(pred_s[mask], 0, 1);
                    t_s = true_s[mask]
                    plot_results(t_s, p_s, f"Batt {bid} SOH", os.path.join(run_dir, f"SOH_Batt{bid}.png"), "SOH")
                    plot_diagonal(t_s, p_s, f"Batt {bid} SOH Diag", os.path.join(run_dir, f"SOH_Diag_Batt{bid}.png"),
                                  "True SOH", "Pred SOH")

        all_metrics.append(final_metrics)
        print(f"Run {run} Result: {final_metrics}")

    res_df = pd.DataFrame(all_metrics)
    res_df.to_csv(os.path.join(config.save_path, 'final_summary.csv'), index=False)
    print("\nFinal Summary (Top 5 Rows):\n", res_df.head())


if __name__ == '__main__':
    main()