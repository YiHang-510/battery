# dualnet_cyclenet_expnet_no_uw.py
# -*- coding: utf-8 -*-
"""
基于你的“联合预测-训练单步验证外推.py”改写：
- 去掉不确定性加权 uw 与 log_vars
- 使用手动损失权重（Config.w_*）
- 保留原有评估、绘图、RUL 计算接口
"""

import os
import random
import numpy as np
import pandas as pd
import warnings
import joblib
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================
# 1) 配置
# =============================
@dataclass
class Config:
    # 路径（按需修改）
    path_A_sequence: str = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
    path_C_features: str = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
    save_path: str = '/home/scuee_user06/myh/电池/result-累计放电容量/result-dualnet/6'

    # 数据集划分（按你的电池编号修改）
    # train_batteries: list = field(default_factory=lambda: [1, 2, 3, 4, 7, 8, 9, 11, 15, 17, 18, 19, 21, 22, 23, 24])
    # val_batteries:   list = field(default_factory=lambda: [5, 10, 13, 19])
    # test_batteries:  list = field(default_factory=lambda: [6, 12, 14, 20])

    train_batteries: list = field(default_factory=lambda: [1, 2, 3, 4])
    val_batteries:   list = field(default_factory=lambda: [5])
    test_batteries:  list = field(default_factory=lambda: [6])

    # train_batteries: list = field(default_factory=lambda: [7, 8, 9, 11])
    # val_batteries:   list = field(default_factory=lambda: [10])
    # test_batteries:  list = field(default_factory=lambda: [12])
    #
    # train_batteries: list = field(default_factory=lambda: [15, 17, 18, 19])
    # val_batteries:   list = field(default_factory=lambda: [13])
    # test_batteries:  list = field(default_factory=lambda: [14])
    #
    # train_batteries: list = field(default_factory=lambda: [21, 22, 23, 24])
    # val_batteries:   list = field(default_factory=lambda: [19])
    # test_batteries:  list = field(default_factory=lambda: [20])

    # 特征列（来自C路统计特征）
    features_from_C: list = field(default_factory=lambda: [
        '恒压充电时间(s)', '3.3~3.6V充电时间(s)'
    ])

    # 列名（请与你的CSV保持一致）
    col_cycle: str = '循环号'
    col_seq_prefix: str = '弛豫段电压'  # A 路的序列特征前缀
    col_C: str = '累计放电容量(Ah)'      # 累计放电容量
    col_Q: str = '最大容量(Ah)'          # 容量/可换成SOH列；若不存在则只做C的预测

    # 序列形状（按你的A路构造）
    sequence_length: int = 1
    sequence_feature_dim: int = 7  # 例如 7 点弛豫特征 -> (1,7)

    # CycleNet / 模型结构
    meta_cycle_len: int = 7
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.2

    # ExpNet
    exp_n_terms: int = 32

    # 训练
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 40
    seed: int = 2025

    # 训练策略
    use_gpu: bool = True
    device: torch.device = field(init=False)

    # 手动损失权重（根据你的目标自行调节）
    # 下面给一组偏重预测头的默认值；若更关注Q可把 w_pred_Qnext 再提高一些（如2.0~3.0）
    w_est_C: float = 0.3        # 估计当前 C_t
    w_est_nextseq: float = 0.05 # 估计 x_{t+1} 序列
    w_pred_Cnext: float = 1.0   # 预测 C_{t+1}
    w_pred_Qnext: float = 4.0   # 预测 Q_{t+1}（或SOH）

    # 训练预测网时是否用估计的 C_t（True）或真值 C_t（False，Teacher Forcing）
    pred_use_estimated_C: bool = True

    def __post_init__(self):
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        os.makedirs(self.save_path, exist_ok=True)


# =============================
# 2) 随机种子
# =============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================
# 3) 模型
# =============================
class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length: int = 1):
        gather_index = (index.view(-1,1) + torch.arange(length, device=index.device).view(1,-1)) % self.cycle_len
        return self.data[gather_index]


class EstimationModule(nn.Module):
    """
    输入: x_seq (B, L, F), x_scalar (B, S), cycle_number (B,)
    输出: z_t (B, d_model), C_hat_t (B,1), x_hat_{t+1}_flat (B, L*F)
    """
    def __init__(self, configs: Config):
        super().__init__()
        self.configs = configs
        in_seq = configs.sequence_length * configs.sequence_feature_dim
        in_sca = len(configs.features_from_C)

        self.sequence_encoder = nn.Linear(in_seq, configs.d_model // 2)
        self.scalar_encoder   = nn.Linear(in_sca, configs.d_model // 2)

        self.cycle_queue = RecurrentCycle(
            cycle_len=configs.meta_cycle_len,
            channel_size=configs.d_model
        )

        self.head_Ct = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

        self.head_nextseq = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, in_seq)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        B = x_seq.size(0)
        x_seq_flat = x_seq.view(B, -1)
        seq_emb = self.sequence_encoder(x_seq_flat)
        sca_emb = self.scalar_encoder(x_scalar)
        feat = torch.cat([seq_emb, sca_emb], dim=1)

        cyc_idx = cycle_number % self.configs.meta_cycle_len
        z_t = feat - self.cycle_queue(cyc_idx, length=1).squeeze(1)

        C_hat_t      = self.head_Ct(z_t)
        nextseq_flat = self.head_nextseq(z_t)
        return z_t, C_hat_t, nextseq_flat


class ExpNet(nn.Module):
    def __init__(self, n_terms=16, out_dims=2):
        super().__init__()
        self.n_terms = n_terms
        self.b = nn.Parameter(-torch.rand(n_terms) * 0.009 - 0.001)
        self.a = nn.Parameter(torch.rand(out_dims, n_terms))
        self.d = nn.Parameter(torch.rand(out_dims))
        self.out_dims = out_dims

    def forward(self, c):
        c = c.view(-1, 1)
        b = self.b.view(1, -1)
        phi = torch.exp(b * c)  # (B,K)
        out = torch.matmul(phi, self.a.t()) + self.d.view(1, -1)  # (B,O)
        return out


# =============================
# 4) 数据集与加载
# =============================
class BatteryPairDataset(Dataset):
    def __init__(self, df, config: Config, scalers, has_Q: bool):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.scalers = scalers
        self.has_Q = has_Q

        self.x_seq_t = np.array(self.df['voltage_sequence_t'].tolist(), dtype=np.float32)
        self.x_seq_t1 = np.array(self.df['voltage_sequence_t1'].tolist(), dtype=np.float32)
        self.x_scalar = self.df[config.features_from_C].values.astype(np.float32)
        self.cycle_t  = self.df[config.col_cycle].values.astype(np.int64)

        self.C_t  = self.df['C_t'].values.astype(np.float32)
        self.C_t1 = self.df['C_t1'].values.astype(np.float32)

        if self.has_Q:
            self.Q_t1 = self.df['Q_t1'].values.astype(np.float32)
        else:
            self.Q_t1 = np.zeros_like(self.C_t1, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_seq_t  = torch.from_numpy(self.x_seq_t[idx])
        x_seq_t1 = torch.from_numpy(self.x_seq_t1[idx])
        x_scalar = torch.from_numpy(self.x_scalar[idx])
        cycle_t  = torch.tensor(self.cycle_t[idx], dtype=torch.long)

        C_t  = torch.tensor(self.C_t[idx], dtype=torch.float32)
        C_t1 = torch.tensor(self.C_t1[idx], dtype=torch.float32)
        Q_t1 = torch.tensor(self.Q_t1[idx], dtype=torch.float32)

        return x_seq_t, x_scalar, cycle_t, C_t, x_seq_t1, C_t1, Q_t1


def _build_sequence_df_A(path_A, col_cycle, seq_feature_dim, seq_len, col_prefix):
    df_a = pd.read_csv(path_A, sep=',')
    feat_cols = [f'{col_prefix}{i}' for i in range(1, seq_feature_dim + 1)]
    seq_df = df_a.groupby(col_cycle)[feat_cols].apply(lambda x: x.values).reset_index(name='voltage_sequence')
    seq_df = seq_df[seq_df['voltage_sequence'].apply(len) == seq_len]
    return seq_df


def load_and_make_pairs(config: Config):
    all_ids = sorted(list(set(config.train_batteries + config.val_batteries + config.test_batteries)))
    all_df = []

    for bid in all_ids:
        path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{bid}.csv')
        path_c = os.path.join(config.path_C_features,  f'battery{bid}_SOH健康特征提取结果.csv')

        if not os.path.exists(path_a) or not os.path.exists(path_c):
            print(f"[WARN] 缺文件，跳过电池 {bid} | {os.path.basename(path_a)}, {os.path.basename(path_c)}")
            continue

        seq_df = _build_sequence_df_A(
            path_A=path_a,
            col_cycle=config.col_cycle,
            seq_feature_dim=config.sequence_feature_dim,
            seq_len=config.sequence_length,
            col_prefix=config.col_seq_prefix
        )
        stat_df = pd.read_csv(path_c, sep=',')
        stat_df.columns = [c.strip() for c in stat_df.columns]

        merged = pd.merge(seq_df, stat_df, on=config.col_cycle)
        merged['battery_id'] = bid
        merged = merged.sort_values([config.col_cycle]).reset_index(drop=True)

        merged['voltage_sequence_t']  = merged['voltage_sequence']
        merged['voltage_sequence_t1'] = merged['voltage_sequence'].shift(-1)
        merged['C_t']  = merged[config.col_C]
        merged['C_t1'] = merged[config.col_C].shift(-1)

        has_Q = config.col_Q in merged.columns
        if has_Q:
            merged['Q_t1'] = merged[config.col_Q].shift(-1)

        merged = merged.iloc[:-1].copy()
        all_df.append(merged)

    if len(all_df) == 0:
        raise RuntimeError("未加载到任何电池数据，请检查路径与文件名。")

    full_df = pd.concat(all_df, ignore_index=True)

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df   = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df  = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    scaler_seq    = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_C      = StandardScaler()

    has_Q = config.col_Q in full_df.columns
    scaler_Q      = StandardScaler() if has_Q else None

    all_train_seq_t = np.vstack(train_df['voltage_sequence_t'].values)
    scaler_seq.fit(all_train_seq_t)
    scaler_scalar.fit(train_df[config.features_from_C])
    scaler_C.fit(train_df[['C_t']].values)
    if has_Q:
        scaler_Q.fit(train_df[['Q_t1']].values)

    def _tf_seq(arr):
        return scaler_seq.transform(arr)

    for df in [train_df, val_df, test_df]:
        df['voltage_sequence_t']  = df['voltage_sequence_t'].apply(_tf_seq)
        df['voltage_sequence_t1'] = df['voltage_sequence_t1'].apply(_tf_seq)
        df.loc[:, config.features_from_C] = scaler_scalar.transform(df[config.features_from_C])
        df['C_t']  = scaler_C.transform(df[['C_t']].values)
        df['C_t1'] = scaler_C.transform(df[['C_t1']].values)
        if has_Q:
            df['Q_t1'] = scaler_Q.transform(df[['Q_t1']].values)

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar, 'C': scaler_C, 'Q': scaler_Q}
    return train_df, val_df, test_df, scalers, has_Q


# =============================
# 5) 训练 & 评估
# =============================

def train_one_epoch(E_net, P_net, loader, optimizer, cfg: Config, has_Q: bool):
    E_net.train(); P_net.train()
    mse = nn.MSELoss()
    total_loss = 0.0

    for (x_seq_t, x_scalar, cycle_t, C_t, x_seq_t1, C_t1, Q_t1) in loader:
        x_seq_t  = x_seq_t.to(cfg.device)
        x_seq_t1 = x_seq_t1.to(cfg.device)
        x_scalar = x_scalar.to(cfg.device)
        cycle_t  = cycle_t.to(cfg.device)
        C_t      = C_t.to(cfg.device).unsqueeze(-1)
        C_t1     = C_t1.to(cfg.device).unsqueeze(-1)
        Q_t1     = Q_t1.to(cfg.device).unsqueeze(-1)

        # ---- Estimation ----
        z_t, C_hat_t, nextseq_flat = E_net(x_seq_t, x_scalar, cycle_t)
        x_seq_t1_flat = x_seq_t1.view(x_seq_t1.size(0), -1)

        L_est_C   = mse(C_hat_t, C_t) * cfg.w_est_C
        L_est_seq = mse(nextseq_flat, x_seq_t1_flat) * cfg.w_est_nextseq

        # ---- Prediction ----
        C_input = C_hat_t if cfg.pred_use_estimated_C else C_t
        out = P_net(C_input.squeeze(-1))  # (B,O)

        if has_Q:
            C_next_hat = out[:, 0:1]
            Q_next_hat = out[:, 1:2]
            L_pred_C = mse(C_next_hat, C_t1) * cfg.w_pred_Cnext
            L_pred_Q = mse(Q_next_hat, Q_t1) * cfg.w_pred_Qnext
            loss = L_est_C + L_est_seq + L_pred_C + L_pred_Q
        else:
            C_next_hat = out[:, 0:1]
            L_pred_C = mse(C_next_hat, C_t1) * cfg.w_pred_Cnext
            loss = L_est_C + L_est_seq + L_pred_C

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(E_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(P_net.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(E_net, P_net, loader, cfg: Config, scalers, has_Q: bool):
    E_net.eval(); P_net.eval()
    mse = nn.MSELoss()

    all_Ct1_pred, all_Ct1_true = [], []
    all_Qt1_pred, all_Qt1_true = [], []
    all_Ct_pred,  all_Ct_true  = [], []

    total_loss = 0.0

    for (x_seq_t, x_scalar, cycle_t, C_t, x_seq_t1, C_t1, Q_t1) in loader:
        x_seq_t  = x_seq_t.to(cfg.device)
        x_seq_t1 = x_seq_t1.to(cfg.device)
        x_scalar = x_scalar.to(cfg.device)
        cycle_t  = cycle_t.to(cfg.device)
        C_t      = C_t.to(cfg.device).unsqueeze(-1)
        C_t1     = C_t1.to(cfg.device).unsqueeze(-1)
        Q_t1     = Q_t1.to(cfg.device).unsqueeze(-1)

        z_t, C_hat_t, nextseq_flat = E_net(x_seq_t, x_scalar, cycle_t)
        x_seq_t1_flat = x_seq_t1.view(x_seq_t1.size(0), -1)

        all_Ct_pred.append(C_hat_t.cpu().numpy())
        all_Ct_true.append(C_t.cpu().numpy())

        out = P_net(C_t.squeeze(-1))  # teacher forcing for eval
        if has_Q:
            C_next_hat = out[:, 0:1]
            Q_next_hat = out[:, 1:2]
            L = mse(C_hat_t, C_t) * cfg.w_est_C \
                + mse(nextseq_flat, x_seq_t1_flat) * cfg.w_est_nextseq \
                + mse(C_next_hat, C_t1) * cfg.w_pred_Cnext \
                + mse(Q_next_hat, Q_t1) * cfg.w_pred_Qnext

            all_Ct1_pred.append(C_next_hat.cpu().numpy())
            all_Ct1_true.append(C_t1.cpu().numpy())
            all_Qt1_pred.append(Q_next_hat.cpu().numpy())
            all_Qt1_true.append(Q_t1.cpu().numpy())
        else:
            C_next_hat = out[:, 0:1]
            L = mse(C_hat_t, C_t) * cfg.w_est_C \
                + mse(nextseq_flat, x_seq_t1_flat) * cfg.w_est_nextseq \
                + mse(C_next_hat, C_t1) * cfg.w_pred_Cnext

            all_Ct1_pred.append(C_next_hat.cpu().numpy())
            all_Ct1_true.append(C_t1.cpu().numpy())

        total_loss += L.item()

    Ct_pred  = np.concatenate(all_Ct_pred, axis=0).flatten()
    Ct_true  = np.concatenate(all_Ct_true, axis=0).flatten()
    Ct1_pred = np.concatenate(all_Ct1_pred, axis=0).flatten()
    Ct1_true = np.concatenate(all_Ct1_true, axis=0).flatten()

    Ct_pred_orig  = scalers['C'].inverse_transform(Ct_pred.reshape(-1,1)).flatten()
    Ct_true_orig  = scalers['C'].inverse_transform(Ct_true.reshape(-1,1)).flatten()
    Ct1_pred_orig = scalers['C'].inverse_transform(Ct1_pred.reshape(-1,1)).flatten()
    Ct1_true_orig = scalers['C'].inverse_transform(Ct1_true.reshape(-1,1)).flatten()

    metrics = {
        'EstC_t_MAE': mean_absolute_error(Ct_true_orig, Ct_pred_orig),
        'EstC_t_RMSE': np.sqrt(mean_squared_error(Ct_true_orig, Ct_pred_orig)),
        'EstC_t_R2': r2_score(Ct_true_orig, Ct_pred_orig),
        'PredC_t1_MAE': mean_absolute_error(Ct1_true_orig, Ct1_pred_orig),
        'PredC_t1_RMSE': np.sqrt(mean_squared_error(Ct1_true_orig, Ct1_pred_orig)),
        'PredC_t1_R2': r2_score(Ct1_true_orig, Ct1_pred_orig),
    }

    if has_Q:
        Qt1_pred = np.concatenate(all_Qt1_pred, axis=0).flatten()
        Qt1_true = np.concatenate(all_Qt1_true, axis=0).flatten()
        Qt1_pred_orig = scalers['Q'].inverse_transform(Qt1_pred.reshape(-1,1)).flatten()
        Qt1_true_orig = scalers['Q'].inverse_transform(Qt1_true.reshape(-1,1)).flatten()

        metrics.update({
            'PredQ_t1_MAE': mean_absolute_error(Qt1_true_orig, Qt1_pred_orig),
            'PredQ_t1_RMSE': np.sqrt(mean_squared_error(Qt1_true_orig, Qt1_pred_orig)),
            'PredQ_t1_R2': r2_score(Qt1_true_orig, Qt1_pred_orig)
        })

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, metrics


def plot_scatter(y_true, y_pred, title, path):
    plt.figure(figsize=(6,6))
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.scatter(y_true, y_pred, s=6, alpha=0.6)
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.title(title); plt.xlabel('True'); plt.ylabel('Pred')
    plt.axis('equal'); plt.xlim(mn, mx); plt.ylim(mn, mx)
    plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()


# ---- 时序对比绘图（teacher-forced 单步拼接） ----
@torch.no_grad()
def plot_series_teacher_forced_for_battery(df, cfg, scalers, P_net, battery_id, has_Q, title_suffix="teacher_forced"):
    sub = df[df['battery_id'] == battery_id].copy().sort_values(cfg.col_cycle).reset_index(drop=True)
    if len(sub) == 0:
        print(f"[WARN] 电池 {battery_id} 无数据"); return

    cycles_next = (sub[cfg.col_cycle].values + 1).astype(int)

    C_t_scaled  = sub['C_t'].values.astype(np.float32)
    C_t1_scaled = sub['C_t1'].values.astype(np.float32)

    device = cfg.device
    out = P_net(torch.tensor(C_t_scaled, device=device))
    C_pred_scaled = out[:, 0].detach().cpu().numpy()

    C_true = scalers['C'].inverse_transform(C_t1_scaled.reshape(-1,1)).flatten()
    C_pred = scalers['C'].inverse_transform(C_pred_scaled.reshape(-1,1)).flatten()

    plt.figure(figsize=(9,4))
    plt.plot(cycles_next, C_true, linewidth=1.5, label='True C_{t+1}')
    plt.plot(cycles_next, C_pred, linewidth=1.5, label='Pred C_{t+1}')
    plt.xlabel('Cycle'); plt.ylabel('Cumulative Discharge (Ah)')
    plt.title(f'Battery {battery_id} | C time series ({title_suffix})'); plt.grid(True); plt.legend()
    out_path = os.path.join(cfg.save_path, f"series_C_{title_suffix}_bat{battery_id}.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f"[SAVE] {out_path}")

    if has_Q:
        Q_t1_scaled = sub['Q_t1'].values.astype(np.float32)
        Q_pred_scaled = out[:, 1].detach().cpu().numpy()

        Q_true = scalers['Q'].inverse_transform(Q_t1_scaled.reshape(-1,1)).flatten()
        Q_pred = scalers['Q'].inverse_transform(Q_pred_scaled.reshape(-1,1)).flatten()

        plt.figure(figsize=(9,4))
        plt.plot(cycles_next, Q_true, linewidth=1.5, label='True Q_{t+1}')
        plt.plot(cycles_next, Q_pred, linewidth=1.5, label='Pred Q_{t+1}')
        plt.xlabel('Cycle'); plt.ylabel('Capacity / SOH')
        plt.title(f'Battery {battery_id} | Q/SOH time series ({title_suffix})'); plt.grid(True); plt.legend()
        out_path = os.path.join(cfg.save_path, f"series_Q_{title_suffix}_bat{battery_id}.png")
        plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
        print(f"[SAVE] {out_path}")


# =============================
# 6) 主流程
# =============================

def main():
    warnings.filterwarnings('ignore')
    cfg = Config()
    set_seed(cfg.seed)
    print(f"Save to: {cfg.save_path} | Device: {cfg.device}")

    train_df, val_df, test_df, scalers, has_Q = load_and_make_pairs(cfg)
    joblib.dump(scalers, os.path.join(cfg.save_path, 'scalers.pkl'))

    train_ds = BatteryPairDataset(train_df, cfg, scalers, has_Q)
    val_ds   = BatteryPairDataset(val_df, cfg, scalers, has_Q)
    test_ds  = BatteryPairDataset(test_df, cfg, scalers, has_Q)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    E_net = EstimationModule(cfg).to(cfg.device)
    P_net = ExpNet(n_terms=cfg.exp_n_terms, out_dims=2 if has_Q else 1).to(cfg.device)

    # 单一优化器（仅模型参数）
    opt_all = optim.Adam(
        list(E_net.parameters()) + list(P_net.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    best_val, no_improve = float('inf'), 0
    log_rows = []

    for ep in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(E_net, P_net, train_loader, opt_all, cfg, has_Q)
        val_loss, val_metrics = evaluate(E_net, P_net, val_loader, cfg, scalers, has_Q)

        row = {'epoch': ep, 'train_loss': tr_loss, 'val_loss': val_loss, **{f'val_{k}': v for k,v in val_metrics.items()}}
        log_rows.append(row)
        print(f"Epoch {ep:03d} | train {tr_loss:.6f} | val {val_loss:.6f} | "
              + " | ".join([f"{k}:{v:.4f}" for k,v in val_metrics.items()]))

        # 早停：以预测头的验证RMSE为主（含Q时叠加半权）
        val_key = val_metrics['PredC_t1_RMSE'] + (val_metrics.get('PredQ_t1_RMSE', 0.0) * 0.5)
        if val_key < best_val:
            best_val, no_improve = val_key, 0
            torch.save(E_net.state_dict(), os.path.join(cfg.save_path, 'E_best.pth'))
            torch.save(P_net.state_dict(), os.path.join(cfg.save_path, 'P_best.pth'))
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop at epoch {ep} (val_key={val_key:.6f}, best={best_val:.6f}).")
                break

    pd.DataFrame(log_rows).to_csv(os.path.join(cfg.save_path, 'train_log.csv'), index=False)

    E_net.load_state_dict(torch.load(os.path.join(cfg.save_path, 'E_best.pth'), map_location=cfg.device))
    P_net.load_state_dict(torch.load(os.path.join(cfg.save_path, 'P_best.pth'), map_location=cfg.device))

    test_loss, test_metrics = evaluate(E_net, P_net, test_loader, cfg, scalers, has_Q)
    print("\n== Final Test ==")
    for k,v in test_metrics.items():
        print(f"{k}: {v:.6f}")
    pd.DataFrame([test_metrics]).to_csv(os.path.join(cfg.save_path, 'test_metrics.csv'), index=False)

    for bid in sorted(test_df['battery_id'].unique()):
        plot_series_teacher_forced_for_battery(test_df, cfg, scalers, P_net, bid, has_Q, title_suffix="teacher_forced")

    @torch.no_grad()
    def gather_preds(loader):
        E_net.eval(); P_net.eval()
        C_true, C_pred, Q_true, Q_pred = [], [], [], []
        for (x_seq_t, x_scalar, cycle_t, C_t, x_seq_t1, C_t1, Q_t1) in loader:
            x_seq_t  = x_seq_t.to(cfg.device)
            x_scalar = x_scalar.to(cfg.device)
            cycle_t  = cycle_t.to(cfg.device)
            C_t      = C_t.to(cfg.device).unsqueeze(-1)
            C_t1     = C_t1.to(cfg.device).unsqueeze(-1)
            Q_t1     = Q_t1.to(cfg.device).unsqueeze(-1)

            out = P_net(C_t.squeeze(-1))
            C_next_hat = out[:,0:1]

            C_next_hat = scalers['C'].inverse_transform(C_next_hat.cpu().numpy()).flatten()
            C_t1_true  = scalers['C'].inverse_transform(C_t1.cpu().numpy()).flatten()
            C_pred.append(C_next_hat); C_true.append(C_t1_true)

            if has_Q:
                Q_next_hat = out[:,1:2]
                Q_next_hat = scalers['Q'].inverse_transform(Q_next_hat.cpu().numpy()).flatten()
                Q_t1_true  = scalers['Q'].inverse_transform(Q_t1.cpu().numpy()).flatten()
                Q_pred.append(Q_next_hat); Q_true.append(Q_t1_true)

        C_true = np.concatenate(C_true); C_pred = np.concatenate(C_pred)
        if has_Q:
            Q_true = np.concatenate(Q_true); Q_pred = np.concatenate(Q_pred)
        else:
            Q_true, Q_pred = None, None
        return C_true, C_pred, Q_true, Q_pred

    C_true, C_pred, Q_true, Q_pred = gather_preds(test_loader)
    plot_scatter(C_true, C_pred, 'Accumulated discharge capacity{t+1} True vs Pred', os.path.join(cfg.save_path, 'Accumulated discharge capacity.png'))
    if has_Q:
        plot_scatter(Q_true, Q_pred, 'SOH{t+1} True vs Pred', os.path.join(cfg.save_path, 'SOH.png'))

    print(f"\nArtifacts saved in: {cfg.save_path}")


# =============================
# 7) RUL 估计（按 Q/SOH 触线）
# =============================
@torch.no_grad()
def rul_by_Q(df, cfg, scalers, E_net, P_net, battery_id,
             start_idx=None, soh_thr=0.8, q0=None, max_h=5000,
             use_estCt=True, enforce_monotonic=True, fractional=True):
    if getattr(P_net, "out_dims", 1) < 2 or scalers.get('Q', None) is None:
        return None, None, {}

    device = cfg.device
    sub = df[df['battery_id']==battery_id].sort_values(cfg.col_cycle).reset_index(drop=True)
    if len(sub)==0:
        return None, None, {}

    if start_idx is None:
        start_idx = len(sub) - 1

    cyc0 = int(sub.loc[start_idx, cfg.col_cycle])

    if use_estCt:
        x_seq_t  = torch.from_numpy(sub.loc[start_idx,'voltage_sequence_t']).float().unsqueeze(0).to(device)
        x_scalar = torch.from_numpy(sub.loc[start_idx, cfg.features_from_C].astype(np.float32).values).unsqueeze(0).to(device)
        cycle_t  = torch.tensor([cyc0], dtype=torch.long, device=device)
        _, C_t, _ = E_net(x_seq_t, x_scalar, cycle_t)
    else:
        C_t = torch.tensor([[sub.loc[start_idx,'C_t']]], dtype=torch.float32, device=device)

    if q0 is None:
        if 'Q_t1' in sub.columns and sub['Q_t1'].notna().any():
            q0 = scalers['Q'].inverse_transform(np.array([[sub.loc[0,'Q_t1']]], dtype=np.float32))[0,0]
        else:
            q0 = 1.0
    q_thresh = q0 * soh_thr

    cycles, Q_preds, C_preds = [], [], []
    crossed_idx = None

    for h in range(1, max_h+1):
        out = P_net(C_t.squeeze(-1))
        C_next = out[:, 0:1]
        Q_next_scaled = out[:, 1:2].cpu().numpy()
        Q_next = scalers['Q'].inverse_transform(Q_next_scaled)[0,0]

        C_preds.append(C_next.item())
        Q_preds.append(Q_next)
        cycles.append(cyc0 + h)

        C_t = C_next
        if Q_next <= q_thresh:
            crossed_idx = h
            break

    if enforce_monotonic and len(Q_preds) > 0:
        Q_preds = np.minimum.accumulate(np.array(Q_preds, dtype=float))
        C_preds = np.maximum.accumulate(np.array(C_preds, dtype=float))
    else:
        Q_preds = np.array(Q_preds, dtype=float)
        C_preds = np.array(C_preds, dtype=float)

    RUL = None
    if crossed_idx is not None:
        if fractional and crossed_idx >= 2:
            q_hi = Q_preds[crossed_idx-2]
            q_lo = Q_preds[crossed_idx-1]
            if q_hi > q_lo:
                frac = (q_hi - q_thresh) / (q_hi - q_lo)
                RUL = (crossed_idx-1) + float(np.clip(frac, 0.0, 1.0))
            else:
                RUL = float(crossed_idx)
        else:
            RUL = float(crossed_idx)

    series = {
        'cycles': np.array(cycles, dtype=int),
        'Q_pred': Q_preds,
        'C_pred': scalers['C'].inverse_transform(C_preds.reshape(-1,1)).flatten(),
        'Q_thresh': q_thresh
    }
    return RUL, (cyc0 + (RUL if RUL is not None else np.nan)), series


def plot_rul_series_Q(series, battery_id, cfg, title_suffix="rul_Q"):
    if not series:
        return
    cycles = series['cycles']
    Q_pred = series['Q_pred']
    Q_thr  = series['Q_thresh']
    C_pred = series['C_pred']

    plt.figure(figsize=(9,4))
    plt.plot(cycles, Q_pred, marker='o', linewidth=1.5, label='Pred Q/SOH')
    plt.axhline(Q_thr, linestyle='--', label=f'Threshold={Q_thr:.3f}')
    plt.xlabel('Cycle'); plt.ylabel('Capacity / SOH')
    plt.title(f'Battery {battery_id} | RUL-by-Q ({title_suffix})')
    plt.grid(True); plt.legend(); plt.tight_layout()
    out1 = os.path.join(cfg.save_path, f"RUL_Q_bat{battery_id}_{title_suffix}.png")
    plt.savefig(out1, dpi=300); plt.close()
    print(f"[SAVE] {out1}")

    plt.figure(figsize=(9,4))
    plt.plot(cycles, C_pred, marker='s', linewidth=1.5, label='Pred C (iter)')
    plt.xlabel('Cycle'); plt.ylabel('Cumulative Discharge (Ah)')
    plt.title(f'Battery {battery_id} | C trajectory while RUL-by-Q')
    plt.grid(True); plt.legend(); plt.tight_layout()
    out2 = os.path.join(cfg.save_path, f"C_traj_during_RUL_Q_bat{battery_id}_{title_suffix}.png")
    plt.savefig(out2, dpi=300); plt.close()
    print(f"[SAVE] {out2}")


if __name__ == "__main__":
    main()
