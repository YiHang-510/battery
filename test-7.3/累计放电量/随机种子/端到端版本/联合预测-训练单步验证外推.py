# dualnet_cyclenet_expnet.py
# -*- coding: utf-8 -*-

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

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
    save_path: str = '/home/scuee_user06/myh/电池/result-累计放电容量/result-dualnet/all'

    # 数据集划分（按你的电池编号修改）
    train_batteries: list = field(default_factory=lambda: [1, 2, 3, 4, 7, 8, 9, 11, 15, 17, 18, 19, 21, 22, 23, 24])
    val_batteries:   list = field(default_factory=lambda: [5, 10, 13, 19])
    test_batteries:  list = field(default_factory=lambda: [6, 12, 14, 20])

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
    exp_n_terms: int = 16

    # 训练
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    seed: int = 2025

    # 训练策略
    use_gpu: bool = True
    device: torch.device = field(init=False)

    # 估计网与预测网的损失权重
    w_est_C: float = 1.0       # 估计当前 C_t
    w_est_nextseq: float = 0.2 # 估计 x_{t+1} 序列的损失权重
    w_pred_Cnext: float = 1.0  # 预测 C_{t+1}
    w_pred_Qnext: float = 1.0  # 预测 Q_{t+1}（若有）

    # 训练预测网时是否用估计的 C_t（True）或真值 C_t（False，Teacher Forcing）
    pred_use_estimated_C: bool = False

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
        # index: [B], length: 1
        gather_index = (index.view(-1,1) + torch.arange(length, device=index.device).view(1,-1)) % self.cycle_len
        return self.data[gather_index]  # [B, length, channel_size]


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

        # 估计当前 C_t
        self.head_Ct = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

        # 估计下一步序列特征 x_{t+1}（与输入序列形状一致，输出扁平向量）
        self.head_nextseq = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, in_seq)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        B = x_seq.size(0)
        x_seq_flat = x_seq.view(B, -1)                         # (B, L*F)
        seq_emb = self.sequence_encoder(x_seq_flat)            # (B, d/2)
        sca_emb = self.scalar_encoder(x_scalar)                # (B, d/2)
        feat = torch.cat([seq_emb, sca_emb], dim=1)            # (B, d)

        # 去周期
        cyc_idx = cycle_number % self.configs.meta_cycle_len
        z_t = feat - self.cycle_queue(cyc_idx, length=1).squeeze(1)  # (B, d)

        C_hat_t      = self.head_Ct(z_t)                       # (B,1)
        nextseq_flat = self.head_nextseq(z_t)                  # (B, L*F)
        return z_t, C_hat_t, nextseq_flat


class ExpNet(nn.Module):
    """
    输入: c_t (B,1)，输出: [C_{t+1}, Q_{t+1}] 或 [C_{t+1}]（由 out_dims 决定）
    使用共享的指数基函数 φ_k(c)=exp(b_k * c)
    """
    def __init__(self, n_terms=16, out_dims=2):
        super().__init__()
        self.n_terms = n_terms
        # b < 0 保证随累计容量增长时的衰减/缓变（可按需调整先验）
        self.b = nn.Parameter(-torch.rand(n_terms) * 0.009 - 0.001)  # (K,)
        self.a = nn.Parameter(torch.rand(out_dims, n_terms))         # (O,K)
        self.d = nn.Parameter(torch.rand(out_dims))                  # (O,)

        self.out_dims = out_dims

    def forward(self, c):
        # c: (B,) or (B,1)
        c = c.view(-1, 1)                  # (B,1)
        b = self.b.view(1, -1)             # (1,K)
        phi = torch.exp(b * c)             # (B,K)

        out = torch.matmul(phi, self.a.t()) + self.d.view(1, -1)  # (B,O)
        return out  # [:,0]=C_{t+1}, [:,1]=Q_{t+1} (若存在)


# =============================
# 4) 数据集与加载
# =============================
class BatteryPairDataset(Dataset):
    """
    构造 (t, t+1) 成对样本：
    - 输入: x_seq_t, x_scalar_t, cycle_t
    - 监督: C_t, x_seq_{t+1}, C_{t+1}, (可选) Q_{t+1}
    说明：
      1) 所有数值均已缩放（StandardScaler）
      2) 若无 Q 列，则返回的 y_Q_t1 为 None，占位 float32(0)
    """
    def __init__(self, df, config: Config, scalers, has_Q: bool):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.scalers = scalers
        self.has_Q = has_Q

        # 预取numpy，避免getitem重复开销
        self.x_seq_t = np.array(self.df['voltage_sequence_t'].tolist(), dtype=np.float32)  # (N,L,F)
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
        x_seq_t  = torch.from_numpy(self.x_seq_t[idx])       # (L,F)
        x_seq_t1 = torch.from_numpy(self.x_seq_t1[idx])      # (L,F)
        x_scalar = torch.from_numpy(self.x_scalar[idx])      # (S,)
        cycle_t  = torch.tensor(self.cycle_t[idx], dtype=torch.long)

        C_t  = torch.tensor(self.C_t[idx], dtype=torch.float32)
        C_t1 = torch.tensor(self.C_t1[idx], dtype=torch.float32)
        Q_t1 = torch.tensor(self.Q_t1[idx], dtype=torch.float32)

        return x_seq_t, x_scalar, cycle_t, C_t, x_seq_t1, C_t1, Q_t1


def _build_sequence_df_A(path_A, col_cycle, seq_feature_dim, seq_len, col_prefix):
    """
    从 A 路CSV构造 sequence_df: [循环号, voltage_sequence], voltage_sequence=(L, F)
    """
    df_a = pd.read_csv(path_A, sep=',')
    feat_cols = [f'{col_prefix}{i}' for i in range(1, seq_feature_dim + 1)]
    # 分组聚合为 (L,F) 数组
    seq_df = df_a.groupby(col_cycle)[feat_cols].apply(lambda x: x.values).reset_index(name='voltage_sequence')
    # 过滤出长度==seq_len的片段
    seq_df = seq_df[seq_df['voltage_sequence'].apply(len) == seq_len]
    return seq_df


def load_and_make_pairs(config: Config):
    """
    读取A/C数据，合并，构建(t, t+1)成对样本
    返回:
      train_df, val_df, test_df, scalers, has_Q
    """
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

        # 合并 (按循环号)
        merged = pd.merge(seq_df, stat_df, on=config.col_cycle)
        merged['battery_id'] = bid
        merged = merged.sort_values([config.col_cycle]).reset_index(drop=True)

        # 构建 (t, t+1)
        # 删除最后一个循环（因为没有t+1）
        merged['voltage_sequence_t']  = merged['voltage_sequence']
        merged['voltage_sequence_t1'] = merged['voltage_sequence'].shift(-1)
        merged['C_t']  = merged[config.col_C]
        merged['C_t1'] = merged[config.col_C].shift(-1)

        has_Q = config.col_Q in merged.columns
        if has_Q:
            merged['Q_t1'] = merged[config.col_Q].shift(-1)

        # 丢弃最后一行（t+1缺失）
        merged = merged.iloc[:-1].copy()
        all_df.append(merged)

    if len(all_df) == 0:
        raise RuntimeError("未加载到任何电池数据，请检查路径与文件名。")

    full_df = pd.concat(all_df, ignore_index=True)

    # ===== 划分 =====
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df   = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df  = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # ===== 缩放器 =====
    scaler_seq    = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_C      = StandardScaler()
    scaler_Q      = StandardScaler() if has_Q else None

    # 拟合：序列（用训练集的 t 序列展平后拟合），标量（训练集），目标C/Q
    all_train_seq_t = np.vstack(train_df['voltage_sequence_t'].values)           # (N, F) 因为seq_len=1
    scaler_seq.fit(all_train_seq_t)

    scaler_scalar.fit(train_df[config.features_from_C])

    scaler_C.fit(train_df[['C_t']].values)   # 用 C_t 拟合
    if has_Q:
        scaler_Q.fit(train_df[['Q_t1']].values)

    # 变换：对 t / t+1 序列、标量、目标
    def _tf_seq(arr):
        return scaler_seq.transform(arr)

    for df in [train_df, val_df, test_df]:
        # seq t 与 t+1
        df['voltage_sequence_t']  = df['voltage_sequence_t'].apply(_tf_seq)
        df['voltage_sequence_t1'] = df['voltage_sequence_t1'].apply(_tf_seq)
        # 标量
        df.loc[:, config.features_from_C] = scaler_scalar.transform(df[config.features_from_C])
        # 目标
        df['C_t']  = scaler_C.transform(df[['C_t']].values)
        df['C_t1'] = scaler_C.transform(df[['C_t1']].values)
        if has_Q:
            df['Q_t1'] = scaler_Q.transform(df[['Q_t1']].values)

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar, 'C': scaler_C, 'Q': scaler_Q}
    return train_df, val_df, test_df, scalers, has_Q


# =============================
# 5) 训练 & 评估
# =============================
def uw(loss, log_var):  # uncertainty weighting
    return torch.exp(-log_var)*loss + log_var

def train_one_epoch(E_net, P_net, log_vars, loader, optimizer, cfg: Config, has_Q: bool):
    E_net.train(); P_net.train()
    mse = nn.MSELoss()
    total_loss = 0.0

    for (x_seq_t, x_scalar, cycle_t, C_t, x_seq_t1, C_t1, Q_t1) in loader:
        x_seq_t  = x_seq_t.to(cfg.device)                # (B,L,F)
        x_seq_t1 = x_seq_t1.to(cfg.device)               # (B,L,F)
        x_scalar = x_scalar.to(cfg.device)
        cycle_t  = cycle_t.to(cfg.device)
        C_t      = C_t.to(cfg.device).unsqueeze(-1)      # (B,1)
        C_t1     = C_t1.to(cfg.device).unsqueeze(-1)     # (B,1)
        Q_t1     = Q_t1.to(cfg.device).unsqueeze(-1)     # (B,1)

        # ---- Estimation ----
        z_t, C_hat_t, nextseq_flat = E_net(x_seq_t, x_scalar, cycle_t)
        # 序列目标扁平化
        x_seq_t1_flat = x_seq_t1.view(x_seq_t1.size(0), -1)

        L_est_C   = mse(C_hat_t, C_t) * cfg.w_est_C
        L_est_seq = mse(nextseq_flat, x_seq_t1_flat) * cfg.w_est_nextseq

        # ---- Prediction ----
        C_input = C_hat_t if cfg.pred_use_estimated_C else C_t  # Teacher Forcing 开关
        out = P_net(C_input.squeeze(-1))                         # (B,O)
        if has_Q:
            C_next_hat = out[:, 0:1]
            Q_next_hat = out[:, 1:2]
            L_pred_C = mse(C_next_hat, C_t1)  # 注意：去掉了 * cfg.w_...
            L_pred_Q = mse(Q_next_hat, Q_t1)  # 注意：去掉了 * cfg.w_...

            # 使用不确定性加权计算总损失
            loss = uw(L_est_C, log_vars['estC']) \
                   + uw(L_est_seq, log_vars['estNext']) \
                   + uw(L_pred_C, log_vars['predC']) \
                   + uw(L_pred_Q, log_vars['predQ'])
        else:
            C_next_hat = out[:, 0:1]
            L_pred_C = mse(C_next_hat, C_t1)  # 注意：去掉了 * cfg.w_...

            # 使用不确定性加权计算总损失
            loss = uw(L_est_C, log_vars['estC']) \
                   + uw(L_est_seq, log_vars['estNext']) \
                   + uw(L_pred_C, log_vars['predC'])

            # 使用统一的优化器 opt_all
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(E_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(P_net.parameters(), 1.0)
        # (裁剪log_vars是可选的，但通常不需要)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(E_net, P_net, loader, cfg: Config, scalers, has_Q: bool):
    E_net.eval(); P_net.eval()

    mse = nn.MSELoss()

    all_Ct1_pred, all_Ct1_true = [], []
    all_Qt1_pred, all_Qt1_true = [], []

    all_Ct_pred,  all_Ct_true  = [], []   # 估计当前C_t的表现
    all_nextseq_pred, all_nextseq_true = [], []

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

        # 记录估计网表现
        all_Ct_pred.append(C_hat_t.cpu().numpy())
        all_Ct_true.append(C_t.cpu().numpy())

        all_nextseq_pred.append(nextseq_flat.cpu().numpy())
        all_nextseq_true.append(x_seq_t1_flat.cpu().numpy())

        # 预测网（评估阶段默认用真值 C_t 做teacher forcing，衡量其单步精度）
        out = P_net(C_t.squeeze(-1))
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

    # 拼接
    Ct_pred  = np.concatenate(all_Ct_pred, axis=0).flatten()
    Ct_true  = np.concatenate(all_Ct_true, axis=0).flatten()
    Ct1_pred = np.concatenate(all_Ct1_pred, axis=0).flatten()
    Ct1_true = np.concatenate(all_Ct1_true, axis=0).flatten()

    # 反归一化
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

# ---- 8) 时序对比绘图 ----
@torch.no_grad()
def plot_series_teacher_forced_for_battery(df, cfg, scalers, P_net, battery_id, has_Q, title_suffix="teacher_forced"):
    sub = df[df['battery_id'] == battery_id].copy().sort_values(cfg.col_cycle).reset_index(drop=True)
    if len(sub) == 0:
        print(f"[WARN] 电池 {battery_id} 无数据"); return

    # t->t+1 的目标序列，对应的横轴用 t+1 的循环号
    cycles_next = (sub[cfg.col_cycle].values + 1).astype(int)

    # 取scaled的 C_t, C_{t+1}, (可选) Q_{t+1}
    C_t_scaled  = sub['C_t'].values.astype(np.float32)          # (N,)
    C_t1_scaled = sub['C_t1'].values.astype(np.float32)         # (N,)

    # 单步预测（teacher forcing：用真值 C_t 作为输入）
    device = cfg.device
    out = P_net(torch.tensor(C_t_scaled, device=device))
    C_pred_scaled = out[:, 0].detach().cpu().numpy()

    # 反归一化
    C_true = scalers['C'].inverse_transform(C_t1_scaled.reshape(-1,1)).flatten()
    C_pred = scalers['C'].inverse_transform(C_pred_scaled.reshape(-1,1)).flatten()

    # 画 C 曲线
    plt.figure(figsize=(9,4))
    plt.plot(cycles_next, C_true, linewidth=1.5, label='True C_{t+1}')
    plt.plot(cycles_next, C_pred, linewidth=1.5, label='Pred C_{t+1}')
    plt.xlabel('Cycle'); plt.ylabel('Cumulative Discharge (Ah)')
    plt.title(f'Battery {battery_id} | C time series ({title_suffix})'); plt.grid(True); plt.legend()
    out_path = os.path.join(cfg.save_path, f"series_C_{title_suffix}_bat{battery_id}.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
    print(f"[SAVE] {out_path}")

    # 若有 Q/SOH，再画一张
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

    # 载入数据 & 构造(t,t+1)配对
    train_df, val_df, test_df, scalers, has_Q = load_and_make_pairs(cfg)
    joblib.dump(scalers, os.path.join(cfg.save_path, 'scalers.pkl'))

    train_ds = BatteryPairDataset(train_df, cfg, scalers, has_Q)
    val_ds   = BatteryPairDataset(val_df, cfg, scalers, has_Q)
    test_ds  = BatteryPairDataset(test_df, cfg, scalers, has_Q)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 模型
    E_net = EstimationModule(cfg).to(cfg.device)
    P_net = ExpNet(n_terms=cfg.exp_n_terms, out_dims=2 if has_Q else 1).to(cfg.device)

    # 1. 定义可学习的log_vars（根据has_Q动态调整）
    log_var_dict = {
        'estC': nn.Parameter(torch.zeros(1)),
        'estNext': nn.Parameter(torch.zeros(1)),
        'predC': nn.Parameter(torch.zeros(1)),
    }
    if has_Q:
        log_var_dict['predQ'] = nn.Parameter(torch.zeros(1))

    log_vars = nn.ParameterDict(log_var_dict).to(cfg.device)

    # 2. 用 opt_all 替换 opt_E 和 opt_P，并将 log_vars 的参数加入优化器
    opt_all = optim.Adam(
        list(E_net.parameters()) + list(P_net.parameters()) + list(log_vars.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    best_val, no_improve = float('inf'), 0

    log_rows = []
    for ep in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(E_net, P_net, log_vars, train_loader, opt_all, cfg, has_Q)
        val_loss, val_metrics = evaluate(E_net, P_net, val_loader, cfg, scalers, has_Q)

        row = {'epoch': ep, 'train_loss': tr_loss, 'val_loss': val_loss, **{f'val_{k}': v for k,v in val_metrics.items()}}
        log_rows.append(row)
        print(f"Epoch {ep:03d} | train {tr_loss:.6f} | val {val_loss:.6f} | "
              + " | ".join([f"{k}:{v:.4f}" for k,v in val_metrics.items()]))

        # early stop
        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(E_net.state_dict(), os.path.join(cfg.save_path, 'E_best.pth'))
            torch.save(P_net.state_dict(), os.path.join(cfg.save_path, 'P_best.pth'))
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop at epoch {ep}.")
                break

    # 保存日志
    pd.DataFrame(log_rows).to_csv(os.path.join(cfg.save_path, 'train_log.csv'), index=False)

    # 加载最佳模型做最终评估
    E_net.load_state_dict(torch.load(os.path.join(cfg.save_path, 'E_best.pth'), map_location=cfg.device))
    P_net.load_state_dict(torch.load(os.path.join(cfg.save_path, 'P_best.pth'), map_location=cfg.device))

    test_loss, test_metrics = evaluate(E_net, P_net, test_loader, cfg, scalers, has_Q)
    print("\n== Final Test ==")
    for k,v in test_metrics.items():
        print(f"{k}: {v:.6f}")
    pd.DataFrame([test_metrics]).to_csv(os.path.join(cfg.save_path, 'test_metrics.csv'), index=False)

    for bid in sorted(test_df['battery_id'].unique()):
        # 1) 单步teacher-forced拼接
        plot_series_teacher_forced_for_battery(test_df, cfg, scalers, P_net, bid, has_Q, title_suffix="teacher_forced")

    # 画两张对角散点图（仅为快速检查）
    # 这里为了简便，重新跑一遍 test loader 的前向，拿到反归一化后的数据点
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

            _, C_hat_t, _ = E_net(x_seq_t, x_scalar, cycle_t)
            out = P_net(C_t.squeeze(-1))
            C_next_hat = out[:,0:1]

            # 反归一化
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

    # ===== RUL-by-Q（仅当 has_Q=True 且 P_net.out_dims==2 时可用）=====
    if has_Q and getattr(P_net, "out_dims", 1) >= 2:
        rul_rows = []
        for bid in sorted(test_df['battery_id'].unique()):
            # 从该电池的“最后观测循环”起预测到触线
            RUL, crossing_cycle, series = rul_by_Q(
                df=test_df, cfg=cfg, scalers=scalers,
                E_net=E_net, P_net=P_net,
                battery_id=bid,
                start_idx=None,         # None=默认用该电池最后一点为起点
                soh_thr=0.8,            # EOL 阈值（SOH=80%）
                q0=None,                # None=用该电池首个Q作为Q0；若你的输出是SOH，q0可设为1.0
                max_h=5000,
                use_estCt=True,         # 用估计的 C_t 作为起点，更贴近部署
                enforce_monotonic=True, # 强制单调，避免偶抖
                fractional=True         # 分数循环内插
            )
            print(f"[RUL-Q] Battery {bid} | RUL={RUL} cycles | crossing @ {crossing_cycle}")
            plot_rul_series_Q(series, bid, cfg, title_suffix="final")

            rul_rows.append({'battery_id': bid, 'RUL_cycles': RUL, 'crossing_cycle': crossing_cycle})

        pd.DataFrame(rul_rows).to_csv(os.path.join(cfg.save_path, "RUL_by_Q_test.csv"), index=False)
    else:
        print("[INFO] 跳过 RUL-by-Q：该模型未训练 Q/SOH 头（out_dims<2）或数据无 Q 标签。")



# =============================
# 7) 迭代推理（部署示例，可选）
# =============================
@torch.no_grad()
def rollout_iterative(E_net, P_net, initial_window, initial_scalar, initial_cycle, steps, cfg: Config, scalers, use_estCt=True):
    """
    用单步预测模型做多步迭代:
      initial_window: 最近一步的 x_seq (1,L,F)  -> 取 [:, -1, :]
      initial_scalar: 对应的标量特征 (1,S)
      initial_cycle : 对应的循环号 (1,)
    返回: arrays of C sequence (反归一化单位)
    """
    E_net.eval(); P_net.eval()

    x_seq_t  = torch.from_numpy(scalers['sequence'].transform(initial_window.squeeze(0))).to(cfg.device).unsqueeze(0)
    x_scalar = torch.from_numpy(scalers['scalar'].transform(initial_scalar)).to(cfg.device)
    cycle_t  = torch.tensor(initial_cycle, dtype=torch.long, device=cfg.device)

    z_t, C_hat_t, _ = E_net(x_seq_t, x_scalar, cycle_t)        # scaled
    # 需要一个 C_t 作为起点（若用真值请预先缩放）
    C_t = C_hat_t if use_estCt else C_hat_t

    C_list = []
    for k in range(steps):
        out = P_net(C_t.squeeze(-1))                            # scaled
        C_next = out[:,0:1]
        C_list.append(C_next.cpu().numpy())
        C_t = C_next

    C_arr_scaled = np.concatenate(C_list, axis=0)
    C_arr = scalers['C'].inverse_transform(C_arr_scaled)
    return C_arr.flatten()

# =============================
# 8) RUL 估计：按 Q/SOH 触线法（推荐）
# =============================
@torch.no_grad()
def rul_by_Q(df, cfg, scalers, E_net, P_net, battery_id,
             start_idx=None, soh_thr=0.8, q0=None, max_h=5000,
             use_estCt=True, enforce_monotonic=True, fractional=True):
    """
    从当前点开始迭代预测 Q（或 SOH），找到首次 <= 阈值 的步数作为 RUL。
    需要：P_net.out_dims > 1（即模型有第二个输出头用于 Q/SOH），且 scalers['Q'] 存在。
    返回: (RUL_cycles, crossing_cycle_index, series_dict)
    series_dict: {'cycles','Q_pred','C_pred','Q_thresh'}
    """
    if getattr(P_net, "out_dims", 1) < 2 or scalers.get('Q', None) is None:
        # 没有 Q 头或没有 Q 的 scaler，无法用此法
        return None, None, {}

    device = cfg.device
    sub = df[df['battery_id']==battery_id].sort_values(cfg.col_cycle).reset_index(drop=True)
    if len(sub)==0:
        return None, None, {}

    # 默认从该电池“最后一个已观测点”开始估 RUL
    if start_idx is None:
        start_idx = len(sub) - 1

    cyc0 = int(sub.loc[start_idx, cfg.col_cycle])

    # 起点 C_t：可选用估计网的 \hat C_t（更贴近部署），否则用真值 C_t（scaled）
    if use_estCt:
        x_seq_t  = torch.from_numpy(sub.loc[start_idx,'voltage_sequence_t']).float().unsqueeze(0).to(device)
        x_scalar = torch.from_numpy(sub.loc[start_idx, cfg.features_from_C].astype(np.float32).values).unsqueeze(0).to(device)
        cycle_t  = torch.tensor([cyc0], dtype=torch.long, device=device)
        _, C_t, _ = E_net(x_seq_t, x_scalar, cycle_t)  # scaled
    else:
        C_t = torch.tensor([[sub.loc[start_idx,'C_t']]], dtype=torch.float32, device=device)  # scaled

    # 计算阈值（如果没有传 q0，就用该电池的首个 Q 反归一化作为 Q0）
    if q0 is None:
        if 'Q_t1' in sub.columns and sub['Q_t1'].notna().any():
            q0 = scalers['Q'].inverse_transform(np.array([[sub.loc[0,'Q_t1']]], dtype=np.float32))[0,0]
        else:
            q0 = 1.0  # 若你预测的是 SOH，这里让 q0=1.0 即可
    q_thresh = q0 * soh_thr

    cycles, Q_preds, C_preds = [], [], []
    crossed_idx = None

    for h in range(1, max_h+1):
        out = P_net(C_t.squeeze(-1))            # (1, 2)
        C_next = out[:, 0:1]                    # scaled
        Q_next_scaled = out[:, 1:2].cpu().numpy()
        Q_next = scalers['Q'].inverse_transform(Q_next_scaled)[0,0]

        C_preds.append(C_next.item())
        Q_preds.append(Q_next)
        cycles.append(cyc0 + h)

        C_t = C_next  # 迭代喂回

        if Q_next <= q_thresh:
            crossed_idx = h
            break

    # 后处理（可选）：强制单调，避免偶尔向上抖动
    if enforce_monotonic and len(Q_preds) > 0:
        Q_preds = np.minimum.accumulate(np.array(Q_preds, dtype=float))
        C_preds = np.maximum.accumulate(np.array(C_preds, dtype=float))
    else:
        Q_preds = np.array(Q_preds, dtype=float)
        C_preds = np.array(C_preds, dtype=float)

    # 分数循环内插，得到更细 RUL（可关闭 fractional=False）
    RUL = None
    if crossed_idx is not None:
        if fractional and crossed_idx >= 2:
            q_hi = Q_preds[crossed_idx-2]  # t+h-1
            q_lo = Q_preds[crossed_idx-1]  # t+h
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
    """简易画图：Q 预测轨迹 + 阈值线；另存 C 轨迹方便排查"""
    if not series:
        return
    cycles = series['cycles']
    Q_pred = series['Q_pred']
    Q_thr  = series['Q_thresh']
    C_pred = series['C_pred']

    # Q/SOH vs cycle
    plt.figure(figsize=(9,4))
    plt.plot(cycles, Q_pred, marker='o', linewidth=1.5, label='Pred Q/SOH')
    plt.axhline(Q_thr, linestyle='--', label=f'Threshold={Q_thr:.3f}')
    plt.xlabel('Cycle'); plt.ylabel('Capacity / SOH')
    plt.title(f'Battery {battery_id} | RUL-by-Q ({title_suffix})')
    plt.grid(True); plt.legend(); plt.tight_layout()
    out1 = os.path.join(cfg.save_path, f"RUL_Q_bat{battery_id}_{title_suffix}.png")
    plt.savefig(out1, dpi=300); plt.close()
    print(f"[SAVE] {out1}")

    # 同时把 C 的迭代轨迹也存一张
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
