# -*- coding: utf-8 -*-
"""
L窗 + 单调增量ExpNet 的完整训练脚本（支持“起点未知” & RUL）
【计划采样修改版】
------------------------------------------------------------
- 数据：对每只电芯按循环号生成长度为 L 的连续窗口 (L×F)，模拟“起点未知”。
- 估计网 E：从窗口估计当前累计放电容量 C_t（可选CycleNet去周期）。
- 预测网 P（ExpNetMono）：
    * 输出 ΔC>=0，得到 C_{t+1}=C_t+ΔC（保证单调非减）。
    * 若提供 Q_t，则输出 ΔQ<=0，Q_{t+1}=Q_t+ΔQ（保证单调非增）；
      若不提供，则直接回归 Q_{t+1} 并加单调惩罚。
- 训练：【新增】支持计划采样（Scheduled Sampling），动态调整Teacher Forcing概率。
- 可选：是否用 \hat C_t 喂预测网（pred_use_estimated_C）。

按需修改 Config 里的路径、列名与超参即可运行。
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
    save_path: str = '/home/scuee_user06/myh/电池/result-累计放电容量/result-dualnet3.0/Lwindow-monoExp-SS/20'  # 建议换个新路径保存结果

    # train_batteries: list = field(default_factory=lambda: [1, 2, 3, 4])
    # val_batteries:   list = field(default_factory=lambda: [5])
    # test_batteries:  list = field(default_factory=lambda: [6])
    #
    # train_batteries: list = field(default_factory=lambda: [7, 8, 9, 11])
    # val_batteries:   list = field(default_factory=lambda: [10])
    # test_batteries:  list = field(default_factory=lambda: [12])
    #
    # train_batteries: list = field(default_factory=lambda: [15, 17, 18, 16])
    # val_batteries:   list = field(default_factory=lambda: [13])
    # test_batteries:  list = field(default_factory=lambda: [14])
    #
    train_batteries: list = field(default_factory=lambda: [21, 22, 23, 24])
    val_batteries: list = field(default_factory=lambda: [19])
    test_batteries: list = field(default_factory=lambda: [20])

    # 特征列（来自C路统计特征）
    features_from_C: list = field(default_factory=lambda: [
        '恒压充电时间(s)', '3.3~3.6V充电时间(s)'
    ])

    # 列名（与你CSV一致）
    col_cycle: str = '循环号'
    col_seq_prefix: str = '弛豫段电压'
    col_C: str = '累计放电容量(Ah)'
    col_Q: str = '最大容量(Ah)'

    # —— 窗口相关 ——
    window_L: int = 10
    make_windows_stride: int = 1
    random_windows_per_epoch: int = 0

    sequence_feature_dim: int = 7
    use_cyclenet: bool = False
    meta_cycle_len: int = 7

    # 模型结构
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.2

    # ExpNetMono
    exp_n_terms: int = 32
    use_q_t_as_input: bool = False

    # 训练
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 2025

    # 训练策略
    use_gpu: bool = True
    device: torch.device = field(init=False)

    # --- 新增：计划采样 ---
    use_scheduled_sampling: bool = False  # 是否启用计划采样
    # -----------------------

    # 损失权重
    w_est_C: float = 0.6
    w_est_nextseq: float = 0.05
    w_pred_Cnext: float = 1.0
    w_pred_Qnext: float = 2.0
    w_mono_penalty: float = 0.1

    huber_beta_Q: float = 0.03

    # 当 use_scheduled_sampling=False 时，此项才生效
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
# 3) 模型 (模型代码无需修改)
# =============================
class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length: int = 1):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class EstimationModule(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        in_seq = cfg.window_L * cfg.sequence_feature_dim
        in_sca = len(cfg.features_from_C)
        self.sequence_encoder = nn.Linear(in_seq, cfg.d_model // 2)
        self.scalar_encoder = nn.Linear(in_sca, cfg.d_model // 2) if in_sca > 0 else nn.Identity()
        self.cycle_queue = RecurrentCycle(cfg.meta_cycle_len, cfg.d_model) if cfg.use_cyclenet else None
        self.head_Ct = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, 1)
        )
        self.head_nextseq = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, in_seq)
        )

    def forward(self, x_win, x_scalar, cycle_number):
        B = x_win.size(0)
        x_seq_flat = x_win.view(B, -1)
        seq_emb = self.sequence_encoder(x_seq_flat)
        if isinstance(self.scalar_encoder, nn.Identity):
            sca_emb = torch.zeros(B, self.cfg.d_model // 2, device=x_win.device)
        else:
            sca_emb = self.scalar_encoder(x_scalar)
        feat = torch.cat([seq_emb, sca_emb], dim=1)
        if self.cycle_queue is not None:
            cyc_idx = cycle_number % self.cfg.meta_cycle_len
            z_t = feat - self.cycle_queue(cyc_idx, length=1).squeeze(1)
        else:
            z_t = feat
        C_hat_t = self.head_Ct(z_t)
        nextseq_flat = self.head_nextseq(z_t)
        return z_t, C_hat_t, nextseq_flat


class ExpNetMono(nn.Module):
    def __init__(self, n_terms=16, out_dims=2):
        super().__init__()
        self.out_dims = out_dims
        log_b_start, log_b_end = np.log(1e-4), np.log(1e-1)
        self.log_b = nn.Parameter(torch.linspace(log_b_start, log_b_end, n_terms).float())
        self.Wc = nn.Linear(n_terms, 1)
        self.Wq = nn.Linear(n_terms, 1) if out_dims > 1 else None
        self.bias_c = nn.Parameter(torch.zeros(1))
        self.bias_q = nn.Parameter(torch.zeros(1)) if out_dims > 1 else None

    def _phi(self, c):
        b = -(torch.nn.functional.softplus(self.log_b) + 1e-8)
        return torch.exp(c * b.unsqueeze(0))

    def forward(self, c, q_t: torch.Tensor | None = None):
        phi = self._phi(c)
        dC = torch.nn.functional.softplus(self.Wc(phi) + self.bias_c)
        C_next = c + dC
        if self.out_dims > 1:
            if (q_t is not None):
                dQ = -torch.nn.functional.softplus(self.Wq(phi) + self.bias_q)
                Q_next = q_t + dQ
            else:
                Q_next = self.Wq(phi) + self.bias_q
            return torch.cat([C_next, Q_next], dim=1)
        else:
            return C_next


# =============================
# 4) 数据构建 (无需修改)
# =============================
def _load_per_cycle_features_A(path_csv, col_cycle, feat_prefix, F):
    df = pd.read_csv(path_csv)
    feat_cols = [f'{feat_prefix}{i}' for i in range(1, F + 1)]
    per_cycle = df.groupby(col_cycle, as_index=False)[feat_cols].first()
    per_cycle['feat_vec'] = per_cycle[feat_cols].values.tolist()
    return per_cycle[[col_cycle, 'feat_vec']]


def _make_L_windows(merged_one, cfg: Config, stride=1, randN=0):
    df = merged_one.sort_values(cfg.col_cycle).reset_index(drop=True)
    L = cfg.window_L
    rows = []
    valid_t = list(range(L - 1, len(df) - 1))
    if randN and randN > 0:
        valid_t = random.sample(valid_t, k=min(randN, len(valid_t)))
        valid_t.sort()
    else:
        valid_t = valid_t[::max(1, stride)]
    for t in valid_t:
        win = np.stack(df.loc[t - L + 1:t, 'feat_vec'].to_list(), axis=0).astype(np.float32)
        x_sca = df.loc[t, cfg.features_from_C].values.astype(np.float32) if len(cfg.features_from_C) > 0 else np.zeros(
            (0,), np.float32)
        row = {
            'window_seq': win, 'x_scalar': x_sca, 'cycle': int(df.loc[t, cfg.col_cycle]),
            'C_t': float(df.loc[t, cfg.col_C]), 'C_t1': float(df.loc[t + 1, cfg.col_C]),
            'battery_id': int(df['battery_id'].iloc[0])
        }
        if cfg.col_Q in df.columns:
            row['Q_t'] = float(df.loc[t, cfg.col_Q])
            row['Q_t1'] = float(df.loc[t + 1, cfg.col_Q])
        rows.append(row)
    return pd.DataFrame(rows)


def load_and_make_windows(cfg: Config):
    all_ids = sorted(list(set(cfg.train_batteries + cfg.val_batteries + cfg.test_batteries)))
    all_cycles_df = []
    for bid in all_ids:
        path_a = os.path.join(cfg.path_A_sequence, f'relaxation_battery{bid}.csv')
        path_c = os.path.join(cfg.path_C_features, f'battery{bid}_SOH健康特征提取结果.csv')
        if not os.path.exists(path_a) or not os.path.exists(path_c):
            print(f"[WARN] 缺文件，跳过电池 {bid}")
            continue
        df_a = _load_per_cycle_features_A(path_a, cfg.col_cycle, cfg.col_seq_prefix, cfg.sequence_feature_dim)
        df_c = pd.read_csv(path_c)
        df_c.columns = [c.strip() for c in df_c.columns]
        merged = pd.merge(df_a, df_c, on=cfg.col_cycle)
        merged['battery_id'] = bid
        all_cycles_df.append(merged)
    if len(all_cycles_df) == 0: raise RuntimeError("未加载到任何电池数据。")
    full_df = pd.concat(all_cycles_df, ignore_index=True)

    seq_mat = np.vstack(full_df['feat_vec'].values).astype(np.float32)
    train_mask = full_df['battery_id'].isin(cfg.train_batteries)
    scaler_seq = StandardScaler().fit(seq_mat[train_mask])
    scaler_scalar = StandardScaler().fit(full_df.loc[train_mask, cfg.features_from_C]) if len(
        cfg.features_from_C) > 0 else None
    scaler_C = StandardScaler().fit(full_df.loc[train_mask, [cfg.col_C]].values)
    has_Q = cfg.col_Q in full_df.columns
    scaler_Q = StandardScaler().fit(full_df.loc[train_mask, [cfg.col_Q]].values) if has_Q else None

    all_win = []
    for bid in all_ids:
        sub = full_df[full_df['battery_id'] == bid].copy().sort_values(cfg.col_cycle)
        feat_arr = np.vstack(sub['feat_vec'].values).astype(np.float32)
        sub['feat_vec'] = list(scaler_seq.transform(feat_arr))
        if scaler_scalar is not None and len(cfg.features_from_C) > 0:
            sub.loc[:, cfg.features_from_C] = scaler_scalar.transform(sub[cfg.features_from_C])
        sub[cfg.col_C] = scaler_C.transform(sub[[cfg.col_C]].values)
        if has_Q: sub[cfg.col_Q] = scaler_Q.transform(sub[[cfg.col_Q]].values)
        win_df = _make_L_windows(
            sub, cfg, stride=cfg.make_windows_stride,
            randN=cfg.random_windows_per_epoch if (
                        bid in cfg.train_batteries and cfg.random_windows_per_epoch > 0) else 0
        )
        all_win.append(win_df)

    full_win = pd.concat(all_win, ignore_index=True)
    train_df = full_win[full_win['battery_id'].isin(cfg.train_batteries)].copy()
    val_df = full_win[full_win['battery_id'].isin(cfg.val_batteries)].copy()
    test_df = full_win[full_win['battery_id'].isin(cfg.test_batteries)].copy()

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar, 'C': scaler_C, 'Q': scaler_Q}
    return train_df, val_df, test_df, scalers, has_Q


# =============================
# 5) Dataset (无需修改)
# =============================
class LWindowDataset(Dataset):
    def __init__(self, df, cfg: Config, has_Q: bool):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.has_Q = has_Q

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x_win = torch.from_numpy(r['window_seq'])
        x_sca = torch.from_numpy(r['x_scalar']) if len(self.cfg.features_from_C) > 0 else torch.zeros(0)
        cycle_t = torch.tensor(r['cycle'], dtype=torch.long)
        C_t = torch.tensor(r['C_t'], dtype=torch.float32)
        C_t1 = torch.tensor(r['C_t1'], dtype=torch.float32)
        if self.has_Q and ('Q_t' in r and 'Q_t1' in r):
            Q_t = torch.tensor(r['Q_t'], dtype=torch.float32)
            Q_t1 = torch.tensor(r['Q_t1'], dtype=torch.float32)
        else:
            Q_t, Q_t1 = torch.tensor(0.0), torch.tensor(0.0)
        return x_win, x_sca, cycle_t, C_t, C_t1, Q_t, Q_t1


# =============================
# 6) 训练 & 评估
# =============================

# <<< --- 函数已修改 --- >>>
def train_one_epoch(E_net, P_net, loader, optimizer, cfg: Config, has_Q: bool, ep: int, total_epochs: int):
    E_net.train();
    P_net.train()
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=cfg.huber_beta_Q)
    total = 0.0

    # --- 计划采样逻辑 ---
    # 计算当前epoch的Teacher Forcing概率 (tf_ratio)
    # 概率从1.0 (完全使用真实值) 线性衰减到接近0.0 (主要使用模型自身预测)
    if cfg.use_scheduled_sampling:
        tf_ratio = 1.0 - (ep / total_epochs)
    # 如果不使用计划采样，则遵循原有的固定逻辑
    else:
        tf_ratio = 1.0 if not cfg.pred_use_estimated_C else 0.0

    for (x_win, x_sca, cycle_t, C_t, C_t1, Q_t, Q_t1) in loader:
        x_win = x_win.to(cfg.device)
        x_sca = x_sca.to(cfg.device)
        cycle_t = cycle_t.to(cfg.device)
        C_t = C_t.to(cfg.device).unsqueeze(-1)
        C_t1 = C_t1.to(cfg.device).unsqueeze(-1)
        Q_t = Q_t.to(cfg.device).unsqueeze(-1)
        Q_t1 = Q_t1.to(cfg.device).unsqueeze(-1)

        # ---- 1. 估计网络 ----
        z_t, C_hat_t, nextseq_flat = E_net(x_win, x_sca, cycle_t)
        L_est_C = mse(C_hat_t, C_t) * cfg.w_est_C
        x_win_flat = x_win.view(x_win.size(0), -1)
        L_est_seq = mse(nextseq_flat, x_win_flat) * cfg.w_est_nextseq

        # ---- 2. 预测网络 ----
        # --- MODIFIED PART: Scheduled Sampling ---
        # 以 tf_ratio 的概率决定是使用真实值 C_t 还是估计值 C_hat_t
        if random.random() < tf_ratio:
            C_in = C_t  # Teacher Forcing: 使用真实值
        else:
            C_in = C_hat_t.detach()  # 使用模型自身估计值，.detach()很重要，防止梯度二次传入估计网络

        if has_Q and cfg.use_q_t_as_input:
            out = P_net(C_in, q_t=Q_t)
        else:
            out = P_net(C_in, q_t=None)

        C_next_hat = out[:, 0:1]
        L_pred_C = mse(C_next_hat, C_t1) * cfg.w_pred_Cnext

        if has_Q and out.size(1) > 1:
            Q_next_hat = out[:, 1:2]
            L_pred_Q = huber(Q_next_hat, Q_t1) * cfg.w_pred_Qnext
            if not cfg.use_q_t_as_input:
                L_mono = torch.relu(Q_next_hat - Q_t).mean() * cfg.w_mono_penalty
            else:
                L_mono = 0.0
        else:
            L_pred_Q, L_mono = 0.0, 0.0

        # ---- 3. 汇总损失与反向传播 ----
        loss = L_est_C + L_est_seq + L_pred_C + L_pred_Q + (L_mono if isinstance(L_mono, float) else L_mono)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(E_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(P_net.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


# 评估函数无需修改，总是使用 Teacher Forcing 来获得可比较的、稳定的验证指标
@torch.no_grad()
def evaluate(E_net, P_net, loader, cfg: Config, scalers, has_Q: bool):
    E_net.eval();
    P_net.eval()
    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=cfg.huber_beta_Q)
    total = 0.0

    all = {k: [] for k in ['Ct_pred', 'Ct_true', 'Ct1_pred', 'Ct1_true', 'Qt1_pred', 'Qt1_true']}
    for (x_win, x_sca, cycle_t, C_t, C_t1, Q_t, Q_t1) in loader:
        x_win, x_sca = x_win.to(cfg.device), x_sca.to(cfg.device)
        cycle_t = cycle_t.to(cfg.device)
        C_t, C_t1 = C_t.to(cfg.device).unsqueeze(-1), C_t1.to(cfg.device).unsqueeze(-1)
        Q_t, Q_t1 = Q_t.to(cfg.device).unsqueeze(-1), Q_t1.to(cfg.device).unsqueeze(-1)

        z_t, C_hat_t, nextseq_flat = E_net(x_win, x_sca, cycle_t)
        # 评估时总是使用真实值 (Teacher Forcing)
        out = P_net(C_t, q_t=Q_t if (has_Q and cfg.use_q_t_as_input) else None)
        C_next_hat = out[:, 0:1]
        L = mse(C_hat_t, C_t) * cfg.w_est_C + mse(nextseq_flat, x_win.view(x_win.size(0), -1)) * cfg.w_est_nextseq
        L = L + mse(C_next_hat, C_t1) * cfg.w_pred_Cnext
        all['Ct_pred'].append(C_hat_t.cpu().numpy());
        all['Ct_true'].append(C_t.cpu().numpy())
        all['Ct1_pred'].append(C_next_hat.cpu().numpy());
        all['Ct1_true'].append(C_t1.cpu().numpy())

        if has_Q and out.size(1) > 1:
            Q_next_hat = out[:, 1:2]
            L = L + huber(Q_next_hat, Q_t1) * cfg.w_pred_Qnext
            if not cfg.use_q_t_as_input:
                L = L + torch.relu(Q_next_hat - Q_t).mean() * cfg.w_mono_penalty
            all['Qt1_pred'].append(Q_next_hat.cpu().numpy());
            all['Qt1_true'].append(Q_t1.cpu().numpy())
        total += L.item()

    def _invC(x):
        return scalers['C'].inverse_transform(x.reshape(-1, 1)).ravel()

    def _invQ(x):
        return scalers['Q'].inverse_transform(x.reshape(-1, 1)).ravel()

    def _cat(v):
        return np.concatenate(v, axis=0).flatten() if len(v) > 0 else np.array([])

    Ct_pred, Ct_true = _cat(all['Ct_pred']), _cat(all['Ct_true'])
    Ct1_pred, Ct1_true = _cat(all['Ct1_pred']), _cat(all['Ct1_true'])
    metrics = {
        'EstC_t_MAE': mean_absolute_error(_invC(Ct_true), _invC(Ct_pred)),
        'EstC_t_RMSE': np.sqrt(mean_squared_error(_invC(Ct_true), _invC(Ct_pred))),
        'EstC_t_R2': r2_score(_invC(Ct_true), _invC(Ct_pred)),  # <--- 已恢复
        'PredC_t1_MAE': mean_absolute_error(_invC(Ct1_true), _invC(Ct1_pred)),
        'PredC_t1_RMSE': np.sqrt(mean_squared_error(_invC(Ct1_true), _invC(Ct1_pred))),
        'PredC_t1_R2': r2_score(_invC(Ct1_true), _invC(Ct1_pred)),  # <--- 已恢复
    }
    if has_Q and len(all['Qt1_pred']) > 0:
        Qt1_pred, Qt1_true = _cat(all['Qt1_pred']), _cat(all['Qt1_true'])
        metrics.update({
            'PredQ_t1_MAE': mean_absolute_error(_invQ(Qt1_true), _invQ(Qt1_pred)),
            'PredQ_t1_RMSE': np.sqrt(mean_squared_error(_invQ(Qt1_true), _invQ(Qt1_pred))),
            'PredQ_t1_R2': r2_score(_invQ(Qt1_true), _invQ(Qt1_pred)),  # <--- 已恢复
        })
    return total / max(1, len(loader)), metrics


# =============================
# 7) 可视化 & RUL (无需修改)
# =============================
def plot_scatter(y_true, y_pred, title, path):
    plt.figure(figsize=(6, 6))
    mn = min(np.min(y_true), np.min(y_pred));
    mx = max(np.max(y_true), np.max(y_pred))
    plt.scatter(y_true, y_pred, s=6, alpha=0.6);
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.title(title);
    plt.xlabel('True');
    plt.ylabel('Pred')
    plt.axis('equal');
    plt.xlim(mn, mx);
    plt.ylim(mn, mx);
    plt.grid(True);
    plt.tight_layout();
    plt.savefig(path, dpi=300);
    plt.close()


@torch.no_grad()
def plot_series_teacher_forced_for_battery(df_bat, cfg, scalers, P_net, has_Q,
                                           title_suffix="teacher_forced_degradation_curve"):
    """
    【修改版】绘制 Teacher-Forcing 模式下的 Q-vs-C 退化曲线，并标记交点坐标。
    """
    sub = df_bat.copy().sort_values('cycle').reset_index(drop=True)
    if len(sub) == 0 or not has_Q:
        return

    device = cfg.device
    rol_threshold = 2.6  # 设定阈值

    # 准备 ExpNet 的输入 (t时刻的状态)
    C_t_scaled = sub['C_t'].values.astype(np.float32)
    q_t_scaled_in = None
    if has_Q and 'Q_t' in sub.columns and cfg.use_q_t_as_input:
        q_t_scaled_in = sub['Q_t'].values.astype(np.float32)

    C_in = torch.tensor(C_t_scaled, device=device).view(-1, 1)
    q_in = torch.tensor(q_t_scaled_in, device=device).view(-1, 1) if q_t_scaled_in is not None else None

    out = P_net(C_in, q_in)

    # --- 反归一化 ---
    C_t1_true_orig = scalers['C'].inverse_transform(sub['C_t1'].values.reshape(-1, 1)).ravel()
    Q_t1_true_orig = scalers['Q'].inverse_transform(sub['Q_t1'].values.reshape(-1, 1)).ravel()
    C_t1_pred_orig = scalers['C'].inverse_transform(out[:, 0].detach().cpu().numpy().reshape(-1, 1)).ravel()
    Q_t1_pred_orig = scalers['Q'].inverse_transform(out[:, 1].detach().cpu().numpy().reshape(-1, 1)).ravel()

    # --- 寻找交点 ---
    def find_intersection(x_coords, y_coords, y_threshold):
        cross_indices = np.where(y_coords <= y_threshold)[0]
        if len(cross_indices) == 0: return None
        first_cross_idx = cross_indices[0]
        if first_cross_idx == 0: return x_coords[0], y_coords[0]

        x1, y1 = x_coords[first_cross_idx - 1], y_coords[first_cross_idx - 1]
        x2, y2 = x_coords[first_cross_idx], y_coords[first_cross_idx]
        if (y1 - y2) == 0: return x2, y_threshold
        x_intersect = x1 + (x2 - x1) * (y_threshold - y1) / (y2 - y1)
        return x_intersect, y_threshold

    true_intersect = find_intersection(C_t1_true_orig, Q_t1_true_orig, rol_threshold)
    pred_intersect = find_intersection(C_t1_pred_orig, Q_t1_pred_orig, rol_threshold)

    # --- 绘图 ---
    battery_id = sub['battery_id'].iloc[0]
    plt.figure(figsize=(9, 6))

    plt.plot(C_t1_true_orig, Q_t1_true_orig, markersize=4, linewidth=1.5, label='Ground Truth Degradation Curve')
    plt.plot(C_t1_pred_orig, Q_t1_pred_orig, markersize=4, linewidth=1.5, alpha=0.8,
             label='Predicted Degradation Curve (Single-Step)')
    plt.axhline(y=rol_threshold, color='green', linestyle='--', label='ROL Threshold')

    # --- 修正区域：绘制交点并添加文本标签 ---
    if true_intersect:
        plt.scatter(true_intersect[0], true_intersect[1], c='red', marker='o', s=80, zorder=5,
                    label=f'True Intersection({true_intersect[0]:.1f})')

    if pred_intersect:
        plt.scatter(pred_intersect[0], pred_intersect[1], c='red', marker='x', s=80, zorder=5,
                    label=f'Predicted Intersection({pred_intersect[0]:.1f})')

    plt.xlabel('Cumulative Discharge Capacity (Ah)')
    plt.ylabel('Capacity / SOH (Ah)')
    plt.title(f'Battery {battery_id}: Capacity vs. Cumulative Discharge ({title_suffix})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(cfg.save_path, f"degradation_curve_{title_suffix}_bat{battery_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVE] Correctly plotted Q-vs-C curve with annotations to: {out_path}")


@torch.no_grad()
def rul_by_Q(df_bat, cfg, scalers, E_net, P_net, soh_thr=0.8, q0=None, start_idx=None, max_h=5000, use_estCt=True,
             enforce_monotonic=True, fractional=True):
    # This function remains unchanged.
    sub = df_bat.copy().sort_values('cycle').reset_index(drop=True)
    if len(sub) == 0: return None, None, {}
    device = cfg.device
    if start_idx is None: start_idx = len(sub) - 1
    cyc0 = int(sub.loc[start_idx, 'cycle'])

    x_win = torch.from_numpy(sub.loc[start_idx, 'window_seq']).float().unsqueeze(0).to(device)
    x_sca = torch.from_numpy(sub.loc[start_idx, 'x_scalar']).float().unsqueeze(0).to(device) if len(
        cfg.features_from_C) > 0 else torch.zeros(1, 0, device=device)
    if use_estCt:
        _, C_t, _ = E_net(x_win, x_sca, torch.tensor([cyc0], dtype=torch.long, device=device))
    else:
        C_t = torch.tensor([[sub.loc[start_idx, 'C_t']]], dtype=torch.float32, device=device)

    has_Q = ('Q_t1' in sub.columns)
    if q0 is None:
        if has_Q and sub['Q_t1'].notna().any():
            q0 = scalers['Q'].inverse_transform(np.array([[sub.loc[0, 'Q_t1']]], dtype=np.float32))[0, 0]
        else:
            q0 = 1.0
    q_thresh = q0 * soh_thr

    cycles, Q_preds, C_preds, crossed_idx = [], [], [], None
    q_t_cur = torch.tensor([[sub.loc[start_idx, 'Q_t']]], dtype=torch.float32, device=device) if (
                has_Q and cfg.use_q_t_as_input) else None

    for h in range(1, max_h + 1):
        out = P_net(C_t, q_t=q_t_cur)
        C_next = out[:, 0:1]
        C_preds.append(C_next.item());
        cycles.append(cyc0 + h)
        if has_Q and out.size(1) > 1:
            Q_next = out[:, 1:2]
            q_next = scalers['Q'].inverse_transform(Q_next.cpu().numpy())[0, 0]
            Q_preds.append(q_next)
            if q_next <= q_thresh and crossed_idx is None: crossed_idx = h
            if cfg.use_q_t_as_input: q_t_cur = Q_next.detach()
        C_t = C_next.detach()
        if crossed_idx is not None: break

    if enforce_monotonic and len(Q_preds) > 0:
        Q_preds = np.minimum.accumulate(np.array(Q_preds, dtype=float))
    else:
        Q_preds = np.array(Q_preds, dtype=float)

    RUL = None
    if crossed_idx is not None:
        if fractional and crossed_idx >= 2:
            q_hi, q_lo = Q_preds[crossed_idx - 2], Q_preds[crossed_idx - 1]
            if q_hi > q_lo:
                frac = (q_hi - q_thresh) / (q_hi - q_lo)
                RUL = (crossed_idx - 1) + float(np.clip(frac, 0.0, 1.0))
            else:
                RUL = float(crossed_idx)
        else:
            RUL = float(crossed_idx)

    series = {'cycles': np.array(cycles, dtype=int), 'Q_pred': Q_preds,
              'C_pred': scalers['C'].inverse_transform(np.array(C_preds).reshape(-1, 1)).ravel(),
              'Q_thresh': q_thresh, 'start_cycle': cyc0}
    return RUL, (cyc0 + (RUL if RUL is not None else np.nan)), series


# =============================
# 8) 主流程
# =============================
def main():
    warnings.filterwarnings('ignore')
    cfg = Config();
    set_seed(cfg.seed)
    print(f"Save to: {cfg.save_path} | Device: {cfg.device}")
    print(f"Scheduled Sampling Enabled: {cfg.use_scheduled_sampling}")

    train_df, val_df, test_df, scalers, has_Q = load_and_make_windows(cfg)
    joblib.dump(scalers, os.path.join(cfg.save_path, 'scalers.pkl'))

    train_loader = DataLoader(LWindowDataset(train_df, cfg, has_Q), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(LWindowDataset(val_df, cfg, has_Q), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(LWindowDataset(test_df, cfg, has_Q), batch_size=cfg.batch_size, shuffle=False)

    E_net = EstimationModule(cfg).to(cfg.device)
    P_net = ExpNetMono(n_terms=cfg.exp_n_terms, out_dims=2 if has_Q else 1).to(cfg.device)

    opt_all = optim.Adam(list(E_net.parameters()) + list(P_net.parameters()), lr=cfg.learning_rate,
                         weight_decay=cfg.weight_decay)

    best_val, no_improve = float('inf'), 0
    log_rows = []
    for ep in range(1, cfg.epochs + 1):
        # <<< --- 函数调用已修改 --- >>>
        tr_loss = train_one_epoch(E_net, P_net, train_loader, opt_all, cfg, has_Q, ep, cfg.epochs)
        val_loss, val_metrics = evaluate(E_net, P_net, val_loader, cfg, scalers, has_Q)

        log_entry = {'epoch': ep, 'train_loss': tr_loss, 'val_loss': val_loss}
        log_entry.update({f'val_{k}': v for k, v in val_metrics.items()})
        log_rows.append(log_entry)

        print(f"Epoch {ep:03d} | train {tr_loss:.6f} | val {val_loss:.6f} | " + " | ".join(
            [f"{k}:{v:.4f}" for k, v in val_metrics.items()]))

        val_key = val_metrics.get('PredQ_t1_RMSE', 0.0) if has_Q else val_metrics.get('PredC_t1_RMSE', 0.0)
        if val_key < best_val:
            best_val, no_improve = val_key, 0
            torch.save(E_net.state_dict(), os.path.join(cfg.save_path, 'E_best.pth'))
            torch.save(P_net.state_dict(), os.path.join(cfg.save_path, 'P_best.pth'))
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop at epoch {ep}.")
                break
    pd.DataFrame(log_rows).to_csv(os.path.join(cfg.save_path, 'train_log.csv'), index=False)

    # --- 后续测试和可视化部分无需修改 ---
    E_net.load_state_dict(torch.load(os.path.join(cfg.save_path, 'E_best.pth'), map_location=cfg.device))
    P_net.load_state_dict(torch.load(os.path.join(cfg.save_path, 'P_best.pth'), map_location=cfg.device))
    test_loss, test_metrics = evaluate(E_net, P_net, test_loader, cfg, scalers, has_Q)
    print("\n== Final Test ==")
    for k, v in test_metrics.items(): print(f"{k}: {v:.6f}")
    pd.DataFrame([test_metrics]).to_csv(os.path.join(cfg.save_path, 'test_metrics.csv'), index=False)

    @torch.no_grad()
    def gather_preds(loader):
        E_net.eval();
        P_net.eval()
        C_true, C_pred, Q_true, Q_pred = [], [], [], []
        for (x_win, x_sca, cycle_t, C_t, C_t1, Q_t, Q_t1) in loader:
            out = P_net(C_t.unsqueeze(-1).to(cfg.device),
                        q_t=Q_t.unsqueeze(-1).to(cfg.device) if (has_Q and cfg.use_q_t_as_input) else None)
            C_true.append(scalers['C'].inverse_transform(C_t1.unsqueeze(-1).cpu().numpy()).ravel())
            C_pred.append(scalers['C'].inverse_transform(out[:, 0:1].cpu().numpy()).ravel())
            if has_Q and out.size(1) > 1:
                Q_true.append(scalers['Q'].inverse_transform(Q_t1.unsqueeze(-1).cpu().numpy()).ravel())
                Q_pred.append(scalers['Q'].inverse_transform(out[:, 1:2].cpu().numpy()).ravel())
        C_true, C_pred = np.concatenate(C_true), np.concatenate(C_pred)
        if has_Q and len(Q_pred) > 0:
            Q_true, Q_pred = np.concatenate(Q_true), np.concatenate(Q_pred)
        else:
            Q_true, Q_pred = None, None
        return C_true, C_pred, Q_true, Q_pred

    C_true, C_pred, Q_true, Q_pred = gather_preds(test_loader)
    plot_scatter(C_true, C_pred, 'C_{t+1} True vs Pred', os.path.join(cfg.save_path, 'scatter_C.png'))
    if has_Q and Q_true is not None:
        plot_scatter(Q_true, Q_pred, 'Q_{t+1} True vs Pred', os.path.join(cfg.save_path, 'scatter_Q.png'))

    for bid in sorted(test_df['battery_id'].unique()):
        plot_series_teacher_forced_for_battery(test_df[test_df['battery_id'] == bid], cfg, scalers, P_net, has_Q)

    print(f"\nArtifacts saved in: {cfg.save_path}")


if __name__ == '__main__':
    main()