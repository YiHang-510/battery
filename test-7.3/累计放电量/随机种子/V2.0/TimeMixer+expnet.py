import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib
import shutil
from dataclasses import dataclass
from typing import Tuple
import torch.nn.functional as F
import math  # ▼▼▼【核心修改】为新的 ExpNetTR 添加 math 库 ▼▼▼

# --- 前置准备：确保绘图后端正常 ---
matplotlib.use('Agg')


# =================================================================================
# 1. 模型定义
# =================================================================================

# --- TimeMixer+PI-Res 模型定义 (保持不变) ---

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
    def __init__(self, seq_channels: int, scalar_dim: int, cfg: TMPIResConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.embed = nn.Linear(seq_channels, d)
        self.tm = TimeMixerStack(d, cfg.n_blocks, cfg.kernel_sizes, cfg.dropout)
        self.enc_s = nn.Sequential(
            nn.Linear(scalar_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d)
        )
        self.basis = ISplineBasis(cfg.n_basis, cfg.degree, cfg.n_grid)
        self.c0 = nn.Parameter(torch.randn(cfg.n_basis))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.res_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_basis)
        )

    def forward(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x = self.embed(v)
        H = self.tm(x)
        h_seq = H.mean(dim=1)
        h = h_seq + self.enc_s(s)

        B_I = self.basis.eval(t_norm)
        c0_pos = F.softplus(self.c0)
        m = (B_I @ c0_pos) + self.b0
        c_h = F.softplus(self.res_head(h))
        R = (B_I * c_h).sum(dim=-1)
        Q = m + R
        return Q, {"c_h": c_h}


# ▼▼▼【核心修改】替换为最终版的 ExpNetTR 模型 ▼▼▼
class ExpNetTR(nn.Module):
    """
    Trend (mixture of exponentials) + local Residual (Gaussian bumps).
    - 允许容量“再生”局部上升（由残差负责）
    - 趋势项稳、可解释；残差项可局部正/负，幅度有界，避免发散
    """

    def __init__(self, n_terms=16, n_bumps=8, use_logspace_tau=True):
        super().__init__()
        self.n_terms = n_terms
        self.n_bumps = n_bumps

        # ---- Trend：指数混合（不做单调硬约束，b自动学负值为主，也允许正值）
        # 权重 α -> softmax；衰减率 τ -> softplus；输入尺度 gamma -> softplus
        self.raw_alpha = nn.Parameter(0.01 * torch.randn(n_terms))
        if use_logspace_tau:
            # 覆盖更快到更慢（前段需要非常快的衰减项）
            max_log = 4.0  # 可试 4.5~5.0 -> tau_max ≈ e^4.5≈90 或 e^5≈148
            init_tau = torch.exp(torch.linspace(-2.5, max_log, steps=n_terms))  # ~[0.082, 90]
            self.raw_tau = nn.Parameter(torch.log(init_tau)+ 0.01 * torch.randn(n_terms))
        else:
            self.raw_tau = nn.Parameter(torch.randn(n_terms) * 0.1)
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))  # 输入尺度

        # 趋势的上下界（可选）：不强制到 [0,1]，给线性输出更自由
        # 也可以改成 y = y_inf + (y0 - y_inf)*mix，提升可解释性
        self.trend_bias = nn.Parameter(torch.tensor(0.8) + 0.01 * torch.randn(()))       # 类似 y0
        self.trend_gain = nn.Parameter(torch.tensor(-0.5) + 0.01 * torch.randn(()))      # 类似 (y_inf - y0)，初始向下

        # ---- Residual：局部高斯凸起（允许正/负），专门刻画“再生/回落”
        # 中心 μ 放在 [0,1] 的等距初值；σ 用 softplus 保正；权重用 tanh 限幅更稳
        # mu = torch.linspace(0.05, 0.95, steps=n_bumps)          # 归一化 C 轴上的中心
        # self.mu = nn.Parameter(mu)                               # 可学习中心
        # self.raw_sigma = nn.Parameter(torch.full((n_bumps,), -1.0))  # softplus(-1)≈0.31

        # 残差：头/中/尾分配，更窄的头尾以刻画局部形状
        n_head = max(2, int(self.n_bumps * 0.35))  # ~35% 盯头部
        n_mid = max(1, int(self.n_bumps * 0.20))  # ~20% 过渡
        n_tail = self.n_bumps - n_head - n_mid

        # 头部用“对数间距”更密更靠近 0（0.001~0.15）
        mu_head = torch.exp(torch.linspace(math.log(1e-3), math.log(0.15), steps=n_head))
        # 中段均匀
        mu_mid = torch.linspace(0.15, 0.70, steps=n_mid)
        # 尾段更密，并允许略超 1 兜住边界效应
        mu_tail = torch.linspace(0.70, 1.02, steps=n_tail)

        self.mu = nn.Parameter(torch.cat([mu_head, mu_mid, mu_tail]))

        # 头部更窄，中段中等，尾部较窄
        self.raw_sigma = nn.Parameter(torch.cat([
            torch.full((n_head,), -2.3),  # σ≈softplus(-2.3)≈0.10
            torch.full((n_mid,), -1.3),  # σ≈0.27
            torch.full((n_tail,), -2.0),  # σ≈0.13
        ]))

        self.raw_beta = nn.Parameter(torch.zeros(n_bumps))  # 残差权重（tanh 限幅）
        self.raw_res_scale = nn.Parameter(torch.tensor(-2.0))  # 残差总幅度缩放

        # ---- 可选：学习一个输入平移（适配不同起点）
        self.input_shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, c, return_components=False):
        # c: [B] or [B,1]，建议外部把 C 归一化到 [0,1]；若未归一，也能靠 gamma 学到尺度
        c = c.view(-1, 1)
        c_ = c - self.input_shift

        # Trend
        alpha = F.softmax(self.raw_alpha, dim=0)  # [K]
        tau = torch.exp(self.raw_tau)  # ≥0，数值更稳
        # 可选再限幅，防数值病态：
        tau = tau.clamp_max(80.0)
        gamma = F.softplus(self.raw_gamma) + 1e-6  # >=0
        # 指数基函数：exp(-tau * gamma * c_ )，允许 c_ < 0 时更灵活
        expo = torch.exp(- (c_ * gamma) @ tau.view(1, -1))  # [B,K]
        mix = (expo * alpha.view(1, -1)).sum(dim=1, keepdim=True)  # [B,1] in (0,1]
        trend = self.trend_bias + self.trend_gain * mix  # [B,1]

        # Residual：Gaussian bumps（允许正负，幅度受 tanh + scale 控制）
        sigma = F.softplus(self.raw_sigma) + 1e-6  # [M] >0
        beta = torch.tanh(self.raw_beta)  # [-1,1]
        res_scale = torch.sigmoid(self.raw_res_scale)  # (0,1) 小幅度优先
        # [B,M]
        gauss = torch.exp(-0.5 * ((c_ - self.mu.view(1, -1)) / sigma.view(1, -1)) ** 2)
        residual = res_scale * (gauss * beta.view(1, -1)).sum(dim=1, keepdim=True)  # [B,1]

        y = (trend + residual).view(-1)  # 不强制到 [0,1]

        if not return_components:
            return y
        else:
            comps = {
                "alpha": alpha, "tau": tau, "gamma": gamma,
                "trend_bias": self.trend_bias, "trend_gain": self.trend_gain,
                "mu": self.mu, "sigma": sigma, "beta": beta, "res_scale": res_scale,
                "trend": trend.view(-1), "residual": residual.view(-1)
            }
            return y, comps


# =================================================================================
# 2. 配置参数
# =================================================================================
class Config:
    def __init__(self):
        # --- 1. 输入路径设置 ---
        self.data_path_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.data_path_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'

        self.capacity_model_base_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/TM_PIRes/20'
        self.expnet_base_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/result-expnetTR-64-test/20'

        # --- 2. 输出路径设置 ---
        self.save_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/combine_TMPIRes_ExpNetTR_final-test/20'

        # --- 3. 待测试电池ID ---
        # self.test_battery_ids = [4, 12, 14, 20] 
        self.test_battery_ids = [20]

        # --- 4. 模型配置 (必须与训练时完全一致!) ---
        # TM_PIRes 的配置
        self.sequence_length = 1
        self.sequence_feature_dim = 7
        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.scalar_feature_dim = len(self.features_from_C)

        self.d_model = 128
        self.n_blocks = 3
        self.kernel_sizes = (3, 5, 7)
        self.dropout = 0.1
        self.n_basis = 10
        self.degree = 3
        self.n_grid = 512
        self.residual_l2 = 1e-4

        self.cycle_norm_min = None
        self.cycle_norm_max = None

        # ExpNetTR 的配置 (保持不变)
        self.expnet_config = {
            'n_terms': 64,
            'n_bumps': 8,
            'use_logspace_tau': True,
            'nominal_capacity': 3.5
        }

        # --- 5. 其他设置 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================================================
# 3. 数据加载与预处理函数 (保持不变)
# =================================================================================
def load_and_preprocess_single_battery(config, scalers, battery_id):
    try:
        path_a = os.path.join(config.data_path_sequence, f'relaxation_battery{battery_id}.csv')
        path_c = os.path.join(config.data_path_features, f'battery{battery_id}_SOH健康特征提取结果.csv')
        df_a = pd.read_csv(path_a)
        df_c = pd.read_csv(path_c)
        df_c.rename(columns=lambda x: x.strip(), inplace=True)
    except FileNotFoundError as e:
        print(f"警告: 电池 {battery_id} 的数据文件未找到，已跳过。错误: {e}")
        return None, None, None, None, None, None

    feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
    sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(name='voltage_sequence')
    sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.sequence_length]

    full_df = pd.merge(sequence_df, df_c, on='循环号')

    true_capacity = full_df['最大容量(Ah)'].values
    true_soh = true_capacity / config.expnet_config['nominal_capacity']

    scaler_seq = scalers['sequence']
    scaler_scalar = scalers['scalar']

    scalar_feature_cols = config.features_from_C
    full_df['voltage_sequence'] = full_df['voltage_sequence'].apply(lambda x: scaler_seq.transform(x))
    full_df[scalar_feature_cols] = scaler_scalar.transform(full_df[scalar_feature_cols])

    sequences = np.array(full_df['voltage_sequence'].tolist(), dtype=np.float32)
    scalars = full_df[scalar_feature_cols].values.astype(np.float32)
    cycle_indices = full_df['循环号'].values.astype(np.int64)

    x_seq_tensor = torch.from_numpy(sequences).to(config.device)
    x_scalar_tensor = torch.from_numpy(scalars).to(config.device)
    cycle_idx_tensor = torch.from_numpy(cycle_indices).to(config.device)
    cumulative_capacity = full_df['累计放电容量(Ah)'].values

    return x_seq_tensor, x_scalar_tensor, cycle_idx_tensor, true_soh, full_df['循环号'].values, cumulative_capacity


# =================================================================================
# 4. 预测函数 (保持不变)
# =================================================================================
def predict_capacity(model, config, x_seq, x_scalar, cycle_idx):
    """使用 TM_PIRes 模型预测累计放电容量"""
    model.eval()
    with torch.no_grad():
        t_min = torch.tensor(config.cycle_norm_min, device=config.device, dtype=torch.float32)
        t_max = torch.tensor(config.cycle_norm_max, device=config.device, dtype=torch.float32)
        t_norm = (cycle_idx.float() - t_min) / (t_max - t_min)
        t_norm = torch.clamp(t_norm, 0.0, 1.0)

        scaled_preds, _ = model(x_seq, x_scalar, t_norm)
        scaled_preds = scaled_preds.cpu().numpy()

    predicted_capacity = scaled_preds.flatten()
    return predicted_capacity


def predict_soh_with_expnet(model, predicted_capacity, device):
    """使用 ExpNetTR 和预测的容量来预测SOH"""
    model.eval()
    capacity_tensor = torch.tensor(predicted_capacity, dtype=torch.float32, device=device)
    with torch.no_grad():
        predicted_soh = model(capacity_tensor).cpu().numpy()
    return predicted_soh


# =================================================================================
# 5. 评估与可视化 (保持不变)
# =================================================================================
def plot_diagonal_scatter(labels, preds, title, save_path):
    plt.figure(figsize=(8, 8))
    min_val = min(np.min(labels), np.min(preds)) * 0.98
    max_val = max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6, label='True Value vs. Predicted Value')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Best prediction')
    plt.xlabel('True SOH', fontsize=12)
    plt.ylabel('Predicted SOH', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_and_visualize(save_path, battery_id, cycle_nums, true_soh, final_pred_soh, cumulative_capacity):
    mae = mean_absolute_error(true_soh, final_pred_soh)
    mape = mean_absolute_percentage_error(true_soh, final_pred_soh)
    mse = mean_squared_error(true_soh, final_pred_soh)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_soh, final_pred_soh)
    metrics_data = {'Battery_ID': [battery_id], 'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2]}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_save_path = os.path.join(save_path, 'fusion_model_evaluation_metrics.csv')
    if os.path.exists(metrics_save_path):
        metrics_df.to_csv(metrics_save_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        metrics_df.to_csv(metrics_save_path, index=False, encoding='utf-8')
    results_df = pd.DataFrame({'循环号': cycle_nums, '真实SOH': true_soh, '预测SOH': final_pred_soh})
    csv_path = os.path.join(save_path, f'battery_{battery_id}_fusion_prediction.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    plt.figure(figsize=(12, 7))
    plt.plot(cumulative_capacity, true_soh, 'o-', label='True SOH', color='royalblue', markersize=4)
    plt.plot(cumulative_capacity, final_pred_soh, '^-', label='Predicted SOH', color='darkorange', markersize=4,
             alpha=0.8)
    plt.title(f'battery {battery_id}: True SOH vs. Predicted SOH', fontsize=16)
    plt.xlabel('Accumulated discharge capacity (Ah)', fontsize=12)
    plt.ylabel('SOH', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'battery_{battery_id}_soh_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    scatter_plot_path = os.path.join(save_path, f'battery_{battery_id}_diagonal_scatter_plot.png')
    plot_diagonal_scatter(labels=true_soh, preds=final_pred_soh,
                          title=f'battery {battery_id}: True SOH vs. Predicted SOH', save_path=scatter_plot_path)


# =================================================================================
# 6. 主执行函数 (保持不变)
# =================================================================================
def main():
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"使用设备: {config.device}")

    num_runs = 5
    all_runs_metrics = []
    best_run_mae = float('inf')
    best_run_dir = None
    best_run_number = -1

    for run_number in range(1, num_runs + 1):
        print(f"\n{'=' * 30}\n 开始第 {run_number}/{num_runs} 次融合实验 \n{'=' * 30}")

        current_capacity_model_path = os.path.join(config.capacity_model_base_path, f'run_{run_number}')
        current_expnet_path = os.path.join(config.expnet_base_path, f'run_{run_number}')
        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)
        metrics_summary_file = os.path.join(run_save_path, 'fusion_model_evaluation_metrics.csv')
        if os.path.exists(metrics_summary_file):
            os.remove(metrics_summary_file)

        try:
            capacity_model_path = os.path.join(current_capacity_model_path, 'best_model.pth')
            expnet_model_path = os.path.join(current_expnet_path, 'best_expnet_model.pth')
            scalers_path = os.path.join(current_capacity_model_path, 'scalers.pkl')
            scalers = joblib.load(scalers_path)

            if 'cycle_norm' in scalers and isinstance(scalers['cycle_norm'], dict):
                config.cycle_norm_min = scalers['cycle_norm']['min']
                config.cycle_norm_max = scalers['cycle_norm']['max']
                print(f"成功加载周期归一化参数: min={config.cycle_norm_min:.2f}, max={config.cycle_norm_max:.2f}")
            else:
                raise ValueError("错误: 在 scalers.pkl 中未找到 'cycle_norm' 参数，无法初始化模型。")

            # 加载 TM_PIRes 模型
            tm_config = TMPIResConfig(
                d_model=config.d_model, n_blocks=config.n_blocks, kernel_sizes=config.kernel_sizes,
                dropout=config.dropout, n_basis=config.n_basis, degree=config.degree,
                n_grid=config.n_grid, residual_l2=config.residual_l2
            )
            capacity_model = TM_PIRes(
                seq_channels=config.sequence_feature_dim,
                scalar_dim=config.scalar_feature_dim,
                cfg=tm_config
            ).to(config.device)
            capacity_model.load_state_dict(torch.load(capacity_model_path, map_location=config.device))

            # 加载 ExpNetTR 模型
            expnet_model = ExpNetTR(
                n_terms=config.expnet_config['n_terms'],
                n_bumps=config.expnet_config['n_bumps'],
                use_logspace_tau=config.expnet_config['use_logspace_tau']
            ).to(config.device)
            expnet_model.load_state_dict(torch.load(expnet_model_path, map_location=config.device))

            print(f"Run {run_number}: 模型和缩放器加载成功！")
        except Exception as e:
            print(f"\n错误: 无法加载 Run {run_number} 的模型或缩放器文件: {e}")
            print("请确保之前的训练脚本已成功生成对应 run 文件夹。跳过此次实验。")
            continue

        for battery_id in config.test_battery_ids:
            print(f"\n--- 正在处理电池 {battery_id} ---")
            x_seq, x_scalar, cycle_idx, true_soh, cycle_nums, cumulative_capacity = load_and_preprocess_single_battery(
                config, scalers, battery_id)
            if x_seq is None:
                continue

            # 1. TM_PIRes 预测出原始尺度的容量
            predicted_capacity = predict_capacity(capacity_model, config, x_seq, x_scalar, cycle_idx)
            predicted_capacity = np.clip(predicted_capacity, a_min=0.0, a_max=None)  # 可以保留裁剪

            # 2. 加载 ExpNetTR 训练时使用的归一化参数
            cap_scaler_path = os.path.join(current_expnet_path, 'capacity_scaler.pkl')
            capacity_scaler = joblib.load(cap_scaler_path)
            c_min = capacity_scaler['c_min']
            c_max = capacity_scaler['c_max']

            # 3. 对预测出的容量进行归一化
            predicted_capacity_normalized = (predicted_capacity - c_min) / (c_max - c_min + 1e-8)

            # 4. 将【归一化后】的容量传入 ExpNetTR 进行SOH预测
            final_predicted_soh = predict_soh_with_expnet(expnet_model, predicted_capacity_normalized, config.device)

            # 5. 评估最终结果
            evaluate_and_visualize(
                save_path=run_save_path, battery_id=battery_id, cycle_nums=cycle_nums,
                true_soh=true_soh, final_pred_soh=final_predicted_soh, cumulative_capacity=cumulative_capacity
            )
            print(f"电池 {battery_id} 处理完成。结果已保存至 {run_save_path}")

        if os.path.exists(metrics_summary_file):
            run_metrics_df = pd.read_csv(metrics_summary_file)
            avg_metrics = run_metrics_df.mean().to_dict()
            avg_metrics['run'] = run_number
            all_runs_metrics.append(avg_metrics)
            current_run_mae = avg_metrics.get('MAE', float('inf'))
            print(
                f"\n--- Run {run_number} 评估汇总 ---\n  - 平均 MAE: {current_run_mae:.4f}\n  - 平均 R²:  {avg_metrics.get('R2', 0):.4f}")

            if current_run_mae < best_run_mae:
                best_run_mae = current_run_mae
                best_run_dir = run_save_path
                best_run_number = run_number
                print(f"*** 新的最佳表现！平均 MAE: {best_run_mae:.4f} ***")

    print(f"\n\n{'=' * 50}\n 所有融合实验均已完成。\n{'=' * 50}")
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        cols = ['run'] + [col for col in summary_df.columns if col != 'run']
        summary_df = summary_df[cols]
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 ---\n", summary_df.to_string())
        print(f"\n汇总指标已保存到: {summary_path}")

    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (平均 MAE 最低: {best_run_mae:.4f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")
        for filename in os.listdir(best_run_dir):
            source_file = os.path.join(best_run_dir, filename)
            destination_file = os.path.join(config.save_path, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)
        print("最佳结果复制完成。")
    else:
        print("未能确定最佳实验轮次。")

    print(f"\n评估完成。所有结果已保存到: {config.save_path}")


if __name__ == '__main__':
    main()