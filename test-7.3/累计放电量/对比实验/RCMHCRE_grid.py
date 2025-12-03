"""
RCMHCRE + BatteryPINN (Physics-Guided) - Grid Search Script
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
import itertools  # 必须导入

# ==========================================
# 0. 网格搜索超参数空间定义
# ==========================================
GRID_SEARCH_PARAMS = {
    # --- 模型结构参数 ---
    'd_model': [64, 128, 256],  # 隐藏层维度
    'n_basis': [10, 20],  # I-spline 基函数数量
    'degree': [3],  # Spline 阶数 (通常固定为3)

    # --- 正则化与训练参数 ---
    'dropout': [0.1, 0.2, 0.3],
    'residual_l2': [1e-4, 1e-3],  # PINN 残差约束的权重
    'learning_rate': [0.001, 0.005],

    # --- 特征参数 (可选) ---
    'entropy_max_scale': [6]  # RCMHCRE 的最大尺度
}


# ==========================================
# 1. 配置参数 (已修改支持动态传参)
# ==========================================
class Config:
    def __init__(self, **kwargs):
        # --- 基础路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'

        # 网格搜索结果保存的根目录
        self.base_save_path = '/home/scuee_user06/myh/电池/result-累计放电容量V2.0_correct/RCMHCRE/grid_search_result'

        # --- 电池分组 (保持不变) ---
        self.train_batteries = [16, 8, 15, 7]
        self.val_batteries = [23]
        self.test_batteries = [24]

        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7
        self.sequence_length = 1
        self.cap_norm = 3.5

        # --- 模型超参数 (默认值) ---
        self.entropy_max_scale = 6
        self.d_model = 128
        self.dropout = 0.1
        self.n_basis = 10
        self.degree = 3
        self.n_grid = 512
        self.residual_l2 = 1e-4

        # --- 训练参数 ---
        self.epochs = 300  # 网格搜索时可适当减少
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.patience = 15
        self.mode = 'both'

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.cycle_norm_min = None
        self.cycle_norm_max = None

        # --- 核心逻辑: 使用 kwargs 覆盖默认值 ---
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # --- 自动计算依赖参数 ---
        # 注意: scalar_feature_dim 依赖于 features_from_C 和 entropy_max_scale
        # 如果 entropy_max_scale 被 grid search 修改了，这里会自动更新
        self.scalar_feature_dim = len(self.features_from_C) + self.entropy_max_scale

        # --- 生成参数标识字符串 ---
        self.param_str = "_".join([f"{k}-{v}" for k, v in kwargs.items()])
        if not self.param_str:
            self.param_str = "default_params"

        self.save_path = os.path.join(self.base_save_path, self.param_str)


# ==========================================
# 2. 固定随机种子 (不变)
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 3. 模型组件 (RCMHCREExtractor, Spline, Model)
# ==========================================
class RCMHCREExtractor(nn.Module):
    def __init__(self, max_scale: int = 6):
        super().__init__()
        self.max_scale = max_scale

    @staticmethod
    def _composite_multiscale(x: torch.Tensor, s: int) -> torch.Tensor:
        B, T = x.shape
        L = (T // s) * s
        if L <= 0: return x.unsqueeze(1)
        xs = x[:, :L]
        ys = torch.stack([xs[:, e:L:s] for e in range(s)], dim=1)
        return ys.mean(dim=1)

    @staticmethod
    def _hilbert_transform_fft(y: torch.Tensor) -> torch.Tensor:
        Y = torch.fft.rfft(y, dim=-1)
        H = torch.zeros_like(Y.real)
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
        inst_amp = torch.abs(z)
        inst_pow = inst_amp ** 2
        total_pow = torch.sum(inst_pow, dim=-1, keepdim=True)
        return inst_pow / (total_pow + 1e-9)

    @staticmethod
    def _cre(p: torch.Tensor) -> torch.Tensor:
        p_sorted, _ = torch.sort(p, dim=-1, descending=True)
        cdf = torch.cumsum(p_sorted, dim=-1)
        cdf = torch.clamp(cdf, 0.0, 1.0)
        entropy = -torch.sum(torch.log(cdf + 1e-9), dim=-1)
        return entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        entropies = []
        for s in range(1, self.max_scale + 1):
            y_s = self._composite_multiscale(x, s)
            z_s = self._hilbert_transform_fft(y_s)
            p_s = self._power_norm(z_s)
            h_s = self._cre(p_s)
            entropies.append(h_s)
        return torch.stack(entropies, dim=1)


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
        B_grid = self._bspline_basis_grid(knots, degree, grid)
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

    @staticmethod
    def _bspline_basis_grid(knots, degree, grid):
        device, dtype = knots.device, knots.dtype
        G, M = grid.numel(), knots.numel() - degree - 1
        Bk = torch.zeros(G, M, device=device, dtype=dtype)
        for i in range(M):
            left, right = knots[i], knots[i + 1]
            cond = (grid >= left) & (grid < right) if i < M - 1 else (grid >= left) & (grid <= right)
            Bk[:, i] = cond.to(dtype)
        for k in range(1, degree + 1):
            Bk_next = torch.zeros_like(Bk)
            for i in range(M):
                term1 = ((grid - knots[i]) / (knots[i + k] - knots[i]).clamp_min(1e-8)) * Bk[:, i] if (knots[i + k] -
                                                                                                       knots[
                                                                                                           i]) > 0 else 0.0
                term2 = ((knots[i + k + 1] - grid) / (knots[i + k + 1] - knots[i + 1]).clamp_min(1e-8)) * Bk[:,
                                                                                                          i + 1] if i + 1 < M and (
                            knots[i + k + 1] - knots[i + 1]) > 0 else 0.0
                Bk_next[:, i] = term1 + term2
            Bk = Bk_next
        return Bk


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
        self.entropy_extractor = RCMHCREExtractor(cfg.entropy_max_scale)
        self.enc_s = nn.Sequential(nn.Linear(scalar_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d))
        self.basis = ISplineBasis(cfg.n_basis, cfg.degree, cfg.n_grid)
        self.c0 = nn.Parameter(torch.randn(cfg.n_basis))
        self.res_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_basis))

    def forward(self, v, s, t_norm):
        v_squeezed = v.squeeze(1)
        entropy_feats = self.entropy_extractor(v_squeezed)
        combined_scalars = torch.cat([s, entropy_feats], dim=1)
        h = self.enc_s(combined_scalars)
        t_norm = t_norm.clamp(0.0, 1.0)
        B_inc = self.basis.eval(t_norm)
        c0_pos = F.softplus(self.c0)
        S_main = (B_inc @ c0_pos) / self.cfg.n_basis
        c_h = F.softplus(self.res_head(h))
        S_res = (B_inc * c_h).sum(dim=-1) / self.cfg.n_basis
        gamma = 1.0
        S = S_main + gamma * S_res
        Q = torch.exp(-S)
        return Q, {"S_main": S_main, "S_res": S_res, "gamma": gamma, "c_h": c_h}


# ==========================================
# 4. 数据集与加载 (不变)
# ==========================================
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

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.sequences[idx]), torch.from_numpy(self.scalars[idx]),
                torch.tensor(self.cycle_indices[idx], dtype=torch.long),
                torch.tensor(self.targets[idx], dtype=torch.float32))


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
        except Exception:
            continue

    if not all_battery_data: raise ValueError("No data loaded.")
    full_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '最大容量(Ah)'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    config.cycle_norm_min = float(train_df['循环号'].min())
    config.cycle_norm_max = float(train_df['循环号'].max())
    if config.cycle_norm_max <= config.cycle_norm_min: config.cycle_norm_max = config.cycle_norm_min + 1.0

    scaler_seq, scaler_scalar = StandardScaler(), StandardScaler()
    scaler_seq.fit(np.vstack(train_df[sequence_col].values))
    scaler_scalar.fit(train_df[scalar_feature_cols])

    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])
        df[target_col] = df[target_col].astype(float) / config.cap_norm

    return (BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col),
            BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col),
            BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col),
            {'sequence': scaler_seq, 'scalar': scaler_scalar})


# ==========================================
# 5. 训练与评估 (不变)
# ==========================================
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
    all_preds, all_labels = [], []
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

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    metrics = {'RMSE': np.sqrt(mean_squared_error(labels, predictions)), 'R2': r2_score(labels, predictions)}
    return avg_loss, metrics, predictions, labels, None


# ==========================================
# 6. 网格搜索执行器 (单组参数逻辑)
# ==========================================
def run_experiment(config: Config, param_combo: dict):
    """运行一组超参数的多次重复实验"""
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    os.makedirs(config.save_path, exist_ok=True)

    NUM_REPEATS = 3  # 每组参数跑3次取平均
    metrics_buffer = []
    print(f"--> 开始测试参数组合: {param_combo}")

    for run_idx in range(1, NUM_REPEATS + 1):
        set_seed(2025 + run_idx * 100)
        run_dir = os.path.join(config.save_path, f'run_{run_idx}')
        os.makedirs(run_dir, exist_ok=True)

        try:
            train_ds, val_ds, test_ds, scalers = load_and_preprocess_data(config)
        except Exception as e:
            print(f"    数据加载错误: {e}")
            return {'avg_rmse': 999.0, 'avg_r2': 0.0}

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

        # --- 关键: 使用 Config 中的参数初始化模型 ---
        model_cfg = ModelConfig(
            d_model=config.d_model, dropout=config.dropout,
            n_basis=config.n_basis, degree=config.degree,
            n_grid=config.n_grid, residual_l2=config.residual_l2,
            entropy_max_scale=config.entropy_max_scale
        )
        model = RCMHCRE_PIRes(scalar_dim=config.scalar_feature_dim, cfg=model_cfg).to(config.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(config.epochs):
            train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler, config)
            val_loss, _, _, _, _ = evaluate(model, val_loader, criterion, config.device, config)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    break

        # Test
        model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
        _, _, test_preds, test_labels, _ = evaluate(model, test_loader, criterion, config.device, config)

        # 反归一化
        test_preds_orig = np.clip(test_preds * config.cap_norm, 0, None)
        test_labels_orig = test_labels * config.cap_norm
        rmse = np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig))
        r2 = r2_score(test_labels_orig, test_preds_orig)
        metrics_buffer.append({'rmse': rmse, 'r2': r2})
        print(f"    Run {run_idx}: RMSE={rmse:.4f}")

    avg_metrics = {
        'avg_rmse': np.mean([m['rmse'] for m in metrics_buffer]),
        'avg_r2': np.mean([m['r2'] for m in metrics_buffer])
    }
    print(f"    平均 RMSE: {avg_metrics['avg_rmse']:.6f}")
    return avg_metrics


# ==========================================
# 7. 主函数
# ==========================================
def main():
    keys, values = zip(*GRID_SEARCH_PARAMS.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"{'=' * 50}\n开始 RCMHCRE 网格搜索 | 共 {len(param_combinations)} 组\n{'=' * 50}")

    results = []
    for i, params in enumerate(param_combinations):
        print(f"进度: {i + 1}/{len(param_combinations)}")
        config = Config(**params)
        metrics = run_experiment(config, params)

        record = params.copy()
        record.update(metrics)
        record['param_str'] = config.param_str
        results.append(record)

        pd.DataFrame(results).to_csv(os.path.join(config.base_save_path, 'grid_search_running.csv'), index=False)

    final_df = pd.DataFrame(results)
    best_record = final_df.sort_values(by='avg_rmse', ascending=True).iloc[0]
    print(
        f"\n{'=' * 50}\n网格搜索完成!\n最佳 RMSE: {best_record['avg_rmse']:.6f}\n最佳参数: {best_record['param_str']}\n{'=' * 50}")
    final_df.to_csv(os.path.join(config.base_save_path, 'grid_search_final.csv'), index=False)


if __name__ == '__main__':
    main()