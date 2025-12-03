import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib
import warnings
from dataclasses import dataclass
from typing import Tuple
import torch.nn.functional as F


# --- 1. 验证配置 (请仔细修改这里的路径) ---
class ValidationConfig:
    """
    用于模型验证的配置参数。
    请确保这里的模型结构参数与您训练时使用的参数完全一致！
    """

    def __init__(self):
        # --- ▼▼▼【核心配置】请务必修改以下路径 ▼▼▼ ---

        # 1. 模型路径: 指向您训练好的 'best_model.pth' 文件
        self.model_path = '/home/scuee_user06/myh/电池/result-累计放电容量/TM_PIRes/20/best_model.pth'

        # 2. Scaler路径: 指向与模型配对的 'scalers.pkl' 文件
        self.scalers_path = '/home/scuee_user06/myh/电池/result-累计放电容量/TM_PIRes/20/scalers.pkl'

        # 3. 结果输出路径: 指定验证结果CSV文件的保存位置和文件名
        self.output_csv_path = '/home/scuee_user06/myh/电池/result-累计放电容量/TM_PIRes/20/validation_results.csv'

        # --- ▲▲▲【核心配置】请务必修改以上路径 ▲▲▲ ---

        # --- 数据路径和定义 ---
        # 指定包含验证数据的文件夹
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        # 指定要验证的电池ID列表 (可以和训练时的 test_batteries 一样，也可以是任何您想验证的电池)
        self.validation_batteries = [20]  # 您可以修改为 [6, 12, 14, 20] 或其他

        # --- 特征定义 (必须与训练时一致) ---
        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]
        self.sequence_feature_dim = 7
        self.sequence_length = 1

        # --- 模型超参数 (必须与训练时一致) ---
        self.d_model = 128
        self.n_blocks = 3
        self.kernel_sizes = (3, 5, 7)
        self.dropout = 0.1
        self.n_basis = 10
        self.degree = 3
        self.n_grid = 512
        self.residual_l2 = 1e-4

        # --- 运行参数 ---
        self.batch_size = 128
        self.seed = 2025
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.scalar_feature_dim = len(self.features_from_C)


# --- 2. 固定随机种子 (保持不变) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 模型定义 (从训练脚本完整复制，确保结构一致) ---

# B-spline / I-spline Basis Functions
def _bspline_basis_grid(knots: torch.Tensor, degree: int, grid: torch.Tensor) -> torch.Tensor:
    device, dtype, G, M = knots.device, knots.dtype, grid.numel(), knots.numel() - degree - 1
    assert M > 0, "n_basis must be > degree."
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
            if knots[i + k] - knots[i] > 0:
                term1 = ((grid - knots[i]) / (knots[i + k] - knots[i])) * Bk[:, i]
            term2 = 0.0
            if i + 1 < M and knots[i + k + 1] - knots[i + 1] > 0:
                term2 = ((knots[i + k + 1] - grid) / (knots[i + k + 1] - knots[i + 1])) * Bk[:, i + 1]
            Bk_next[:, i] = term1 + term2
        Bk = Bk_next
    return Bk


class ISplineBasis(nn.Module):
    def __init__(self, n_basis: int = 10, degree: int = 3, n_grid: int = 512, device=None, dtype=None):
        super().__init__()
        assert n_basis > degree
        self.n_basis, self.degree, self.n_grid = n_basis, degree, n_grid
        device, dtype = device or torch.device("cpu"), dtype or torch.float32
        n_knots = n_basis + degree + 1
        interior = max(0, n_knots - 2 * (degree + 1))
        interior_knots = torch.linspace(0.0, 1.0, interior + 2, dtype=dtype, device=device)[
                         1:-1] if interior > 0 else torch.empty(0, dtype=dtype, device=device)
        knots = torch.cat([torch.zeros(degree + 1, dtype=dtype, device=device), interior_knots,
                           torch.ones(degree + 1, dtype=dtype, device=device)])
        grid = torch.linspace(0.0, 1.0, n_grid, dtype=dtype, device=device)
        B_grid = _bspline_basis_grid(knots, degree, grid)
        M_grid = (degree + 1) * B_grid / (knots[degree + 1: degree + 1 + n_basis] - knots[:n_basis]).clamp_min(1e-8)
        I_grid = torch.zeros_like(M_grid)
        I_grid[1:, :] = torch.cumsum(0.5 * (M_grid[1:, :] + M_grid[:-1, :]) * (1.0 / (n_grid - 1)), dim=0)
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
        return (1.0 - w) * self.I_grid[i0.squeeze(-1), :] + w * self.I_grid[i1.squeeze(-1), :]


# TimeMixer Components
class DepthwiseTemporalMix(nn.Module):
    def __init__(self, d_model: int, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model) for k in kernel_sizes])
        self.proj, self.alpha, self.act, self.drop = nn.Conv1d(d_model, d_model, 1), nn.Parameter(
            torch.zeros(len(kernel_sizes))), nn.SiLU(), nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t, a = x.transpose(1, 2), F.softmax(self.alpha, dim=0)
        y = sum(a[i] * self.act(conv(x_t)) for i, conv in enumerate(self.convs))
        return self.drop(self.proj(y)).transpose(1, 2)


class ChannelMix(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * expansion), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout))

    def forward(self, x): return self.ff(x)


class TimeMixerBlock(nn.Module):
    def __init__(self, d_model: int, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.norm1, self.tmix, self.norm2, self.cmix = nn.LayerNorm(d_model), DepthwiseTemporalMix(d_model,
                                                                                                   kernel_sizes,
                                                                                                   dropout), nn.LayerNorm(
            d_model), ChannelMix(d_model, 4, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tmix(self.norm1(x)) + self.cmix(self.norm2(x + self.tmix(self.norm1(x))))


class TimeMixerStack(nn.Module):
    def __init__(self, d_model: int, n_blocks: int, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TimeMixerBlock(d_model, kernel_sizes, dropout) for _ in range(n_blocks)])

    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return x


# Main Model: TM_PIRes
@dataclass
class TMPIResConfig:
    d_model: int = 128;
    n_blocks: int = 3;
    kernel_sizes: Tuple[int, ...] = (3, 5, 7);
    dropout: float = 0.1;
    n_basis: int = 10;
    degree: int = 3;
    n_grid: int = 512;
    residual_l2: float = 1e-4


class TM_PIRes(nn.Module):
    def __init__(self, seq_channels: int, scalar_dim: int, cfg: TMPIResConfig):
        super().__init__()
        self.cfg, d = cfg, cfg.d_model
        self.embed, self.tm = nn.Linear(seq_channels, d), TimeMixerStack(d, cfg.n_blocks, cfg.kernel_sizes, cfg.dropout)
        self.enc_s = nn.Sequential(nn.Linear(scalar_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d))
        self.basis, self.c0, self.b0 = ISplineBasis(cfg.n_basis, cfg.degree, cfg.n_grid), nn.Parameter(
            torch.randn(cfg.n_basis)), nn.Parameter(torch.zeros(1))
        self.res_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_basis))

    def forward(self, v: torch.Tensor, s: torch.Tensor, t_norm: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        H = self.tm(self.embed(v))
        h = H.mean(dim=1) + self.enc_s(s)
        B_I = self.basis.eval(t_norm)
        m = (B_I @ F.softplus(self.c0)) + self.b0
        c_h = F.softplus(self.res_head(h))
        R = (B_I * c_h).sum(dim=-1)
        return m + R, {"c_h": c_h}


# --- 4. 数据集定义 (从训练脚本复制) ---
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
        self.battery_ids = self.df['battery_id'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.from_numpy(self.scalars[idx]),
            torch.tensor(self.cycle_indices[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.battery_ids[idx], dtype=torch.long)
        )


# --- 5. 数据加载和预处理 (为验证过程定制) ---
def load_validation_data(config, scalers):
    """加载并使用已有的scalers来处理验证数据"""
    all_battery_data = []
    for battery_id in config.validation_batteries:
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

    if not all_battery_data:
        raise ValueError("未能加载任何指定的验证电池数据。")

    val_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '累计放电容量(Ah)'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C

    # 使用加载的scalers进行转换
    scaler_seq = scalers['sequence']
    scaler_scalar = scalers['scalar']
    val_df[sequence_col] = val_df[sequence_col].apply(lambda x: scaler_seq.transform(x))
    val_df.loc[:, scalar_feature_cols] = scaler_scalar.transform(val_df[scalar_feature_cols])

    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return val_loader, scalers['cycle_norm']


# --- 6. 核心验证函数 ---
def run_validation(model, dataloader, device, cycle_norm_params):
    """在数据集上运行模型并返回预测结果和真实标签"""
    model.eval()
    all_preds, all_labels, all_cycle_indices, all_battery_ids = [], [], [], []

    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y, batch_batt_id in dataloader:
            batch_seq, batch_scalar, batch_cycle_idx = batch_seq.to(device), batch_scalar.to(
                device), batch_cycle_idx.to(device)

            # 归一化循环号
            t_min = torch.tensor(cycle_norm_params['min'], device=device, dtype=torch.float32)
            t_max = torch.tensor(cycle_norm_params['max'], device=device, dtype=torch.float32)
            t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min).clamp_min(1.0)
            t_norm = torch.clamp(t_norm, 0.0, 1.0)

            outputs, _ = model(batch_seq, batch_scalar, t_norm)

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_cycle_indices.append(batch_cycle_idx.cpu().numpy())
            all_battery_ids.append(batch_batt_id.cpu().numpy())

    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    cycle_indices = np.concatenate(all_cycle_indices).flatten()
    battery_ids = np.concatenate(all_battery_ids).flatten()

    return predictions, labels, cycle_indices, battery_ids


# --- 7. 主执行函数 ---
def main():
    warnings.filterwarnings('ignore')
    config = ValidationConfig()
    set_seed(config.seed)

    print("--- 开始验证流程 ---")
    print(f"使用设备: {config.device}")
    print(f"加载模型: {config.model_path}")
    print(f"加载Scalers: {config.scalers_path}")
    print(f"验证电池: {config.validation_batteries}")

    # 1. 加载 Scalers
    try:
        scalers = joblib.load(config.scalers_path)
        print("Scalers 加载成功。")
    except FileNotFoundError:
        print(f"错误: 找不到 Scaler 文件 '{config.scalers_path}'。请确保路径正确。")
        return

    # 2. 初始化模型并加载权重
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

    try:
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
        print("模型权重加载成功。")
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{config.model_path}'。请确保路径正确。")
        return
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        return

    # 3. 加载验证数据
    try:
        val_loader, cycle_norm_params = load_validation_data(config, scalers)
        print(f"验证数据加载完成，共 {len(val_loader.dataset)} 个样本。")
    except (FileNotFoundError, ValueError) as e:
        print(f"加载验证数据时失败: {e}")
        return

    # 4. 运行验证
    predictions, labels, cycle_indices, battery_ids = run_validation(model, val_loader, config.device,
                                                                     cycle_norm_params)
    print("模型预测完成。")

    # 注意：根据您的训练脚本，目标值没有被归一化，所以这里不需要 inverse_transform
    # 如果您的训练脚本对目标值进行了归一化，请取消下面的注释
    # scaler_target = scalers['target']
    # predictions_orig = scaler_target.inverse_transform(predictions.reshape(-1, 1)).flatten()
    # labels_orig = scaler_target.inverse_transform(labels.reshape(-1, 1)).flatten()
    predictions_orig = predictions
    labels_orig = labels

    # 5. 创建并保存结果到CSV
    results_df = pd.DataFrame({
        'Battery_ID': battery_ids,
        'Cycle': cycle_indices,
        'True_Value': labels_orig,
        'Predicted_Value': predictions_orig
    })
    results_df['Difference'] = results_df['True_Value'] - results_df['Predicted_Value']

    # 确保输出目录存在
    output_dir = os.path.dirname(config.output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_df.to_csv(config.output_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n--- 验证完成 ---")
    print(f"结果已成功保存到: {config.output_csv_path}")


if __name__ == '__main__':
    main()