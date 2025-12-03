"""
基于 Temporal Pattern Attention + CNN-LSTM 的 SOH 估计模型 - 网格搜索版 (Grid Search Version)
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
from typing import Tuple, Dict
import torch.nn.functional as F
import itertools  # 新增: 用于生成参数组合

# ==========================================
# 0. 网格搜索超参数空间定义
# ==========================================
GRID_SEARCH_PARAMS = {
    # --- 这里的 Key 必须与 Config 类中的属性名一致 ---

    # 卷积核数量
    'cnn_channels': [16, 32, 64],

    # LSTM 隐藏层维度
    'lstm_hidden': [64, 128, 256],

    # LSTM 层数
    'lstm_layers': [1, 2],

    # Dropout 比率
    'dropout': [0.1, 0.2, 0.3],

    # 学习率
    'learning_rate': [0.001, 0.0005],

    # 输出层激活函数
    'out_activation': ['sigmoid']  # 如果想对比，可以加 'softplus'
}


# ==========================================
# 1. 配置参数类 (已修改支持动态传参)
# ==========================================
class Config:
    def __init__(self, **kwargs):
        # --- 数据和路径设置 (固定) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'

        # 基础保存路径
        self.base_save_path = '/home/scuee_user06/myh/电池/result-累计放电容量V2.0_correct/CNN-LSTM-TPA/grid_search_result'

        # --- 电池分组 (固定) ---
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
        self.cnn_channels = 64
        self.lstm_hidden = 256
        self.lstm_layers = 2
        self.dropout = 0.2
        self.out_activation = 'sigmoid'

        # --- 训练参数 (默认值) ---
        self.epochs = 300  # 网格搜索时通常可以适当减少 epoch，或者配合 Early Stopping
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.patience = 15  # 由于是搜索，Patience 可以稍微大一点或保持不变
        self.mode = 'both'  # train & validate

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.cycle_norm_min = None
        self.cycle_norm_max = None

        # --- 核心逻辑: 使用 kwargs 覆盖默认值 ---
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # --- 生成参数标识字符串，用于文件夹命名 ---
        # 例如: cnn_channels-32_lstm_hidden-128 ...
        self.param_str = "_".join([f"{k}-{v}" for k, v in kwargs.items()])
        if not self.param_str:
            self.param_str = "default_params"

        # 最终保存路径: base_path / param_str
        self.save_path = os.path.join(self.base_save_path, self.param_str)


# ==========================================
# 2. 工具函数 (不变)
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
# 3. 模型定义 (不变)
# ==========================================
class MultiKernelCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: Tuple[int, ...], p: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            padding = k - 1
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(p)
            ))
        self.merge_conv = nn.Conv1d(out_ch * len(kernels), out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for conv in self.convs:
            feat = conv(x)
            features.append(feat[..., :x.size(-1)])
        cat_features = torch.cat(features, dim=1)
        return self.merge_conv(cat_features)


class TemporalPatternAttention(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1)
        )

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_net(H).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        context = torch.sum(H * weights.unsqueeze(-1), dim=1)
        return context, weights


class Model(nn.Module):
    def __init__(self, cnn_channels: int = 64, lstm_hidden: int = 64, lstm_layers: int = 1,
                 dropout: float = 0.1, out_activation: str = 'sigmoid'):
        super().__init__()
        self.out_activation = out_activation
        self.cnn = MultiKernelCNN(in_ch=4, out_ch=cnn_channels, kernels=(3, 5), p=dropout)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            dropout=0.0 if lstm_layers == 1 else dropout, bidirectional=False, batch_first=True)
        self.tpa = TemporalPatternAttention(d=lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    @staticmethod
    def _assemble_input(v, s, t_norm):
        B, _, T = v.shape
        s_exp = s.unsqueeze(-1).repeat(1, 1, T)
        t_exp = t_norm.unsqueeze(-1).repeat(1, 1, T)
        x = torch.cat([v, s_exp, t_exp], dim=1)
        return x

    def forward(self, v, s, t_norm):
        x = self._assemble_input(v, s, t_norm)
        Hc = self.cnn(x)
        Hseq = Hc.transpose(1, 2)
        Hlstm, _ = self.lstm(Hseq)
        Hlstm = self.dropout(Hlstm)
        context, alpha = self.tpa(Hlstm)
        Q = self.head(context)
        if self.out_activation == 'sigmoid':
            Q = torch.sigmoid(Q)
        elif self.out_activation == 'softplus':
            Q = F.softplus(Q)
        return Q, {'attn_weights_time': alpha}


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
        return (torch.from_numpy(self.sequences[idx]),
                torch.from_numpy(self.scalars[idx]),
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
    if config.cycle_norm_max <= config.cycle_norm_min: config.cycle_norm_max = config.cycle_norm_min + 1.0

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_seq.fit(np.vstack(train_df[sequence_col].values))
    scaler_scalar.fit(train_df[scalar_feature_cols])

    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])
        df[target_col] = df[target_col].astype(float) / config.cap_norm

    return (BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col),
            BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col),
            BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col),
            {'sequence': scaler_seq, 'scalar': scaler_scalar,
             'cycle_norm': {'min': config.cycle_norm_min, 'max': config.cycle_norm_max}})


# ==========================================
# 5. 训练与评估函数 (不变)
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
        t_norm = torch.clamp(t_norm, 0.0, 1.0).unsqueeze(1)

        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                outputs, _ = model(batch_seq, batch_scalar, t_norm)
                loss = criterion(outputs.squeeze(-1), batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs, _ = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs.squeeze(-1), batch_y)
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
            t_norm = torch.clamp(t_norm, 0.0, 1.0).unsqueeze(1)

            outputs, _ = model(batch_seq, batch_scalar, t_norm)
            loss = criterion(outputs.squeeze(-1), batch_y)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            all_cycle_indices.append(batch_cycle_idx.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    cycle_indices = np.concatenate(all_cycle_indices).flatten()
    metrics = {'MSE': mean_squared_error(labels, predictions),
               'MAE': mean_absolute_error(labels, predictions),
               'RMSE': np.sqrt(mean_squared_error(labels, predictions)),
               'R2': r2_score(labels, predictions)}
    return avg_loss, metrics, predictions, labels, cycle_indices


def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Pred', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title);
    plt.legend();
    plt.grid(True);
    plt.savefig(save_path);
    plt.close()


def plot_diagonal_results(labels, preds, title, save_path):
    plt.figure(figsize=(8, 8))
    min_v, max_v = min(np.min(labels), np.min(preds)) * 0.98, max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6);
    plt.plot([min_v, max_v], [min_v, max_v], 'r--')
    plt.title(title);
    plt.grid(True);
    plt.axis('equal');
    plt.xlim(min_v, max_v);
    plt.ylim(min_v, max_v)
    plt.savefig(save_path);
    plt.close()


# ==========================================
# 6. 网格搜索执行器 (单组参数运行逻辑)
# ==========================================
def run_experiment(config: Config, param_combo: dict):
    """运行特定超参数组合的一组实验(多次重复求平均)"""
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    os.makedirs(config.save_path, exist_ok=True)

    # --- 设置该组参数的重复次数 ---
    # 网格搜索时为了速度，可以设为 3 次取平均；如果时间充裕可设为 5
    NUM_REPEATS = 3

    metrics_buffer = []
    print(f"--> 开始测试参数组合: {param_combo}")
    print(f"    结果保存至: {config.save_path}")

    for run_idx in range(1, NUM_REPEATS + 1):
        current_seed = 2025 + run_idx * 100  # 简单的种子策略
        set_seed(current_seed)
        run_dir = os.path.join(config.save_path, f'run_{run_idx}')
        os.makedirs(run_dir, exist_ok=True)

        # 加载数据
        train_ds, val_ds, test_ds, scalers = load_and_preprocess_data(config)
        joblib.dump(scalers, os.path.join(run_dir, 'scalers.pkl'))
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

        # 初始化模型 (使用 Config 中的动态参数)
        model = Model(
            cnn_channels=config.cnn_channels,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
            out_activation=config.out_activation
        ).to(config.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

        # 训练循环
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

        # 测试最佳模型
        model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pth')))
        _, _, test_preds, test_labels, _ = evaluate(model, test_loader, criterion, config.device, config)

        # 反归一化并计算指标
        test_preds_orig = np.clip(test_preds * config.cap_norm, 0, None)
        test_labels_orig = test_labels * config.cap_norm

        rmse = np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig))
        mae = mean_absolute_error(test_labels_orig, test_preds_orig)
        r2 = r2_score(test_labels_orig, test_preds_orig)

        metrics_buffer.append({'rmse': rmse, 'mae': mae, 'r2': r2})
        # 可选: 保存图表 (为了节省空间，这里只在第一次运行时保存)
        if run_idx == 1:
            plot_results(test_labels_orig, test_preds_orig, f'Run 1 Test Result',
                         os.path.join(run_dir, 'test_plot.png'))

    # 计算平均指标
    avg_metrics = {
        'avg_rmse': np.mean([m['rmse'] for m in metrics_buffer]),
        'avg_mae': np.mean([m['mae'] for m in metrics_buffer]),
        'avg_r2': np.mean([m['r2'] for m in metrics_buffer])
    }
    print(f"    组合完成. 平均 RMSE: {avg_metrics['avg_rmse']:.6f}, R2: {avg_metrics['avg_r2']:.4f}")
    return avg_metrics


# ==========================================
# 7. 主函数: 网格搜索调度
# ==========================================
def main():
    # 1. 生成参数组合
    keys, values = zip(*GRID_SEARCH_PARAMS.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"{'=' * 50}")
    print(f"开始网格搜索 | 总共有 {len(param_combinations)} 种参数组合")
    print(f"{'=' * 50}\n")

    results = []

    # 2. 遍历所有组合
    for i, params in enumerate(param_combinations):
        print(f"进度: {i + 1}/{len(param_combinations)}")

        # 实例化配置
        config = Config(**params)

        # 运行实验
        metrics = run_experiment(config, params)

        # 记录结果
        record = params.copy()
        record.update(metrics)
        record['param_folder_name'] = config.param_str
        results.append(record)

        # 实时保存 (防止中断)
        current_df = pd.DataFrame(results)
        current_df.to_csv(os.path.join(config.base_save_path, 'grid_search_running.csv'), index=False)

    # 3. 结束处理
    final_df = pd.DataFrame(results)
    # 按 RMSE 排序找最好的
    best_record = final_df.sort_values(by='avg_rmse', ascending=True).iloc[0]

    print(f"\n{'=' * 50}")
    print("网格搜索全部完成！")
    print(f"最佳 RMSE: {best_record['avg_rmse']:.6f}")
    print("最佳参数组合:")
    for k in GRID_SEARCH_PARAMS.keys():
        print(f"  {k}: {best_record[k]}")
    print(f"{'=' * 50}")

    final_df.to_csv(os.path.join(config.base_save_path, 'grid_search_final_results.csv'), index=False)


if __name__ == '__main__':
    main()