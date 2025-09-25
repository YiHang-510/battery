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
from typing import Tuple, Dict
import torch.nn.functional as F

# --- 1. 配置参数 (已修改为新模型) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        # --- 修改: 更新保存路径以反映新模型 ---
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/消融-ExpNetTR-最大容量/20'

        # self.train_batteries = [1, 2, 3, 6]
        # self.val_batteries = [5]
        # self.test_batteries = [4]

        # self.train_batteries = [7, 8, 9, 11]
        # self.val_batteries = [10]
        # self.test_batteries = [12]

        # self.train_batteries = [15, 16, 17, 18]
        # self.val_batteries = [13]
        # self.test_batteries = [14]

        self.train_batteries = [21, 22, 23, 24]
        self.val_batteries = [19]
        self.test_batteries = [20]

        # --- 模型无关的数据加载参数 (保留以兼容数据加载器) ---
        self.features_from_C = ['恒压充电时间(s)', '3.3~3.6V充电时间(s)']
        self.sequence_feature_dim = 7
        self.sequence_length = 1
        self.cap_norm = 3.5

        # --- 模型超参数 (ExpNetTR) ---
        self.n_terms = 16  # 指数项数量
        self.n_bumps = 8   # 高斯凸起数量
        self.use_logspace_tau = True # tau参数是否使用对数空间初始化

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


# --- 3. 新的预测网络 (ExpNetTR) ---
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

        # ---- Trend: Mixture of exponentials ----
        self.raw_alpha = nn.Parameter(0.01 * torch.randn(n_terms))
        if use_logspace_tau:
            max_log = 4.5
            init_tau = torch.exp(torch.linspace(-2.5, max_log, steps=n_terms))
            self.raw_tau = nn.Parameter(torch.log(init_tau) + 0.01 * torch.randn(n_terms))
        else:
            self.raw_tau = nn.Parameter(torch.randn(n_terms) * 0.1)
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))
        self.trend_bias = nn.Parameter(torch.tensor(0.8) + 0.01 * torch.randn(()))
        self.trend_gain = nn.Parameter(torch.tensor(-0.5) + 0.01 * torch.randn(()))

        # ---- Residual: Gaussian bumps ----
        n_head = max(2, int(self.n_bumps * 0.35))
        n_mid = max(1, int(self.n_bumps * 0.20))
        n_tail = self.n_bumps - n_head - n_mid
        mu_head = torch.exp(torch.linspace(math.log(1e-3), math.log(0.15), steps=n_head))
        mu_mid = torch.linspace(0.15, 0.70, steps=n_mid)
        mu_tail = torch.linspace(0.70, 1.02, steps=n_tail)
        self.mu = nn.Parameter(torch.cat([mu_head, mu_mid, mu_tail]))
        self.raw_sigma = nn.Parameter(torch.cat([
            torch.full((n_head,), -2.3),
            torch.full((n_mid,), -1.3),
            torch.full((n_tail,), -2.0),
        ]))
        self.raw_beta = nn.Parameter(0.01 * torch.randn(n_bumps))
        self.raw_res_scale = nn.Parameter(torch.tensor(-2.0))
        self.input_shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, c, return_components=False):
        c = c.view(-1, 1)
        c_ = c - self.input_shift

        # Trend
        alpha = F.softmax(self.raw_alpha, dim=0)
        tau = torch.exp(self.raw_tau).clamp_max(80.0)
        gamma = F.softplus(self.raw_gamma) + 1e-6
        expo = torch.exp(-(c_ * gamma) @ tau.view(1, -1))
        mix = (expo * alpha.view(1, -1)).sum(dim=1, keepdim=True)
        trend = self.trend_bias + self.trend_gain * mix

        # Residual
        sigma = F.softplus(self.raw_sigma) + 1e-6
        beta = torch.tanh(self.raw_beta)
        res_scale = torch.sigmoid(self.raw_res_scale)
        gauss = torch.exp(-0.5 * ((c_ - self.mu.view(1, -1)) / sigma.view(1, -1)) ** 2)
        residual = res_scale * (gauss * beta.view(1, -1)).sum(dim=1, keepdim=True)

        y = (trend + residual).view(-1)

        if not return_components:
            return y
        else:
            return y, {
                "trend": trend.view(-1),
                "residual": residual.view(-1)
            }


# --- 4. 数据集定义 (已修正) ---
class BatteryMultimodalDataset(Dataset):
    def __init__(self, dataframe, sequence_col, scalar_cols, target_col):
        self.df = dataframe.reset_index(drop=True)
        self.target_col = target_col
        # The following are loaded but will be ignored by the ExpNetTR model
        self.sequences = np.array(self.df[sequence_col].tolist(), dtype=np.float32)
        # --- 此处已修正 ---
        self.scalars = self.df[scalar_cols].values.astype(np.float32)
        # These are the only two fields used by the model
        self.targets = self.df[self.target_col].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # We still return all items to keep the interface consistent
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)
        return x_seq, x_scalar, cycle_idx, y


# --- 5. 数据加载和预处理 (不变) ---
def load_and_preprocess_data(config):
    # This function remains unchanged as the dataset class handles the data structure.
    # The training loop will simply ignore the unneeded data.
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

    full_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '最大容量(Ah)'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    config.cycle_norm_min = float(train_df['循环号'].min())
    config.cycle_norm_max = float(train_df['循环号'].max())

    # We only need to normalize the target
    for df in [train_df, val_df, test_df]:
        df[target_col] = df[target_col].astype(float) / config.cap_norm

    # Scalers are not needed for this model but we keep the structure
    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col)
    scalers = {'cycle_norm': {'min': config.cycle_norm_min, 'max': config.cycle_norm_max}}
    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练/验证函数 (已修改以适配新模型) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler, config):
    model.train()
    total_loss = 0
    # Note: batch_seq and batch_scalar are loaded but ignored
    for _, _, batch_cycle_idx, batch_y in dataloader:
        batch_cycle_idx, batch_y = batch_cycle_idx.to(device), batch_y.to(device)

        t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
        t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
        t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min).clamp_min(1.0)
        t_norm = torch.clamp(t_norm, 0.0, 1.0)

        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                # --- 修改: 模型只接收 t_norm ---
                outputs = model(t_norm)
                loss = criterion(outputs, batch_y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs = model(t_norm)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_cycle_indices = [], [], []
    with torch.no_grad():
        for _, _, batch_cycle_idx, batch_y in dataloader:
            batch_cycle_idx, batch_y = batch_cycle_idx.to(device), batch_y.to(device)

            t_min = torch.tensor(config.cycle_norm_min, device=device, dtype=torch.float32)
            t_max = torch.tensor(config.cycle_norm_max, device=device, dtype=torch.float32)
            t_norm = (batch_cycle_idx.float() - t_min) / (t_max - t_min).clamp_min(1.0)
            t_norm = torch.clamp(t_norm, 0.0, 1.0)

            # --- 修改: 模型只接收 t_norm ---
            outputs = model(t_norm)
            loss = criterion(outputs, batch_y)

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


# --- 7. 可视化函数 (新增可解释性绘图) ---
def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predictions', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_learned_curve(model, test_points_x, test_points_y, title, save_path, device, cap_norm):
    model.eval()
    with torch.no_grad():
        # Generate a dense grid for plotting the continuous curve
        t_grid = torch.linspace(min(test_points_x), max(test_points_x), 400).to(device)
        y_curve, components = model(t_grid, return_components=True)

        # Un-normalize for plotting
        y_curve = y_curve.cpu().numpy() * cap_norm
        trend_curve = components['trend'].cpu().numpy() * cap_norm
        residual_curve = components['residual'].cpu().numpy() * cap_norm
        t_grid_numpy = t_grid.cpu().numpy()

        plt.figure(figsize=(12, 8))
        # Plot true data points
        plt.scatter(test_points_x, test_points_y, color='red', label='True Data', s=20, zorder=5)
        # Plot learned curves
        plt.plot(t_grid_numpy, y_curve, color='blue', label='Total Fit (Trend + Residual)', linewidth=2.5)
        plt.plot(t_grid_numpy, trend_curve, color='green', linestyle='--', label='Learned Trend', linewidth=2)
        # Plot residual on a shifted baseline for visibility
        plt.fill_between(t_grid_numpy, trend_curve, trend_curve + residual_curve, color='orange', alpha=0.5, label='Residual Component')

        plt.title(title, fontsize=16)
        plt.xlabel('Normalized Cycle Number', fontsize=12)
        plt.ylabel('Capacity (Ah)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.savefig(save_path, dpi=300)
        plt.close()

# --- 8. 主执行函数 (已更新) ---
def main():
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"所有实验的总保存路径: {config.save_path}")

    num_runs = 5
    all_runs_metrics, all_runs_PER_BATTERY_metrics = [], []

    for run_number in range(1, num_runs + 1):
        current_seed = random.randint(0, 99999)
        set_seed(current_seed)
        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)
        print(f"\n{'=' * 30}\n 开始第 {run_number}/{num_runs} 次实验 | 随机种子: {current_seed} \n{'=' * 30}")

        train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        print(f"数据加载完成。")

        # --- 修改: 初始化新模型 ---
        model = ExpNetTR(
            n_terms=config.n_terms,
            n_bumps=config.n_bumps,
            use_logspace_tau=config.use_logspace_tau
        ).to(config.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

        best_val_loss_this_run = float('inf')
        epochs_no_improve = 0

        print("\n开始训练模型...")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler, config)
            val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, config.device, config)

            print(f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")

            if val_loss < best_val_loss_this_run:
                best_val_loss_this_run = val_loss
                torch.save(model.state_dict(), os.path.join(run_save_path, 'best_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"提前停止训练。")
                    break

        print('\n加载本轮最佳模型进行最终评估...')
        model.load_state_dict(torch.load(os.path.join(run_save_path, 'best_model.pth'), map_location=config.device))
        _, _, test_preds, test_labels, test_cycle_nums = evaluate(model, test_loader, criterion, config.device, config)

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

        # --- 新增: 为每个测试电池绘制可解释性曲线图 ---
        print("\n正在生成可解释性曲线图...")
        eval_df = pd.DataFrame({
            'battery_id': test_dataset.df['battery_id'].values,
            'cycle': test_cycle_nums,
            'true_norm': test_labels,
            'pred_norm': test_preds
        })
        for batt_id in sorted(list(set(config.test_batteries))):
            batt_df = eval_df[eval_df['battery_id'] == batt_id]
            if batt_df.empty: continue

            # Normalize cycle numbers for this battery
            cycle_min = scalers['cycle_norm']['min']
            cycle_max = scalers['cycle_norm']['max']
            batt_x_norm = (batt_df['cycle'].values - cycle_min) / (cycle_max - cycle_min)
            batt_y_orig = batt_df['true_norm'].values * config.cap_norm

            plot_learned_curve(model, batt_x_norm, batt_y_orig,
                               title=f'Battery {batt_id}: Learned Degradation Curve',
                               save_path=os.path.join(run_save_path, f'interpret_plot_battery_{batt_id}.png'),
                               device=config.device, cap_norm=config.cap_norm)
        print("绘图完成。")


    print(f"\n\n{'=' * 50}\n 所有实验均已完成。\n{'=' * 50}")
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 ---\n", summary_df.drop(columns=['run', 'seed']).mean())
        print(f"\n汇总指标已保存到: {summary_path}")


if __name__ == '__main__':
    main()
