import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib
import warnings
import joblib
from torch.cuda.amp import autocast, GradScaler
from joblib import Parallel, delayed
import shutil  # 导入 shutil 库用于文件操作


# --- 1. 配置参数 (与您原始代码一致) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (修改为三个输入路径) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'  # A文件: 弛豫段电压序列 (1200点/循环)
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'  # C文件: 其他特征和目标 (1行/循环)
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/MLP-123/20'  # 保存模型、结果和图像的文件夹路径

        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 15, 16, 17, 18, 21, 22, 23, 24]
        # self.val_batteries = [5, 11, 13, 19]
        # self.test_batteries = [6, 12, 14, 20]  # 假设这些文件存在

        # self.train_batteries = [1, 2, 3, 4]
        # self.val_batteries = [5]
        # self.test_batteries = [6]

        # self.train_batteries = [7, 8, 9, 11]
        # self.val_batteries = [10]
        # self.test_batteries = [12]

        # self.train_batteries = [15, 16, 17, 18]
        # self.val_batteries = [13]
        # self.test_batteries = [14]
        #
        self.train_batteries = [21, 22, 23, 24]
        self.val_batteries = [19]
        self.test_batteries = [20]

        self.features_from_C = [
            '恒压充电时间(s)',
            '3.3~3.6V充电时间(s)',
        ]

        # 文件A的输入特征维度 (例如，'弛豫段电压1'到'弛豫段电压7'就是7维)
        self.sequence_feature_dim = 7
        self.sequence_length = 1  # 在MLP模型中，这个值决定了序列被压平后的向量长度

        # --- 模型超参数 (为MLP调整) ---
        self.scalar_feature_dim = len(self.features_from_C)
        self.hidden_dim = 256  # 隐藏层维度 (替代 d_model)
        self.ff_dim = 1024  # 预测头的中间层维度 (替代 d_ff)
        self.dropout = 0.2
        self.weight_decay = 0.0001

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.001
        self.patience = 15
        self.seed = 2025
        self.mode = 'both'

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

        # —— 曲线参数化相关（新增） ——
        self.curve_type = 'gompertz'  # 备选: 'gompertz', 'power'
        self.use_residual = True  # 是否启用残差头 r_t
        self.residual_lambda = 1e-3  # L2(r_t) 正则权重（训练时加入 loss）

        # —— 时间(循环号)归一化（训练数据统计得到，load 时填充） ——
        self.cycle_norm_mean = None
        self.cycle_norm_std = None


# --- 2. 固定随机种子 (不变) ---
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 3. 新的多模态模型定义 (MLP 版本) ---
class ParametricCurveModel(nn.Module):
    """
    方案B：编码器 -> 曲线参数 θ -> 解析生成 Q_t (+可选 residual)
    支持:
      - Gompertz：Q(t) = Q0 + Qmax * exp(-exp(-k * (t_norm - t0)))
      - Power   ：Q(t) = Q0 + a * (t_norm_clamped) ** p
    参数约束：
      Q0, Qmax, k, a  用 softplus 保证非负；p 由 sigmoid 映射到 [p_min, p_max]
      t0 为实数不约束
    """
    def __init__(self, configs):
        super().__init__()
        self.cfg = configs

        flattened_seq_dim = configs.sequence_length * configs.sequence_feature_dim
        hid = configs.hidden_dim

        # 1) 编码器（与原 MLP 类似）
        self.sequence_encoder = nn.Sequential(
            nn.Linear(flattened_seq_dim, hid // 2),
            nn.ReLU(), nn.Dropout(configs.dropout)
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(configs.scalar_feature_dim, hid // 2),
            nn.ReLU(), nn.Dropout(configs.dropout)
        )

        # 2) 融合
        fused_dim = hid
        self.fuse = nn.Identity()  # 直接拼接后用预测头

        # 3) 曲线参数头（根据曲线类型输出不同数量的参数）
        if configs.curve_type == 'gompertz':
            # [Q0, Qmax, k, t0] 共4个
            out_dim = 4
        elif configs.curve_type == 'power':
            # [Q0, a, p] 共3个
            out_dim = 3
        else:
            raise ValueError(f"Unsupported curve_type: {configs.curve_type}")

        self.theta_head = nn.Sequential(
            nn.Linear(hid, configs.ff_dim), nn.ReLU(), nn.Dropout(configs.dropout),
            nn.Linear(configs.ff_dim, out_dim)
        )

        # 4) 可选 residual 头（输入再额外拼接 t_norm）
        if configs.use_residual:
            self.residual_head = nn.Sequential(
                nn.Linear(hid + 1, configs.ff_dim // 2),
                nn.ReLU(), nn.Dropout(configs.dropout),
                nn.Linear(configs.ff_dim // 2, 1)
            )
        else:
            self.residual_head = None

        # 记录上一次前向的正则损失分量
        self._last_aux = {}

        # Power 曲线的指数范围（可按经验改）
        self.p_min, self.p_max = 0.5, 1.5

    def forward(self, x_seq, x_scalar, cycle_number):
        # 编码
        x_seq = x_seq.view(x_seq.size(0), -1)              # (B, seq_len*feat_dim)
        h_seq = self.sequence_encoder(x_seq)               # (B, hid/2)
        h_sca = self.scalar_encoder(x_scalar)              # (B, hid/2)
        h = torch.cat([h_seq, h_sca], dim=1)               # (B, hid)
        h = self.fuse(h)

        # t 归一化（用训练集统计的均值/方差）
        # 若 std 很小，避免数值问题
        t = cycle_number.float()
        mu = torch.tensor(self.cfg.cycle_norm_mean, device=t.device, dtype=t.dtype)
        std = torch.tensor(self.cfg.cycle_norm_std,  device=t.device, dtype=t.dtype)
        std = torch.clamp(std, min=1e-6)
        t_norm = (t - mu) / std
        t_norm = t_norm.unsqueeze(-1)  # (B, 1)

        # 预测 θ
        theta_raw = self.theta_head(h)

        if self.cfg.curve_type == 'gompertz':
            # 解析参数
            Q0_raw, Qmax_raw, k_raw, t0 = torch.split(theta_raw, [1, 1, 1, 1], dim=-1)
            Q0   = torch.nn.functional.softplus(Q0_raw)       # >=0
            Qmax = torch.nn.functional.softplus(Qmax_raw)     # >=0
            k    = torch.nn.functional.softplus(k_raw) + 1e-6 # >0
            # 解析式
            q_curve = Q0 + Qmax * torch.exp(-torch.exp(-k * (t_norm - t0)))

        elif self.cfg.curve_type == 'power':
            # 解析参数
            Q0_raw, a_raw, p_raw = torch.split(theta_raw, [1, 1, 1], dim=-1)
            Q0 = torch.nn.functional.softplus(Q0_raw)         # >=0
            a  = torch.nn.functional.softplus(a_raw) + 1e-6   # >0
            p  = torch.sigmoid(p_raw) * (self.p_max - self.p_min) + self.p_min
            # 为稳定性，t_norm 平移/截断到正区间
            t_pos = torch.clamp(t_norm + 3.0, min=1e-3)       # 移到正域，避免 0 ** p
            q_curve = Q0 + a * (t_pos ** p)

        # 可选 residual：小幅校正 + L2 约束（平滑性用L2近似）
        if self.residual_head is not None:
            r_in = torch.cat([h, t_norm], dim=1)              # (B, hid+1)
            r = self.residual_head(r_in)                      # (B, 1)
            q_pred = q_curve + r
            # 记录正则：L2(r)
            self._last_aux['residual_l2'] = (r ** 2).mean()
        else:
            q_pred = q_curve
            self._last_aux['residual_l2'] = torch.tensor(0.0, device=q_pred.device)

        return q_pred  # 与原接口保持一致 (B,1)

# --- 4. 数据集定义 (不变) ---
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)
        return x_seq, x_scalar, cycle_idx, y


# --- 5. 数据加载和预处理 (不变) ---
def load_and_preprocess_data(config):
    """加载来自三个文件夹的数据，进行合并、预处理和划分"""
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries

    for battery_id in sorted(list(set(all_ids))):
        try:
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')

            df_a = pd.read_csv(path_a, sep=',')
            df_c = pd.read_csv(path_c, sep=',')
            df_c.rename(columns=lambda x: x.strip(), inplace=True)
            scalar_df = df_c

            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.sequence_length]

            final_df = pd.merge(sequence_df, scalar_df, on='循环号')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)

        except FileNotFoundError as e:
            print(f"警告: 电池 {battery_id} 的文件未找到，已跳过。错误: {e}")
            continue
        except Exception as e:
            print(f"处理电池 {battery_id} 时出错: {e}")
            continue

    if not all_battery_data:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_battery_data, ignore_index=True)
    target_col = '累计放电容量(Ah)'
    sequence_col = 'voltage_sequence'
    scalar_feature_cols = config.features_from_C

    for col in scalar_feature_cols:
        if col not in full_df.columns:
            raise ValueError(f"您选择的特征 '{col}' 不存在于加载的数据中。")

    config.scalar_feature_dim = len(scalar_feature_cols)
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    config.cycle_norm_mean = float(train_df['循环号'].mean())
    config.cycle_norm_std = float(train_df['循环号'].std(ddof=0))  # population std
    if config.cycle_norm_std < 1e-6:
        config.cycle_norm_std = 1.0

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_target = StandardScaler()

    all_train_sequences = np.vstack(train_df[sequence_col].values)
    scaler_seq.fit(all_train_sequences)
    scaler_scalar.fit(train_df[scalar_feature_cols])
    scaler_target.fit(train_df[[target_col]])

    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])
        df.loc[:, [target_col]] = scaler_target.transform(df[[target_col]])

    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col)
    scalers = {
        'sequence': scaler_seq,
        'scalar': scaler_scalar,
        'target': scaler_target,
        'cycle_norm': {'mean': config.cycle_norm_mean, 'std': config.cycle_norm_std}
    }

    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练函数 (不变) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        batch_seq, batch_scalar, batch_cycle_idx, batch_y = batch_seq.to(device), batch_scalar.to(
            device), batch_cycle_idx.to(device), batch_y.to(device).unsqueeze(-1)
        optimizer.zero_grad()
        if grad_scaler:
            with autocast():
                outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
                loss = criterion(outputs, batch_y)
                # —— 新增：把 residual L2 并入 loss ——
                if hasattr(model, "_last_aux") and 'residual_l2' in model._last_aux:
                    loss = loss + model.cfg.residual_lambda * model._last_aux['residual_l2']
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            loss = criterion(outputs, batch_y)
            if hasattr(model, "_last_aux") and 'residual_l2' in model._last_aux:
                loss = loss + model.cfg.residual_lambda * model._last_aux['residual_l2']
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 7. 验证/测试函数 (不变) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_cycle_indices = [], [], []
    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
            batch_seq, batch_scalar, batch_cycle_idx, batch_y = batch_seq.to(device), batch_scalar.to(
                device), batch_cycle_idx.to(device), batch_y.to(device).unsqueeze(-1)
            outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            loss = criterion(outputs, batch_y)
            # —— 验证/测试时同样计入正则，保证早停一致性 ——
            if hasattr(model, "_last_aux") and 'residual_l2' in model._last_aux:
                loss = loss + model.cfg.residual_lambda * model._last_aux['residual_l2']
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
               'RMSE': np.sqrt(mean_squared_error(labels, predictions)), 'R2': r2_score(labels, predictions)}
    return avg_loss, metrics, predictions, labels, cycle_indices


# --- 8. 可视化和工具函数 (不变) ---
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


# --- 9. 主执行函数 (主要修改在这里) ---
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


        model = ParametricCurveModel(config).to(config.device)


        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

        metrics_log = []
        best_val_loss_this_run = float('inf')
        epochs_no_improve = 0

        if config.mode in ['both', 'train']:
            print("\n开始训练模型...")
            for epoch in range(config.epochs):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)
                val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, config.device)
                print(
                    f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")
                log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                             **{'val_' + k: v for k, v in val_metrics.items()}}
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
            scaler_target = joblib.load(os.path.join(run_save_path, 'scalers.pkl'))['target']
            _, _, test_preds, test_labels, test_cycle_nums = evaluate(model, test_loader, criterion, config.device)
            test_preds_orig = scaler_target.inverse_transform(test_preds.reshape(-1, 1)).flatten()
            test_labels_orig = scaler_target.inverse_transform(test_labels.reshape(-1, 1)).flatten()
            test_preds_orig = np.clip(test_preds_orig, a_min=0.0, a_max=None)

            print("\n--- 本轮评估结果 (按单电池) ---")
            eval_df = pd.DataFrame(
                {'battery_id': test_dataset.df['battery_id'].values, 'cycle': test_cycle_nums, 'true': test_labels_orig,
                 'pred': test_preds_orig})
            per_battery_metrics_list = []
            for batt_id in config.test_batteries:
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
                batt_metrics_with_run_info = {**batt_metrics_dict, 'run': run_number, 'seed': current_seed}
                all_runs_PER_BATTERY_metrics.append(batt_metrics_with_run_info)
                plot_results(batt_true, batt_pred, f'Run {run_number} Battery {batt_id}: True vs Predicted Capacity',
                             os.path.join(run_save_path, f'test_plot_battery_{batt_id}.png'))
                plot_diagonal_results(batt_true, batt_pred, f'Run {run_number} Battery {batt_id}: Diagonal Plot',
                                      os.path.join(run_save_path, f'test_diagonal_plot_battery_{batt_id}.png'))
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
            current_run_summary = {'run': run_number, 'seed': current_seed, **final_test_metrics}
            all_runs_metrics.append(current_run_summary)
            print(
                f"测试集(汇总): MSE={final_test_metrics['MSE']:.6f}, MAE={final_test_metrics['MAE']:.6f}, RMSE={final_test_metrics['RMSE']:.6f}, R2={final_test_metrics['R2']:.4f}")

            if best_val_loss_this_run < best_run_val_loss:
                best_run_val_loss, best_run_dir, best_run_number = best_val_loss_this_run, run_save_path, run_number
                print(f"*** 新的最佳表现！验证集损失: {best_val_loss_this_run:.6f} ***")

    print(f"\n\n{'=' * 50}\n 所有实验均已完成。\n{'=' * 50}")
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
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
        core_cols = ['Battery_ID', 'run', 'seed', 'MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
        ordered_cols = [col for col in core_cols if col in per_batt_summary_df.columns] + [col for col in
                                                                                           per_batt_summary_df.columns
                                                                                           if col not in core_cols]
        per_batt_summary_df = per_batt_summary_df[ordered_cols].sort_values(by=['Battery_ID', 'run'])
        summary_path_per_batt = os.path.join(config.save_path, 'all_runs_per_battery_summary.csv')
        per_batt_summary_df.to_csv(summary_path_per_batt, index=False)
        print(f"“分电池”详细汇总报告已保存到: {summary_path_per_batt}")


if __name__ == '__main__':
    main()