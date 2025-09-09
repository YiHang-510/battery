import os
import random
import re
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


# --- 1. 配置参数 ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/E2E-Fusion/all'  # 端到端训练的全新保存路径

        # --- 数据集划分 ---
        self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 11, 15, 17, 18, 19, 21, 22, 23, 24]
        self.val_batteries = [5, 10, 13, 19]
        self.test_batteries = [6, 12, 14, 20]

        self.nominal_capacity = 3.5  # 用于计算SOH

        # --- CycleNet 子模型配置 ---
        self.cyclenet_config = {
            'sequence_length': 1,
            'sequence_feature_dim': 7,
            'features_from_C': [
                '恒压充电时间(s)',
                '3.3~3.6V充电时间(s)',
            ],
            'scalar_feature_dim': 2,  # 将由 features_from_C 自动更新
            'meta_cycle_len': 7,
            'd_model': 256,
            'd_ff': 1024,
            'dropout': 0.2,
        }

        # --- ExpNet 子模型配置 ---
        self.expnet_config = {
            'n_terms': 4,  # 保持与之前一致
        }

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.001  # 端到端模型通常需要稍低的学习率
        self.patience = 20
        self.weight_decay = 0.0001

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")


# --- 2. 固定随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =================================================================================
# 3. 所有模型定义 (CycleNet, ExpNet, 和新的融合模型)
# =================================================================================

# --- CycleNet 子模型 ---
class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class CycleNetForSOH(nn.Module):
    def __init__(self, configs_dict):
        super(CycleNetForSOH, self).__init__()
        self.configs = configs_dict
        self.sequence_encoder = nn.Linear(configs_dict['sequence_length'] * configs_dict['sequence_feature_dim'],
                                          configs_dict['d_model'] // 2)
        self.scalar_encoder = nn.Linear(configs_dict['scalar_feature_dim'], configs_dict['d_model'] // 2)
        self.combined_feature_dim = configs_dict['d_model']
        self.cycle_queue = RecurrentCycle(
            cycle_len=configs_dict['meta_cycle_len'],
            channel_size=self.combined_feature_dim
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, configs_dict['d_ff']),
            nn.ReLU(),
            nn.Dropout(configs_dict['dropout']),
            nn.Linear(configs_dict['d_ff'], 1)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        x_seq_flat = x_seq.view(x_seq.size(0), -1)
        seq_embedding = self.sequence_encoder(x_seq_flat)
        scalar_embedding = self.scalar_encoder(x_scalar)
        combined_features = torch.cat((seq_embedding, scalar_embedding), dim=1)
        cycle_index = cycle_number % self.configs['meta_cycle_len']
        decycled_features = combined_features - self.cycle_queue(cycle_index, length=1).squeeze(1)
        prediction = self.prediction_head(decycled_features)
        return prediction


# --- ExpNet 子模型 (使用随机初始化) ---
class ExpNet(nn.Module):
    def __init__(self, n_terms=4):
        super(ExpNet, self).__init__()
        self.n_terms = n_terms

        # --- 使用随机初始化，使种子有意义 ---
        self.b = nn.Parameter(torch.rand(n_terms) * -0.01)  # 随机的小负数
        self.a = nn.Parameter(torch.rand(n_terms))  # 0~1的随机数
        self.d = nn.Parameter(torch.rand(n_terms))  # 0~1的随机数

    def forward(self, c):
        c = c.view(-1, 1)
        a = self.a.view(1, -1)
        b = self.b.view(1, -1)
        d = self.d.view(1, -1)
        out = a * torch.exp(b * c) + d
        out = out.sum(dim=1)
        return out


# --- ★★★ 全新的端到端融合模型 ★★★ ---
class EndToEndModel(nn.Module):
    def __init__(self, cyclenet_config, expnet_config, capacity_mean, capacity_scale):
        """
        初始化端到端模型。
        :param capacity_mean: 训练集中“累计放电容量”的均值 (来自StandardScaler)
        :param capacity_scale: 训练集中“累计放电容量”的标准差 (来自StandardScaler)
        """
        super(EndToEndModel, self).__init__()

        # 1. 实例化两个子模型
        self.cyclenet = CycleNetForSOH(cyclenet_config)
        self.expnet = ExpNet(expnet_config['n_terms'])

        # 2. 将均值和标准差注册为模型的 "buffer"
        # Buffer是模型的状态部分，但不是可训练的参数（即优化器不会更新它）
        # 这确保了它们与模型一起被保存，并移动到正确的设备(如GPU)
        self.register_buffer('capacity_mean', torch.tensor(capacity_mean, dtype=torch.float32))
        self.register_buffer('capacity_scale', torch.tensor(capacity_scale, dtype=torch.float32))

    def forward(self, x_seq, x_scalar, cycle_number):
        # 步骤 1: CycleNet 预测 *归一化的* 容量
        # 输出形状: [batch_size, 1]
        scaled_pred_cap = self.cyclenet(x_seq, x_scalar, cycle_number)

        # 步骤 2: ★反归一化层★
        # 将归一化的容量变回真实的物理容量 (y = (x - mean) / scale  =>  x = y * scale + mean)
        # 这个操作是可微分的，因此梯度可以流回 CycleNet
        real_pred_cap = (scaled_pred_cap * self.capacity_scale) + self.capacity_mean

        # 步骤 3: ExpNet 预测 SOH
        # ExpNet 需要 [batch_size] 的输入, 所以我们 squeeze 掉最后一个维度
        final_pred_soh = self.expnet(real_pred_cap.squeeze(-1))

        # 返回最终的SOH预测值，形状: [batch_size]
        return final_pred_soh


# =================================================================================
# 4. 数据集和数据加载 (修改为输出 SOH 作为目标)
# =================================================================================

class BatteryE2EDataset(Dataset):
    """端到端数据集：输入是CycleNet的输入, 目标是SOH"""

    def __init__(self, dataframe, sequence_col, scalar_cols, target_soh_col):
        self.df = dataframe.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.scalar_cols = scalar_cols
        self.target_soh_col = target_soh_col  # <--- 目标现在是 SOH

        # 转换输入数据
        self.sequences = np.array(self.df[self.sequence_col].tolist(), dtype=np.float32)
        self.scalars = self.df[self.scalar_cols].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

        # 转换目标数据 (SOH)
        self.targets = self.df[self.target_soh_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)

        y_soh = torch.tensor(self.targets[idx], dtype=torch.float32)  # <--- 返回SOH

        return x_seq, x_scalar, cycle_idx, y_soh


def load_and_preprocess_e2e_data(config):
    """
    加载端到端训练所需的数据。
    - 缩放CycleNet的输入特征。
    - 计算并返回CycleNet目标的Scaler（但不缩放目标本身）。
    - 创建SOH作为最终目标。
    """
    all_battery_data = []
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries

    # 从配置中获取信息
    seq_conf = config.cyclenet_config
    scalar_feature_cols = seq_conf['features_from_C']
    seq_feat_dim = seq_conf['sequence_feature_dim']
    seq_len = seq_conf['sequence_length']

    for battery_id in sorted(list(set(all_ids))):
        try:
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')
            df_a = pd.read_csv(path_a, sep=',')
            df_c = pd.read_csv(path_c, sep=',')
            df_c.rename(columns=lambda x: x.strip(), inplace=True)

            feature_cols = [f'弛豫段电压{i}' for i in range(1, seq_feat_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == seq_len]

            final_df = pd.merge(sequence_df, df_c, on='循环号')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)
        except Exception as e:
            print(f"警告: 电池 {battery_id} 文件加载失败: {e}")
            continue

    if not all_battery_data:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_battery_data, ignore_index=True)

    # --- 关键修改：创建新的目标列 ---
    full_df['soh'] = full_df['最大容量(Ah)'] / config.nominal_capacity
    target_soh_col = 'soh'  # <--- 这是我们的新Y
    sequence_col = 'voltage_sequence'

    # 更新配置中的标量维度
    config.cyclenet_config['scalar_feature_dim'] = len(scalar_feature_cols)

    # 划分数据集
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # --- 特征缩放 ---
    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()
    # ★★★ 我们仍然需要这个scaler，不是为了转换Y，而是为了获取它的mean和scale值 ★★★
    scaler_capacity_target = StandardScaler()

    # 在训练集上拟合缩放器
    scaler_seq.fit(np.vstack(train_df[sequence_col].values))
    scaler_scalar.fit(train_df[scalar_feature_cols])
    # ★ 拟合“累计放电容量” (CycleNet的原始目标)，以便模型内部可以反归一化 ★
    scaler_capacity_target.fit(train_df[['累计放电容量(Ah)']])

    # 应用缩放 (仅对输入特征)
    for df in [train_df, val_df, test_df]:
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])
        # !!! 注意：我们不再缩放任何目标列 !!!

    # 创建Dataset
    train_dataset = BatteryE2EDataset(train_df, sequence_col, scalar_feature_cols, target_soh_col)
    val_dataset = BatteryE2EDataset(val_df, sequence_col, scalar_feature_cols, target_soh_col)
    test_dataset = BatteryE2EDataset(test_df, sequence_col, scalar_feature_cols, target_soh_col)

    # 将输入缩放器保存到字典中 (我们单独返回capacity_scaler)
    input_scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar}

    return train_dataset, val_dataset, test_dataset, input_scalers, scaler_capacity_target


# =================================================================================
# 5. 训练和评估函数 (目标现在是 SOH)
# =================================================================================

# --- 训练函数 (目标是SOH) ---
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    # 数据解包 (y_target 现在是 y_soh)
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y_soh in dataloader:
        batch_seq = batch_seq.to(device)
        batch_scalar = batch_scalar.to(device)
        batch_cycle_idx = batch_cycle_idx.to(device)
        batch_y_soh = batch_y_soh.to(device).unsqueeze(-1)  # 形状变为 [batch, 1]

        optimizer.zero_grad()

        if grad_scaler:
            with autocast():
                # 模型直接输出SOH预测值
                outputs_soh = model(batch_seq, batch_scalar, batch_cycle_idx).unsqueeze(-1)
                loss = criterion(outputs_soh, batch_y_soh)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs_soh = model(batch_seq, batch_scalar, batch_cycle_idx).unsqueeze(-1)
            loss = criterion(outputs_soh, batch_y_soh)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# --- 评估函数 (目标是SOH) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds_soh = []
    all_labels_soh = []
    all_battery_ids = []  # 我们需要这个来进行分电池评估

    with torch.no_grad():
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y_soh in dataloader:
            batch_seq = batch_seq.to(device)
            batch_scalar = batch_scalar.to(device)
            batch_cycle_idx = batch_cycle_idx.to(device)
            batch_y_soh = batch_y_soh.to(device).unsqueeze(-1)

            outputs_soh = model(batch_seq, batch_scalar, batch_cycle_idx).unsqueeze(-1)
            loss = criterion(outputs_soh, batch_y_soh)

            total_loss += loss.item()
            all_preds_soh.append(outputs_soh.cpu().numpy())
            all_labels_soh.append(batch_y_soh.cpu().numpy())

            # 注意: BatteryE2EDataset 没有返回 battery_id, 我们需要在评估循环外部处理
            # 为了简单起见，我们假设dataloader的顺序与dataset.df一致 (因为shuffle=False)

    avg_loss = total_loss / len(dataloader)

    predictions_soh = np.concatenate(all_preds_soh).flatten()
    labels_soh = np.concatenate(all_labels_soh).flatten()

    # ★★★ 注意：这里的指标都是SOH的指标，不再需要反归一化 ★★★
    mae = mean_absolute_error(labels_soh, predictions_soh)
    mape = mean_absolute_percentage_error(labels_soh, predictions_soh)
    mse = mean_squared_error(labels_soh, predictions_soh)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels_soh, predictions_soh)

    metrics = {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

    # 返回真实的SOH预测值和标签
    return avg_loss, metrics, predictions_soh, labels_soh


# --- 绘图函数 (Y轴现在是SOH) ---
def plot_results(labels_soh, preds_soh, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels_soh, label='True SOH', marker='o', linestyle='-', markersize=4)
    plt.plot(preds_soh, label='Predicted SOH', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index (Cycle)', fontsize=12)
    plt.ylabel('SOH', fontsize=12)  # <--- Y轴是SOH
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


def plot_diagonal_results(labels_soh, preds_soh, title, save_path):
    plt.figure(figsize=(8, 8))
    min_val = min(np.min(labels_soh), np.min(preds_soh)) * 0.98
    max_val = max(np.max(labels_soh), np.max(preds_soh)) * 1.02
    plt.scatter(labels_soh, preds_soh, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')
    plt.xlabel('True SOH', fontsize=12)  # <--- X轴是True SOH
    plt.ylabel('Predicted SOH', fontsize=12)  # <--- Y轴是Predicted SOH
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# =================================================================================
# 6. 主执行函数 (采用5次随机运行架构)
# =================================================================================
def main():
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')

    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"所有实验的总保存路径: {config.save_path}")
    print(f"使用设备: {config.device}")

    # --- 1. ★★★ 加载一次数据, 并获取关键的Scaler参数 ★★★ ---
    try:
        train_dataset, val_dataset, test_dataset, input_scalers, scaler_capacity_target = load_and_preprocess_e2e_data(
            config)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载失败: {e}")
        return

    # 从Scaler中提取均值和标准差，用于注入模型
    # .mean_ 和 .scale_ 都是numpy数组，我们取第一个元素（也是唯一一个）
    capacity_mean = scaler_capacity_target.mean_[0]
    capacity_scale = scaler_capacity_target.scale_[0]
    print(f"\n端到端模型将使用以下参数进行内部反归一化：")
    print(f"  - 容量均值 (Capacity Mean): {capacity_mean:.4f}")
    print(f"  - 容量标准差 (Capacity Scale): {capacity_scale:.4f}")

    # 保存输入缩放器 (用于未来可能的推理)
    joblib.dump(input_scalers, os.path.join(config.save_path, 'input_scalers.pkl'))

    # 创建 DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"数据加载完成。训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    # --- 2. 设置5次随机运行的变量 ---
    num_runs = 5
    all_runs_metrics = []  # 存储每次run的【总体】指标
    all_runs_PER_BATTERY_metrics = []  # 存储【每个电池在每次run】的指标
    best_run_val_loss = float('inf')
    best_run_dir = None
    best_run_number = -1

    # --- 3. 开始多次实验循环 ---
    for run_number in range(1, num_runs + 1):
        current_seed = random.randint(0, 99999)
        set_seed(current_seed)

        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)

        print(f"\n{'=' * 30}")
        print(f" 开始第 {run_number}/{num_runs} 次实验 | 随机种子: {current_seed} ")
        print(f" 本次实验结果将保存到: {run_save_path}")
        print(f"{'=' * 30}")

        # --- 4. 初始化模型 (★ 注入 Mean 和 Scale ★) ---
        model = EndToEndModel(
            config.cyclenet_config,
            config.expnet_config,
            capacity_mean,
            capacity_scale
        ).to(config.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

        metrics_log = []
        best_val_loss_this_run = float('inf')
        epochs_no_improve = 0

        # --- 5. 训练循环 ---
        print("\n开始端到端训练...")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, config.device)  # 评估函数返回SOH指标

            print(
                f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失(SOH): {val_loss:.6f} | 验证R2(SOH): {val_metrics['R2']:.4f}")

            log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                         **{'val_' + k: v for k, v in val_metrics.items()}}
            metrics_log.append(log_entry)

            if val_loss < best_val_loss_this_run:
                best_val_loss_this_run = val_loss
                torch.save(model.state_dict(), os.path.join(run_save_path, 'best_e2e_model.pth'))
                print(f"  - 验证损失降低，保存模型到 {os.path.join(run_save_path, 'best_e2e_model.pth')}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break
        print("\n训练完成。")
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv(os.path.join(run_save_path, 'training_metrics_log.csv'), index=False)

        # --- 6. 评估最佳模型 (完全复用我们之前的最佳逻辑) ---
        print('\n加载本轮最佳模型进行最终评估...')
        model_path = os.path.join(run_save_path, 'best_e2e_model.pth')
        if not os.path.exists(model_path):
            print(f"错误: 找不到已训练的模型 '{model_path}'。")
            continue

        model.load_state_dict(torch.load(model_path, map_location=config.device))

        # 评估测试集。返回的已经是真实的SOH预测值和标签
        _, overall_test_metrics, test_preds_soh, test_labels_soh = evaluate(model, test_loader, criterion,
                                                                            config.device)

        # --- 7. 按电池分别评估和绘图 ---
        print("\n--- 本轮评估结果 (按单电池) ---")

        # 因为 test_loader 的 shuffle=False，预测顺序与 test_dataset.df 严格一致
        eval_df = test_dataset.df.copy()
        eval_df['pred_soh'] = test_preds_soh
        eval_df['true_soh'] = test_labels_soh  # 添加真实SOH以防万一

        per_battery_metrics_list = []
        for batt_id in config.test_batteries:
            batt_df = eval_df[eval_df['battery_id'] == batt_id]
            if batt_df.empty:
                print(f"  - 电池 {batt_id}: 未找到数据，跳过。")
                continue

            batt_true = batt_df['true_soh'].values
            batt_pred = batt_df['pred_soh'].values

            batt_metrics_dict = {
                'Battery_ID': batt_id,
                'MAE': mean_absolute_error(batt_true, batt_pred),
                'MAPE': mean_absolute_percentage_error(batt_true, batt_pred),
                'MSE': mean_squared_error(batt_true, batt_pred),
                'RMSE': np.sqrt(mean_squared_error(batt_true, batt_pred)),
                'R2': r2_score(batt_true, batt_pred)
            }
            per_battery_metrics_list.append(batt_metrics_dict)
            print(
                f"  - 电池 {batt_id}: MAE={batt_metrics_dict['MAE']:.6f}, RMSE={batt_metrics_dict['RMSE']:.6f}, R2={batt_metrics_dict['R2']:.4f}")

            # 添加跨实验总列表
            batt_metrics_with_run_info = batt_metrics_dict.copy()
            batt_metrics_with_run_info['run'] = run_number
            batt_metrics_with_run_info['seed'] = current_seed
            all_runs_PER_BATTERY_metrics.append(batt_metrics_with_run_info)

            # 绘制单独的曲线图 (SOH vs. Sample Index)
            plot_results(batt_true, batt_pred, f'Run {run_number} Battery {batt_id}: True vs Predicted SOH',
                         os.path.join(run_save_path, f'test_soh_plot_battery_{batt_id}.png'))

            # 绘制单独的对角图 (True SOH vs Pred SOH)
            plot_diagonal_results(batt_true, batt_pred, f'Run {run_number} Battery {batt_id}: SOH Diagonal Plot',
                                  os.path.join(run_save_path, f'test_diagonal_plot_battery_{batt_id}.png'))

        # 保存每个电池的指标汇总
        per_batt_df = pd.DataFrame(per_battery_metrics_list)
        per_batt_df.to_csv(os.path.join(run_save_path, 'test_per_battery_metrics.csv'), index=False)
        print(f"  -> 单独指标和图表已保存至: {run_save_path}")

        # --- 8. 汇总总体指标 ---
        print("\n--- 本轮评估结果 (所有测试电池汇总) ---")
        # overall_test_metrics 已经由 evaluate() 函数计算返回
        pd.DataFrame([overall_test_metrics]).to_csv(os.path.join(run_save_path, 'test_overall_metrics.csv'),
                                                    index=False)

        # 添加到跨run总列表
        current_run_summary = {'run': run_number, 'seed': current_seed, **overall_test_metrics}
        all_runs_metrics.append(current_run_summary)

        print(
            f"测试集(汇总): MAE={overall_test_metrics['MAE']:.6f}, RMSE={overall_test_metrics['RMSE']:.6f}, R2={overall_test_metrics['R2']:.4f}")

        # 保存总预测CSV
        eval_df[['battery_id', '循环号', 'true_soh', 'pred_soh']].to_csv(
            os.path.join(run_save_path, 'test_ALL_soh_predictions.csv'), index=False)

        # --- 9. 检查是否为最佳轮次 ---
        if best_val_loss_this_run < best_run_val_loss:
            best_run_val_loss = best_val_loss_this_run
            best_run_dir = run_save_path
            best_run_number = run_number
            print(f"*** 新的最佳表现！验证集损失: {best_val_loss_this_run:.6f} ***")

    # --- 10. 所有实验循环结束后，汇总报告 ---
    print(f"\n\n{'=' * 50}")
    print(" 所有端到端实验均已完成。")
    print(f"{'=' * 50}")

    # 保存“总体指标”的汇总 (每次run一行)
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 (总体指标) ---")
        print(summary_df.to_string())
        print(f"\n汇总指标已保存到: {summary_path}")

    # 保存“分电池指标”的详细汇总 (每个电池*每次run一行)
    print("\n正在生成所有实验的“分电池”详细汇总报告...")
    if all_runs_PER_BATTERY_metrics:
        per_batt_summary_df = pd.DataFrame(all_runs_PER_BATTERY_metrics)
        all_cols = list(per_batt_summary_df.columns)
        core_cols = ['Battery_ID', 'run', 'seed', 'MAE', 'R2', 'RMSE', 'MAPE', 'MSE']
        ordered_cols = [col for col in core_cols if col in all_cols] + [col for col in all_cols if col not in core_cols]
        per_batt_summary_df = per_batt_summary_df[ordered_cols]
        per_batt_summary_df = per_batt_summary_df.sort_values(by=['Battery_ID', 'run'])
        summary_path_per_batt = os.path.join(config.save_path, 'all_runs_per_battery_summary.csv')
        per_batt_summary_df.to_csv(summary_path_per_batt, index=False)
        print(f"“分电池”详细汇总报告已保存到: {summary_path_per_batt}")

    # 复制最佳模型结果
    if best_run_dir:
        print(
            f"\n表现最佳的实验是第 {best_run_number} 轮 (平均 MAE 最低: {best_run_val_loss:.6f})。")  # 注意：这里是按Val Loss最低选的
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