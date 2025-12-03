import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
import joblib
from torch.cuda.amp import autocast, GradScaler
import itertools

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')


# --- 1. 配置参数 (已修改) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (修改为三个输入路径) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-Downsampling_600x'  # A文件: 弛豫段电压序列 (1200点/循环)
        self.path_B_scalar = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/End'  # B文件: 弛豫末端电压 (1点/循环)
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'  # C文件: 其他特征和目标 (1行/循环)
        self.save_path = '/home/scuee_user06/myh/电池/data/cyclenet_result'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        # 这里手动分配电池编号
        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22]
        # self.val_batteries = [5, 11, 17, 23]
        # self.test_batteries = [6, 12, 18, 24]
        self.train_batteries = [1, 2, 3, 4]
        self.val_batteries = [5]
        self.test_batteries = [6]

        # --- 模型超参数 (已修改) ---
        self.sequence_length = 2  # A文件的序列长度 (弛豫段电压的点数)
        self.scalar_feature_dim = 10  # B和C文件合并后的标量特征数量 (请根据实际情况调整)
        self.d_model = 64  # 隐藏层维度
        self.d_ff = 1024  # MLP编码器和预测头的中间层维度
        self.cycle_len = 2000  # 最大循环次数 (应大于任何电池的最大循环号)
        self.dropout = 0.2  # Dropout概率，0.1表示随机丢弃10%的神经元
        self.use_revin = False  # 是否使用可逆实例归一化
        self.weight_decay = 1e-5  # 增加权重衰减，1e-4或1e-5是常用的初始值

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 512
        self.learning_rate = 0.005
        self.patience = 40
        self.seed = 2025
        self.mode = 'both'  # 可选 'train', 'validate', 'both'

        # --- 设备设置 ---
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")


# --- 2. 固定随机种子 ---
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


# --- 3. 新的多模态模型定义 (完全重写) ---
class RecurrentCycle(torch.nn.Module):
    # 这个辅助模块保持不变
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    """
    新的多模态模型架构:
    1. CycleNet Backbone: 处理电压序列数据 (文件A)
    2. MLP Encoder: 处理所有标量特征数据 (文件B, C)
    3. Prediction Head: 拼接两部分输出并预测最终SOH
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_revin = configs.use_revin
        self.sequence_length = configs.sequence_length

        # Part 1: CycleNet 主干 (用于序列数据)
        # RecurrentCycle 的 channel_size 为 1，因为输入只有"弛豫段电压"一个维度
        self.cycleQueue = RecurrentCycle(cycle_len=configs.cycle_len, channel_size=1)
        # 一个线性层将1200点的序列压缩成一个 d_model 维的向量
        self.sequence_processor = nn.Linear(configs.sequence_length, configs.d_model)

        # Part 2: MLP 编码器 (用于标量特征)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(configs.scalar_feature_dim, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, configs.d_model)
        )

        # Part 3: 最终预测头
        # 输入维度是 CycleNet输出(d_model) + MLP输出(d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)  # 输出一个值: 最大容量
        )

    def forward(self, x_seq, x_scalar, cycle_index):
        # x_seq: (batch_size, sequence_length, 1) - 序列数据
        # x_scalar: (batch_size, scalar_feature_dim) - 标量数据
        # cycle_index: (batch_size,)

        # --- 处理序列数据 ---
        # RevIN (可选)
        if self.use_revin:
            seq_mean = torch.mean(x_seq, dim=1, keepdim=True)
            seq_var = torch.var(x_seq, dim=1, keepdim=True) + 1e-5
            x_seq = (x_seq - seq_mean) / torch.sqrt(seq_var)

        # CycleNet核心操作
        # 注意：这里的 cycleQueue 需要的 length 是 sequence_length
        processed_seq = x_seq - self.cycleQueue(cycle_index, self.sequence_length)

        # 将序列展平以输入线性层
        processed_seq = processed_seq.view(processed_seq.size(0), -1)
        seq_embedding = self.sequence_processor(processed_seq)  # -> (batch_size, d_model)

        # --- 处理标量数据 ---
        scalar_embedding = self.scalar_encoder(x_scalar)  # -> (batch_size, d_model)

        # --- 拼接与预测 ---
        combined_embedding = torch.cat((seq_embedding, scalar_embedding), dim=1)
        prediction = self.prediction_head(combined_embedding)

        return prediction


# --- 4. 数据集定义 (完全重写) ---
class BatteryMultimodalDataset(Dataset):
    def __init__(self, dataframe, sequence_col, scalar_cols, target_col):
        """
        初始化多模态数据集。
        Args:
            dataframe (pd.DataFrame): 包含所有预处理好数据的DataFrame。
            sequence_col (str): 包含Numpy序列数组的列名。
            scalar_cols (list): 包含所有标量特征的列名列表。
            target_col (str): 目标列名。
        """
        self.df = dataframe.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.scalar_cols = scalar_cols
        self.target_col = target_col

        # 提前转换，避免在__getitem__中反复转换
        self.sequences = np.array(self.df[self.sequence_col].tolist(), dtype=np.float32)
        self.scalars = self.df[self.scalar_cols].values.astype(np.float32)
        self.targets = self.df[self.target_col].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # x_seq需要增加一个通道维度
        x_seq = torch.from_numpy(self.sequences[idx]).unsqueeze(-1)  # -> (sequence_length, 1)
        x_scalar = torch.from_numpy(self.scalars[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)

        return x_seq, x_scalar, cycle_idx, y


# --- 5. 数据加载和预处理 (完全重写) ---
def load_and_preprocess_data(config):
    """加载来自三个文件夹的数据，进行合并、预处理和划分"""
    all_battery_data = []

    # 假设电池文件名格式为 'battery_X.csv'
    all_ids = config.train_batteries + config.val_batteries + config.test_batteries

    for battery_id in sorted(list(set(all_ids))):
        try:
            print(f"正在处理电池 {battery_id}...")
            # 路径拼接
            path_a = os.path.join(config.path_A_sequence, f'relaxation_battery{battery_id}.csv')
            path_b = os.path.join(config.path_B_scalar, f'EndVrlx_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')

            # 加载数据
            df_a = pd.read_csv(path_a, sep=',', encoding='gbk')  # 文件A：序列
            df_b = pd.read_csv(path_b, sep=',', encoding='gbk')  # 文件B：标量1
            df_c = pd.read_csv(path_c, sep=',', encoding='gbk')  # 文件C：标量2 + 目标

            # 1. 合并标量特征
            # 确保 '循环号' 是正确的合并键
            df_b.rename(columns=lambda x: x.strip(), inplace=True)
            df_c.rename(columns=lambda x: x.strip(), inplace=True)
            scalar_df = pd.merge(df_b, df_c, on='循环号')

            # 2. 处理序列数据
            # 按'循环号'分组，并将1200个'弛豫段电压'点聚合为一个Numpy数组
            sequence_df = df_a.groupby('循环号')['弛豫段电压'].apply(lambda x: x.values).reset_index()
            sequence_df.rename(columns={'弛豫段电压': 'voltage_sequence'}, inplace=True)

            # 过滤掉长度不符合要求的序列
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == config.sequence_length]

            # 3. 合并序列和标量数据
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

    # 定义特征和目标
    target_col = '最大容量(Ah)'
    # 所有非目标、非ID、非序列的列都是标量特征
    scalar_feature_cols = [col for col in full_df.columns if
                           col not in [target_col, 'battery_id', '循环号', 'voltage_sequence']]
    sequence_col = 'voltage_sequence'

    # 更新config中的特征维度
    config.scalar_feature_dim = len(scalar_feature_cols)
    print(f"检测到 {config.scalar_feature_dim} 个标量特征: {scalar_feature_cols}")

    # 划分数据集
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # --- 特征缩放 ---
    # 为序列数据和标量数据创建不同的缩放器
    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()

    # 在训练集上拟合缩放器
    # 序列数据需要先展平才能fit
    all_train_sequences = np.vstack(train_df[sequence_col].values).reshape(-1, 1)
    scaler_seq.fit(all_train_sequences)

    # 标量数据直接fit
    scaler_scalar.fit(train_df[scalar_feature_cols])

    # 应用缩放
    for df in [train_df, val_df, test_df]:
        # 缩放序列
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x.reshape(-1, 1)).flatten())
        # 缩放标量
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])

    # 创建Dataset
    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col)

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar}

    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练函数 (已修改) ---
# def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, grad_scaler):
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    # 修改了数据解包
    for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        batch_seq = batch_seq.to(device)
        batch_scalar = batch_scalar.to(device)
        batch_cycle_idx = batch_cycle_idx.to(device)
        batch_y = batch_y.to(device).unsqueeze(-1)  # 确保y的形状为(batch, 1)

        optimizer.zero_grad()

        # 使用混合精度训练
        if grad_scaler:
            with autocast():
                outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
                loss = criterion(outputs, batch_y)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    # scheduler.step()
    return total_loss / len(dataloader)


# --- 7. 验证/测试函数 (已修改) ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 修改了数据解包
        for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
            batch_seq = batch_seq.to(device)
            batch_scalar = batch_scalar.to(device)
            batch_cycle_idx = batch_cycle_idx.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)  # 确保y的形状为(batch, 1)

            outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # 拼接所有批次的预测和标签
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)

    metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    return avg_loss, metrics, predictions, labels


# --- 8. 可视化和工具函数 (基本不变) ---
def get_exp_tag(config):
    train_ids = '-'.join([str(i) for i in config.train_batteries])
    val_ids = '-'.join([str(i) for i in config.val_batteries])
    test_ids = '-'.join([str(i) for i in config.test_batteries])
    tag = (
        f"train{train_ids}_val{val_ids}_test{test_ids}_"
        f"ep{config.epochs}_bs{config.batch_size}_lr{config.learning_rate}_dp{config.dropout}"
        f"dm{config.d_model}"
    )
    return tag


def plot_results(labels, preds, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels', marker='o', linestyle='-', markersize=4)
    plt.plot(preds, label='Predictions', marker='x', linestyle='--', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index (Cycle)', fontsize=12)
    plt.ylabel('Max Capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


# --- 9. 主执行函数 (已修改) ---
def run_experiment(config):
    """
      运行一次完整的训练、验证和评估流程。
      Args:
          config (Config): 包含所有参数的配置对象。
      Returns:
          float: 本次实验在验证集上的最佳R2分数，用于评估超参数好坏。
      """
    set_seed(config.seed)

    # 每次实验都应该有自己独立的文件夹，避免结果覆盖
    # 为此，我们根据核心超参数生成一个唯一的标签
    exp_tag = (
        f"lr{config.learning_rate}_dp{config.dropout}_bs{config.batch_size}_"
        f"dm{config.d_model}_wd{config.weight_decay}"
    )
    # 注意：这里的 save_path 是基础路径，我们会在其下创建子文件夹
    # config.base_save_path = config.save_path
    config.save_path = os.path.join(config.save_path, exp_tag)
    os.makedirs(config.save_path, exist_ok=True)

    print("-" * 80)
    print(f"开始实验: {exp_tag}")
    print(f"结果将保存到: {config.save_path}")
    print(f"使用设备: {config.device}")

    try:
        train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载失败: {e}")
        return -float('inf')  # 返回一个极差的分数

    joblib.dump(scalers, os.path.join(config.save_path, 'scalers.pkl'))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Model(config).to(config.device)
    criterion = nn.MSELoss()

    # 在优化器中使用 config.weight_decay
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=15, verbose=False)
    grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_val_r2 = -float('inf')  # 我们要最大化 R2 分数
    epochs_no_improve = 0

    if config.mode in ['both', 'train']:
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, config.device)

            # 使用 val_loss 来更新学习率
            scheduler.step(val_loss)

            # 更新最佳R2分数
            if val_metrics['R2'] > best_val_r2:
                best_val_r2 = val_metrics['R2']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.save_path, 'best_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break

    print(f"实验 {exp_tag} 完成。最佳验证集 R2: {best_val_r2:.4f}")
    print("-" * 80)

    # 返回我们最关心的指标，用于比较不同超参数组合的优劣
    return best_val_r2


if __name__ == '__main__':
    # --- 1. 定义超参数网格 ---
    # 定义你想要尝试的超参数和对应的值的列表
    param_grid = {
        'learning_rate': [0.002, 0.005, 0.01],
        'dropout': [0.1, 0.2, 0.3],
        'weight_decay': [1e-4, 1e-5, 0],
        'd_model': [64, 128, 256, 512],
        'batch_size': [256, 512]
    }

    # --- 2. 创建所有超参数组合 ---
    # 获取参数名和参数值列表
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # 使用itertools.product生成所有组合
    param_combinations = list(itertools.product(*param_values))
    total_combos = len(param_combinations)

    print(f"网格搜索开始，总共需要测试 {total_combos} 种超参数组合。")

    # --- 3. 循环执行实验并记录结果 ---
    results = []
    best_score = -float('inf')
    best_params = {}

    base_save_path = '/home/scuee_user06/myh/电池/data/cyclenet_gridsearch-600x_results'  # 设置一个基础保存路径

    for i, params_tuple in enumerate(param_combinations):
        # 创建一个当前的参数字典
        current_params = dict(zip(param_names, params_tuple))

        print(f"\n--- 开始测试第 {i + 1}/{total_combos} 种组合 ---")
        print(current_params)

        # 创建一个新的Config实例
        config = Config()

        # 用当前组合的超参数更新Config对象
        # setattr 是一个很方便的函数，等价于 config.learning_rate = 0.005
        for key, value in current_params.items():
            setattr(config, key, value)

        # 设置基础保存路径，这样每次实验的文件夹都会在同一个地方创建
        config.save_path = base_save_path

        # 运行实验并获取分数
        score = run_experiment(config)

        # 记录结果
        results.append({
            'score': score,
            'params': current_params
        })

        # 更新最佳结果
        if score > best_score:
            best_score = score
            best_params = current_params
            print(f"*** 发现新的最佳组合！分数: {best_score:.4f} ***")

    # --- 4. 打印最终总结 ---
    print("\n\n" + "=" * 80)
    print("网格搜索完成！")
    print(f"最佳验证集 R2 分数: {best_score:.4f}")
    print("对应的最佳超参数组合:")
    print(best_params)

    # 您也可以将 results 列表保存为json或csv文件，以便后续详细分析
    import json

    with open(os.path.join(base_save_path, 'grid_search_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)