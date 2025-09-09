import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import warnings
import joblib
from torch.cuda.amp import autocast, GradScaler
from joblib import Parallel, delayed


# --- 1. 配置参数 (已修改) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (修改为三个输入路径) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'  # A文件: 弛豫段电压序列 (1200点/循环)
        # self.path_B_scalar = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/End'  # B文件: 弛豫末端电压 (1点/循环)
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'  # C文件: 其他特征和目标 (1行/循环)
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/cycle-LSTM/all'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        # 这里手动分配电池编号
        self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 11, 15, 17, 18, 19, 21, 22, 23, 24]
        self.val_batteries = [5, 10, 13, 19]
        self.test_batteries = [6, 12, 14, 20]

        # self.train_batteries = [1, 2, 3, 4]
        # self.val_batteries = [5]
        # self.test_batteries = [6]

        # self.train_batteries = [7, 8, 9, 11]
        # self.val_batteries = [10]
        # self.test_batteries = [12]

        # self.train_batteries = [15, 17, 18, 19]
        # self.val_batteries = [13]
        # self.test_batteries = [14]
        #
        # self.train_batteries = [21, 22, 23, 24]
        # self.val_batteries = [19]
        # self.test_batteries = [20]

        self.features_from_C = [
            # 'ICA峰值位置(V)',
            # '恒流充电时间(s)',
            '恒压充电时间(s)',
            # '恒流与恒压时间比值',
            # '2.8~3.4V放电时间(s)',
            '3.3~3.6V充电时间(s)',
            # '弛豫末端电压'
        ]

        # 2. 修改序列特征维度
        #    因为每个时间步只有一个特征（电压），所以设为 1
        self.sequence_feature_dim = 1

        # --- 模型超参数 ---
        self.meta_cycle_len = 7  # 定义一个元周期长度，比如假设电池每100个循环有一个宏观上的周期性变化
        self.sequence_length = 7 # A文件的序列长度
        self.scalar_feature_dim = len(self.features_from_C)  # B和C文件合并后的标量特征数量 (请根据实际情况调整)
        self.d_model = 256  # 隐藏层维度
        self.d_ff = 1024  # MLP编码器和预测头的中间层维度
        self.cycle_len = 2000  # 最大循环次数 (应大于任何电池的最大循环号)
        self.dropout = 0.2  # Dropout概率，0.1表示随机丢弃10%的神经元
        self.use_revin = False  # 是否使用可逆实例归一化
        self.weight_decay = 0.0001  # 增加权重衰减，1e-4或1e-5是常用的初始值

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.002
        self.patience = 15
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


# --- 3. 新的多模态模型定义 ---
class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        # 这里的 length 始终为1，因为我们是单点预测
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


#
class CycleNetForSOH(nn.Module):
    def __init__(self, configs):
        super(CycleNetForSOH, self).__init__()
        self.configs = configs

        # 1. 为两种不同的输入数据创建编码器
        self.sequence_encoder = nn.LSTM(
            input_size=configs.sequence_feature_dim,  # 输入特征维度
            hidden_size=configs.d_model // 2,  # 隐藏层大小
            num_layers=2,  # 增加网络深度
            batch_first=True,  # 输入数据格式为 (batch, seq, feature)
            dropout=configs.dropout  # 在LSTM层之间添加Dropout
        )

        self.scalar_encoder = nn.Linear(configs.scalar_feature_dim, configs.d_model // 2)

        # 合并后的特征维度
        self.combined_feature_dim = configs.d_model

        # 2. 引入 CycleNet 的核心：RecurrentCycle 模块
        # 注意 channel_size 是我们合并后的特征维度
        self.cycle_queue = RecurrentCycle(
            cycle_len=configs.meta_cycle_len,
            channel_size=self.combined_feature_dim
        )

        # 3. 创建预测头，用于处理去除了周期性后的特征
        # 输出维度为1，因为我们只预测一个值（最大容量）
        self.prediction_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

    # forward 方法 (正确版本)
    def forward(self, x_seq, x_scalar, cycle_number):
        # x_seq: (batch, seq_len, feature_dim) -> (batch, 7, 1)
        # x_scalar: (batch, scalar_dim)
        # cycle_number: (batch,)

        # --- 1. 特征编码与合并 ---

        # 1.1 序列编码
        # 直接将原始序列数据送入LSTM
        # LSTM的输出包括所有时间步的输出 `lstm_out` 和最后一个时间步的隐藏状态 `(h_n, c_n)`
        # 我们需要的是最后一个时间步的隐藏状态 h_n，它代表了整个序列的浓缩信息
        _, (h_n, _) = self.sequence_encoder(x_seq)

        # h_n 的形状是 (num_layers, batch, hidden_size)
        # 我们只需要最后一层的输出，所以取 h_n[-1]
        seq_embedding = h_n[-1]

        # 1.2 标量编码
        scalar_embedding = self.scalar_encoder(x_scalar)

        # 1.3 合并特征
        # 将序列特征的精炼表示和标量特征拼接起来
        combined_features = torch.cat((seq_embedding, scalar_embedding), dim=1)  # 形状: (batch, d_model)

        # --- 2. CycleNet核心：分解周期性 ---
        # 计算在元周期中的位置
        cycle_index = cycle_number % self.configs.meta_cycle_len

        # 从合并的特征中减去学习到的周期成分
        decycled_features = combined_features - self.cycle_queue(cycle_index, length=1).squeeze(1)

        # --- 3. 预测 ---
        # 将去除了周期性的特征送入预测头
        prediction = self.prediction_head(decycled_features)

        return prediction

# --- 4. 数据集定义 ---
class BatteryMultimodalDataset(Dataset):
    def __init__(self, dataframe, sequence_col, scalar_cols, target_col):
        """
        初始化多模态数据集 (单点模式)。
        """
        self.df = dataframe.reset_index(drop=True)
        self.sequence_col = sequence_col
        self.scalar_cols = scalar_cols
        self.target_col = target_col

        # 提前转换，避免在__getitem__中反复转换
        self.sequences = np.array(self.df[self.sequence_col].tolist(), dtype=np.float32)
        self.scalars = self.df[self.scalar_cols].values.astype(np.float32)
        self.targets = self.df[self.target_col].values.astype(np.float32)
        # 我们需要真实的循环号来计算周期索引
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # x_seq 的形状已经是 (sequence_length, 3)，可以直接转为Tensor
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        # 返回真实的循环号
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
            # path_b = os.path.join(config.path_B_scalar, f'EndVrlx_battery{battery_id}.csv')
            path_c = os.path.join(config.path_C_features, f'battery{battery_id}_SOH健康特征提取结果.csv')

            # 加载数据
            df_a = pd.read_csv(path_a, sep=',')  # 文件A：序列
            # df_b = pd.read_csv(path_b, sep=',')  # 文件B：标量1
            df_c = pd.read_csv(path_c, sep=',')  # 文件C：标量2 + 目标

            # 1. 合并标量特征
            # 确保 '循环号' 是正确的合并键
            # df_b.rename(columns=lambda x: x.strip(), inplace=True)
            df_c.rename(columns=lambda x: x.strip(), inplace=True)
            scalar_df = df_c  # 直接将df_c作为scalar_df

            # 2. 处理序列数据 (*** 这是需要修改的地方 ***)
            # 定义序列数据的列名
            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_length + 1)]

            # 检查列是否存在
            if not all(col in df_a.columns for col in feature_cols):
                print(f"警告: 电池 {battery_id} 的文件 {path_a} 中缺少所需的电压列，已跳过。")
                continue

            # --- 以下是核心修改 ---
            # 定义一个函数，将一行中的7个电压值转换为 (7, 1) 的形状
            def reshape_sequence(row):
                return row[feature_cols].values.reshape(config.sequence_length, config.sequence_feature_dim)

            # 为每一行应用这个转换函数
            df_a['voltage_sequence'] = df_a.apply(reshape_sequence, axis=1)

            # 现在 sequence_df 只需要'循环号'和'voltage_sequence'
            sequence_df = df_a[['循环号', 'voltage_sequence']]

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

    # #容量归一化
    # full_df['最大容量(Ah)'] = full_df['最大容量(Ah)'] / 3.5
    # print("已将'最大容量(Ah)'特征列的所有值除以3.5。")

    target_col = '累计放电容量(Ah)'
    sequence_col = 'voltage_sequence'

    # # 从文件B中获取所有列名（除了'循环号'）作为标量特征
    # # 我们需要先加载一个样本文件来获取列名
    # sample_b_path = os.path.join(config.path_B_scalar, f'EndVrlx_battery{config.train_batteries[0]}.csv')
    # sample_b_df = pd.read_csv(sample_b_path, sep=',', encoding='gbk')
    # features_from_B = [col.strip() for col in sample_b_df.columns if col.strip() != '循环号']
    #
    # # 从config中获取文件C的手动选择特征
    # features_from_C = config.features_from_C
    #
    # # 合并来自文件B和文件C的特征列表
    # scalar_feature_cols = features_from_B + features_from_C
    scalar_feature_cols = config.features_from_C
    # 检查所有选择的特征是否存在于DataFrame中
    for col in scalar_feature_cols:
        if col not in full_df.columns:
            raise ValueError(f"您手动选择的特征 '{col}' 不存在于加载的数据中。请检查 Config.features_from_C 中的拼写和列名。")

    # 更新config中的特征维度
    config.scalar_feature_dim = len(scalar_feature_cols)
    print(f"已手动选择 {config.scalar_feature_dim} 个标量特征: {scalar_feature_cols}")


    # 划分数据集
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)].copy()
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)].copy()

    # --- 特征缩放 ---
    # 为序列数据和标量数据创建不同的缩放器
    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()
    scaler_target = StandardScaler()

    # 在训练集上拟合缩放器
    # 序列数据需要先展平才能fit
    all_train_sequences = np.vstack(train_df[sequence_col].values)
    scaler_seq.fit(all_train_sequences)

    # 标量数据直接fit
    scaler_scalar.fit(train_df[scalar_feature_cols])
    scaler_target.fit(train_df[[target_col]])  # <--- 新增：拟合目标值缩放器
    # 应用缩放
    for df in [train_df, val_df, test_df]:
        # x已经是(seq_len, 3)的形状，可以直接transform
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
        # 缩放标量
        df.loc[:, scalar_feature_cols] = scaler_scalar.transform(df[scalar_feature_cols])
        # 对目标值也进行transform
        df.loc[:, [target_col]] = scaler_target.transform(df[[target_col]])

    # 创建Dataset
    train_dataset = BatteryMultimodalDataset(train_df, sequence_col, scalar_feature_cols, target_col)
    val_dataset = BatteryMultimodalDataset(val_df, sequence_col, scalar_feature_cols, target_col)
    test_dataset = BatteryMultimodalDataset(test_df, sequence_col, scalar_feature_cols, target_col)

    # 将所有缩放器保存在一个字典中
    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar, 'target': scaler_target}  # <--- 修改：将target_scaler也存起来

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
    all_cycle_indices = []  # <--- 新增：创建一个列表来存储循环号

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
            all_cycle_indices.append(batch_cycle_idx.cpu().numpy()) # <--- 新增：收集当前批次的循环号

    avg_loss = total_loss / len(dataloader)

    # 拼接所有批次的预测和标签
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    cycle_indices = np.concatenate(all_cycle_indices).flatten() # <--- 新增：将所有循环号拼接成一个数组

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)

    metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
    # <--- 修改：在返回值中增加循环号
    return avg_loss, metrics, predictions, labels, cycle_indices

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
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Cumulative Discharge Capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=1200)
    plt.close()


def plot_diagonal_results(labels, preds, title, save_path):
    """
    绘制真实值与预测值的对角散点图。
    """
    plt.figure(figsize=(8, 8))
    # 找到x和y轴的共同范围
    min_val = min(np.min(labels), np.min(preds)) * 0.98
    max_val = max(np.max(labels), np.max(preds)) * 1.02

    # 绘制散点图
    plt.scatter(labels, preds, alpha=0.6, label='Predicted vs. True')

    # 绘制y=x的完美预测线
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')

    plt.xlabel('True Cumulative Discharge Capacity (Ah)', fontsize=12)  # <--- 修改
    plt.ylabel('Predicted Cumulative Discharge Capacity (Ah)', fontsize=12)  # <--- 修改
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)

    # 设置坐标轴为相等比例，并限定范围
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.savefig(save_path, dpi=1200)
    plt.close()

# --- 9. 主执行函数 (已修改) ---
# 文件: cyclenet2.0-predict.py

def main():
    # 忽略一些不必要的警告
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    config = Config()
    set_seed(config.seed)

    config.save_path = os.path.join(config.save_path)
    os.makedirs(config.save_path, exist_ok=True)
    print(f"本次实验结果将保存到: {config.save_path}")
    print(f"使用设备: {config.device}")

    try:
        # load_and_preprocess_data 现在返回一个包含两个scaler的字典
        train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载失败: {e}")
        return
    # 保存scaler字典
    joblib.dump(scalers, os.path.join(config.save_path, 'scalers.pkl'))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    print(
        f"数据加载完成。训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

    model = CycleNetForSOH(config).to(config.device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)  # 增加权重衰减，1e-4或1e-5是常用的初始值
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6) #余弦退火
    grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

    metrics_log = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if config.mode in ['both', 'train']:
        print("\n开始训练模型...")
        for epoch in range(config.epochs):
            # train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, config.device, grad_scaler)
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)
            val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, config.device)
            print(
                f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")

            log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                         **{'val_' + k: v for k, v in val_metrics.items()}}
            metrics_log.append(log_entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.save_path, f'best_model.pth'))
                print(f"  - 验证损失降低，保存模型到 {os.path.join(config.save_path, f'best_model.pth')}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break
        print("\n训练完成。")
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv(os.path.join(config.save_path, f'training_metrics_log.csv'), index=False)
        if config.mode == 'train':
            return

    if config.mode in ['both', 'validate']:
        print('\n加载最佳模型进行最终评估...')
        model_path = os.path.join(config.save_path, f'best_model.pth')
        if not os.path.exists(model_path):
            print(f"错误: 找不到已训练的模型 '{model_path}'。请先在 'train' 或 'both' 模式下运行。")
            return

        model.load_state_dict(torch.load(model_path, map_location=config.device))

        # 从保存的文件中加载scalers字典
        scalers = joblib.load(os.path.join(config.save_path, 'scalers.pkl'))
        scaler_target = scalers['target']  # <--- 新增：获取目标值的缩放器

        # <--- 修改：接收新增的返回值 cycle_nums
        _, val_metrics, val_preds, val_labels, val_cycle_nums = evaluate(model, val_loader, criterion, config.device)
        _, test_metrics, test_preds, test_labels, test_cycle_nums = evaluate(model, test_loader, criterion,
                                                                             config.device)

        # --- 新增：反归一化以获得真实的容量值 ---
        # reshape(-1, 1) 是因为scaler需要二维输入
        val_preds_orig = scaler_target.inverse_transform(val_preds.reshape(-1, 1)).flatten()
        val_labels_orig = scaler_target.inverse_transform(val_labels.reshape(-1, 1)).flatten()
        test_preds_orig = scaler_target.inverse_transform(test_preds.reshape(-1, 1)).flatten()
        test_labels_orig = scaler_target.inverse_transform(test_labels.reshape(-1, 1)).flatten()

        # --- 使用反归一化后的值重新计算指标 ---
        val_metrics = {
            'MSE': mean_squared_error(val_labels_orig, val_preds_orig),
            'MAE': mean_absolute_error(val_labels_orig, val_preds_orig),
            'RMSE': np.sqrt(mean_squared_error(val_labels_orig, val_preds_orig)),
            'R2': r2_score(val_labels_orig, val_preds_orig)
        }
        test_metrics = {
            'MSE': mean_squared_error(test_labels_orig, test_preds_orig),
            'MAE': mean_absolute_error(test_labels_orig, test_preds_orig),
            'RMSE': np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig)),
            'R2': r2_score(test_labels_orig, test_preds_orig)
        }

        # 由于目标值'最大容量(Ah)'没有被归一化，所以不需要反归一化，直接使用即可
        print("\n--- 评估结果 ---")
        final_metrics = pd.DataFrame([
            {'set': 'validation', **val_metrics},
            {'set': 'test', **test_metrics}
        ])
        final_metrics.to_csv(os.path.join(config.save_path, 'final_evaluation_metrics.csv'), index=False)
        print(
            f"最终验证集指标: MSE={val_metrics['MSE']:.6f}, MAE={val_metrics['MAE']:.6f}, RMSE={val_metrics['RMSE']:.6f}, R2={val_metrics['R2']:.4f}")
        print(
            f"最终测试集指标: MSE={test_metrics['MSE']:.6f}, MAE={test_metrics['MAE']:.6f}, RMSE={test_metrics['RMSE']:.6f}, R2={test_metrics['R2']:.4f}")

        # <--- 修改：在创建DataFrame时加入'循环号'列
        val_results_df = pd.DataFrame(
            {'Original_Cycle_Index': val_cycle_nums, 'True_Capacity_Ah': val_labels_orig, 'Predicted_Capacity_Ah': val_preds_orig})
        test_results_df = pd.DataFrame(
            {'Original_Cycle_Index': test_cycle_nums, 'True_Capacity_Ah': test_labels_orig, 'Predicted_Capacity_Ah': test_preds_orig})

        val_results_df.to_csv(os.path.join(config.save_path, f'validation_predictions.csv'), index=False)
        test_results_df.to_csv(os.path.join(config.save_path, f'test_predictions.csv'), index=False)
        print(f"\n验证集和测试集的预测值已保存。")

        plot_results(val_labels_orig, val_preds_orig, 'Validation Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, f'validation_plot.png'))
        plot_results(test_labels_orig, test_preds_orig, 'Test Set: True vs. Predicted Capacity',
                     os.path.join(config.save_path, f'test_plot.png'))
        print(f"结果对比图已保存。")

        # --- 新增：绘制对角图 ---
        plot_diagonal_results(val_labels_orig, val_preds_orig, 'Validation Set: Diagonal Plot',
                              os.path.join(config.save_path, f'validation_diagonal_plot.png'))
        plot_diagonal_results(test_labels_orig, test_preds_orig, 'Test Set: Diagonal Plot',
                              os.path.join(config.save_path, f'test_diagonal_plot.png'))
        print(f"对角图已保存。")

if __name__ == '__main__':
    main()