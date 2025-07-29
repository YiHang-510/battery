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



# --- 1. 配置参数 (已修改) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (修改为三个输入路径) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw'  # A文件: 弛豫段电压序列 (1200点/循环)
        self.path_B_scalar = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/End'  # B文件: 弛豫末端电压 (1点/循环)
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'  # C文件: 其他特征和目标 (1行/循环)
        self.save_path = '/home/scuee_user06/myh/电池/data/cyclenet_result-3demision'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        # 1. 定义一个包含所有可选电池的“大池子”
        # self.train_val_pool = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23] # 示例：包含20个电池
        self.train_val_pool = [1, 2, 3, 4, 5] # 示例：包含20个电池
        # self.train_val_pool = [7, 8, 9, 10, 11] # 示例：包含20个电池
        # self.train_val_pool = [13, 14, 15, 16, 17] # 示例：包含20个电池
        # self.train_val_pool = [19, 20, 21, 22, 23] # 示例：包含20个电池
        # 2. 定义您想从池子中选出多少个电池进行本次实验
        self.num_batteries_to_select = 5  # <--- 您指定的“选择5个电池”
        # 3. 定义这部分选出的电池中，有几个作为验证集
        self.num_validation_batteries = 1 # <--- 您指定的“1个为验证集”
        # 4. 固定测试集 (保持不变)
        self.test_batteries = [6] # 示例：测试集保持固定
        # self.test_batteries = [12] # 示例：测试集保持固定
        # self.test_batteries = [18] # 示例：测试集保持固定
        # self.test_batteries = [24] # 示例：测试集保持固定

        # 5. 将原来的 train_batteries 和 val_batteries 留空，它们将在 main 函数中被动态赋值
        self.train_batteries = []
        self.val_batteries = []

        self.features_from_C = [
            # 'ICA峰值位置(V)',
            '恒流充电时间(s)',
            '恒压充电时间(s)',
            # '恒流与恒压时间比值',
            # '2.8~3.4V放电时间(s)',
            # '3.3~3.6V充电时间(s)'
        ]

        # --- 模型超参数 ---
        self.cycle_window_size = 15  # 定义窗口大小
        self.sequence_length = 1 # A文件的序列长度 (弛豫段电压的点数)
        self.scalar_feature_dim = 10  # B和C文件合并后的标量特征数量 (请根据实际情况调整)
        self.d_model = 256  # 隐藏层维度
        self.d_ff = 1024  # MLP编码器和预测头的中间层维度
        self.cycle_len = 2000  # 最大循环次数 (应大于任何电池的最大循环号)
        self.dropout = 0.2  # Dropout概率，0.1表示随机丢弃10%的神经元
        self.use_revin = False  # 是否使用可逆实例归一化
        self.weight_decay = 0.0001  # 增加权重衰减，1e-4或1e-5是常用的初始值

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 256
        self.learning_rate = 0.005
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
class SequentialModel(nn.Module):
    """
    能够处理循环序列的LSTM模型。
    """

    def __init__(self, configs):
        super(SequentialModel, self).__init__()
        self.configs = configs

        # 特征提取层，用于处理每个时间点的数据
        # 将原始的3维电压序列和10维标量特征，映射到一个统一的d_model维空间
        self.sequence_processor = nn.Linear(configs.sequence_length * 3, configs.d_model // 2)
        self.scalar_encoder = nn.Linear(configs.scalar_feature_dim, configs.d_model // 2)

        self.combined_feature_dim = configs.d_model  # 合并后单个时间点的特征维度

        # 时序融合层，使用LSTM捕捉30个循环间的时序关系
        self.lstm = nn.LSTM(
            input_size=self.combined_feature_dim,
            hidden_size=configs.d_model,  # LSTM的隐藏层大小
            num_layers=2,  # LSTM的层数，可以设为1或2
            batch_first=True,  # 输入数据格式为 (batch, seq, feature)
            dropout=configs.dropout if 2 > 1 else 0  # 只有多层LSTM才能设置dropout
        )

        # 最终预测头，根据LSTM的输出进行预测
        self.prediction_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

    def forward(self, x_seq_window, x_scalar_window):
        # 输入形状:
        # x_seq_window: (batch, window_size, seq_len, 3) -> (batch, 30, 1, 3)
        # x_scalar_window: (batch, window_size, scalar_dim) -> (batch, 30, 10)

        batch_size = x_seq_window.size(0)
        window_size = self.configs.cycle_window_size

        # --- 1. 特征提取 ---
        # 将batch和window维度合并，以便一次性通过线性层进行特征提取
        x_seq_flat = x_seq_window.view(batch_size * window_size, -1)
        x_scalar_flat = x_scalar_window.view(batch_size * window_size, -1)

        seq_embedding = self.sequence_processor(x_seq_flat)
        scalar_embedding = self.scalar_encoder(x_scalar_flat)

        # 合并特征，并恢复成 (batch, window_size, feature_dim) 的序列形状
        combined_features = torch.cat((seq_embedding, scalar_embedding), dim=1)
        lstm_input = combined_features.view(batch_size, window_size, self.combined_feature_dim)

        # --- 2. 时序融合 ---
        # lstm_out 形状: (batch, window_size, d_model)
        # _ (h_n, c_n) 是最后一个时间步的隐藏状态和细胞状态
        lstm_out, _ = self.lstm(lstm_input)

        # 取序列最后一个时间点的输出，它融合了前面所有时间点的信息
        last_time_step_out = lstm_out[:, -1, :]  # 形状: (batch, d_model)

        # --- 3. 预测 ---
        prediction = self.prediction_head(last_time_step_out)

        return prediction


# --- 4. 数据集定义 ---
class BatterySlidingWindowDataset(Dataset):
    """
    为单块电池生成滑动窗口数据。
    这个Dataset的实例对应一块电池。
    """

    def __init__(self, dataframe, sequence_col, scalar_cols, target_col, window_size):
        self.window_size = window_size

        # 提取所需数据列，并转换为Numpy数组
        self.sequence_data = np.array(dataframe[sequence_col].tolist(), dtype=np.float32)
        self.scalar_data = dataframe[scalar_cols].values.astype(np.float32)
        self.targets = dataframe[target_col].values.astype(np.float32)

        # 因为前 (window_size - 1) 个点无法构成完整窗口，所以有效样本数减少
        self.num_samples = len(dataframe) - self.window_size + 1
        if self.num_samples < 0:
            self.num_samples = 0  # 如果电池循环总数小于窗口大小，则样本数为0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 根据索引，切分出一个窗口的数据
        start_idx = idx
        end_idx = idx + self.window_size

        x_seq_window = self.sequence_data[start_idx:end_idx]
        x_scalar_window = self.scalar_data[start_idx:end_idx]

        # 目标y是窗口中最后一个时间点(即当前时间点)的容量
        y = self.targets[end_idx - 1]

        # 返回PyTorch张量
        return (torch.from_numpy(x_seq_window),
                torch.from_numpy(x_scalar_window),
                torch.tensor(y, dtype=torch.float32))

# --- 5. 数据加载和预处理 (完全重写) ---
def load_and_preprocess_data_for_sliding_window(config):
    """
    加载、预处理并为数据创建滑动窗口数据集。
    """
    # ------------------- 第1步：加载所有数据到一个DataFrame (与旧代码类似) -------------------
    all_battery_data = []
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
            # 按'循环号'分组，并将'弛豫段电压'点聚合为一个Numpy数组
            # 三个特征列名
            feature_cols = ['弛豫段电压1', '弛豫段电压2', '弛豫段电压3']
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            print("重命名之后的列名:", sequence_df.columns)  # 加上这行来查看
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

    target_col = '最大容量(Ah)'
    sequence_col = 'voltage_sequence'

    # 从文件B中获取所有列名（除了'循环号'）作为标量特征
    # 我们需要先加载一个样本文件来获取列名
    sample_b_path = os.path.join(config.path_B_scalar, f'EndVrlx_battery{config.train_batteries[0]}.csv')
    sample_b_df = pd.read_csv(sample_b_path, sep=',', encoding='gbk')
    features_from_B = [col.strip() for col in sample_b_df.columns if col.strip() != '循环号']

    # 从config中获取文件C的手动选择特征
    features_from_C = config.features_from_C

    # 合并来自文件B和文件C的特征列表
    scalar_feature_cols = features_from_B + features_from_C

    config.scalar_feature_dim = len(scalar_feature_cols)
    print(f"已选择 {config.scalar_feature_dim} 个标量特征: {scalar_feature_cols}")

    # 数据划分和缩放
    train_df_for_scaler = full_df[full_df['battery_id'].isin(config.train_batteries)].copy()

    scaler_seq = StandardScaler()
    scaler_scalar = StandardScaler()

    # 在训练集上拟合缩放器
    all_train_sequences = np.vstack(train_df_for_scaler[sequence_col].values)
    scaler_seq.fit(all_train_sequences)
    scaler_scalar.fit(train_df_for_scaler[scalar_feature_cols])

    # 将缩放应用到整个数据集
    full_df[sequence_col] = full_df[sequence_col].apply(lambda x: scaler_seq.transform(x))
    full_df.loc[:, scalar_feature_cols] = scaler_scalar.transform(full_df[scalar_feature_cols])

    def create_concat_dataset(battery_ids, source_df, config):
        """一个辅助函数，用于为一组电池ID创建合并的数据集"""
        list_of_datasets = []
        print(f"\n为电池组 {battery_ids} 创建数据集...")
        for bat_id in battery_ids:
            # 从已经缩放过的完整数据中，筛选出当前电池的数据
            battery_df = source_df[source_df['battery_id'] == bat_id].copy()

            # 只有当电池的循环次数足够长，能够至少形成一个窗口时，才创建Dataset
            if len(battery_df) >= config.cycle_window_size:
                print(
                    f"  - 电池 {bat_id}: 循环数 {len(battery_df)}, 可生成 {len(battery_df) - config.cycle_window_size + 1} 个样本。")
                dataset = BatterySlidingWindowDataset(
                    dataframe=battery_df,
                    sequence_col=sequence_col,
                    scalar_cols=scalar_feature_cols,
                    target_col=target_col,
                    window_size=config.cycle_window_size
                )
                list_of_datasets.append(dataset)
            else:
                print(f"  - 电池 {bat_id}: 循环数 {len(battery_df)}, 不足以形成窗口，已跳过。")

        if not list_of_datasets:
            # 如果这个电池组（例如验证集）没有任何一个电池满足长度要求，则返回None
            return None
            # ConcatDataset可以将多个dataset对象逻辑上拼接成一个大的dataset
        return ConcatDataset(list_of_datasets)

    # 使用辅助函数创建最终的数据集
    train_dataset = create_concat_dataset(config.train_batteries, full_df, config)
    val_dataset = create_concat_dataset(config.val_batteries, full_df, config)
    test_dataset = create_concat_dataset(config.test_batteries, full_df, config)

    scalers = {'sequence': scaler_seq, 'scalar': scaler_scalar}

    return train_dataset, val_dataset, test_dataset, scalers


# --- 6. 训练函数 (已修改) ---
# def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, grad_scaler):
def train_epoch(model, dataloader, optimizer, criterion, device, grad_scaler):
    model.train()
    total_loss = 0
    # 修改了数据解包
    for batch_seq_window, batch_scalar_window, batch_y in dataloader:

        # 将数据移动到指定设备
        batch_seq_window = batch_seq_window.to(device)
        batch_scalar_window = batch_scalar_window.to(device)
        batch_y = batch_y.to(device).unsqueeze(-1)  # 确保y的形状为(batch, 1)

        optimizer.zero_grad()

        if grad_scaler:
            with autocast():
                # 【修改2】: 修改模型调用方式，不再传入cycle_idx
                # 旧: outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
                outputs = model(batch_seq_window, batch_scalar_window)
                loss = criterion(outputs, batch_y)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            # 【修改3】: 同上，修改模型调用方式
            outputs = model(batch_seq_window, batch_scalar_window)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# --- 7. 验证/测试函数 ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # 【修改2】: 修改for循环的数据解包方式
        # 旧: for batch_seq, batch_scalar, batch_cycle_idx, batch_y in dataloader:
        for batch_seq_window, batch_scalar_window, batch_y in dataloader:
            # 将数据移动到指定设备
            batch_seq_window = batch_seq_window.to(device)
            batch_scalar_window = batch_scalar_window.to(device)
            batch_y = batch_y.to(device).unsqueeze(-1)

            # 【修改3】: 修改模型调用方式
            # 旧: outputs = model(batch_seq, batch_scalar, batch_cycle_idx)
            outputs = model(batch_seq_window, batch_scalar_window)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            # 【修改4】: 删除对 cycle_indices 的收集
            # all_cycle_indices.append(batch_cycle_idx.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    # 【修改5】: 删除对 cycle_indices 的拼接
    # cycle_indices = np.concatenate(all_cycle_indices).flatten()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)

    metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

    # 【修改6】: 修改返回值，不再返回 cycle_indices
    # 旧: return avg_loss, metrics, predictions, labels, cycle_indices
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

    plt.xlabel('True Capacity (Ah)', fontsize=12)
    plt.ylabel('Predicted Capacity (Ah)', fontsize=12)
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
    # --- 初始化部分 (保持不变) ---
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    config = Config()
    set_seed(config.seed)
    exp_tag = get_exp_tag(config)
    config.save_path = os.path.join(config.save_path, exp_tag)
    os.makedirs(config.save_path, exist_ok=True)
    print(f"本次实验结果将保存到: {config.save_path}")
    print(f"使用设备: {config.device}")

    print("--- 开始执行随机抽样来划分训练集和验证集 ---")

    # 1. 从大池子中随机选择指定数量的电池
    if len(config.train_val_pool) < config.num_batteries_to_select:
        raise ValueError("电池池子的大小小于需要选择的电池数量！")

    selected_batteries = random.sample(config.train_val_pool, config.num_batteries_to_select)
    print(f"从池子中随机选出的 {config.num_batteries_to_select} 个电池是: {selected_batteries}")

    # 2. 将选出的电池再次随机打乱顺序
    random.shuffle(selected_batteries)

    # 3. 按照设定的数量，切分出验证集和训练集
    val_count = config.num_validation_batteries
    config.val_batteries = selected_batteries[:val_count]
    config.train_batteries = selected_batteries[val_count:]

    # 4. 打印本次运行的最终划分结果，用于记录和检查
    print(f"本次运行的训练集电池: {sorted(config.train_batteries)}")
    print(f"本次运行的验证集电池: {sorted(config.val_batteries)}")
    print(f"本次运行的固定测试集: {sorted(config.test_batteries)}")
    print("--- 随机抽样完成 ---")

    # --- 数据加载与预处理 ---
    try:
        # 【修改1】: 调用新的数据加载函数
        train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data_for_sliding_window(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载失败: {e}")
        return
    joblib.dump(scalers, os.path.join(config.save_path, 'scalers.pkl'))

    # 【修改2】: 增加对空数据集的检查，使代码更健壮
    if train_dataset is None or len(train_dataset) == 0:
        print("错误：训练数据集为空，无法进行训练。请检查电池数据和窗口大小。")
        return

    # 只有在数据集非空时才创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                            pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                             pin_memory=True) if test_dataset else None

    print(f"数据加载完成。")
    print(f"  - 训练集样本数: {len(train_dataset)}")
    print(f"  - 验证集样本数: {len(val_dataset) if val_dataset else 0}")
    print(f"  - 测试集样本数: {len(test_dataset) if test_dataset else 0}")

    # --- 模型、损失函数、优化器定义 ---
    # 【修改3】: 实例化新的 SequentialModel 模型
    model = SequentialModel(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

    metrics_log = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # --- 训练过程 ---
    if config.mode in ['both', 'train']:
        print("\n开始训练模型...")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)

            val_loss = float('inf')
            # 只有在验证集存在时才进行验证
            if val_loader:
                # 【修改4】: evaluate不再返回cycle_nums，调整解包
                val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, config.device)
                print(
                    f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")
                log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                             **{'val_' + k: v for k, v in val_metrics.items()}}
                metrics_log.append(log_entry)
            else:
                # 如果没有验证集，只打印训练损失
                print(f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | (无验证集)")

            # 早停逻辑 (只有在有验证集时才有效)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.save_path, f'best_model_{exp_tag}.pth'))
                print(f"  - 验证损失降低，保存模型。")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break
        print("\n训练完成。")
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv(os.path.join(config.save_path, f'training_metrics_log_{exp_tag}.csv'), index=False)

    # --- 评估过程 ---
    if config.mode in ['both', 'validate']:
        print('\n加载最佳模型进行最终评估...')
        model_path = os.path.join(config.save_path, f'best_model_{exp_tag}.pth')
        if not os.path.exists(model_path):
            print(f"错误: 找不到已训练的模型 '{model_path}'。")
            return
        model.load_state_dict(torch.load(model_path, map_location=config.device))

        print("\n--- 评估结果 ---")
        # 验证集评估
        if val_loader:
            _, val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, config.device)
            print(
                f"最终验证集指标: MSE={val_metrics['MSE']:.6f}, MAE={val_metrics['MAE']:.6f}, R2={val_metrics['R2']:.4f}")
            # 【修改5】: 创建结果DataFrame时不再包含'循环号'列
            val_results_df = pd.DataFrame({'True_Capacity': val_labels, 'Predicted_Capacity': val_preds})
            val_results_df.to_csv(os.path.join(config.save_path, f'validation_predictions_{exp_tag}.csv'), index=False)
            plot_results(val_labels, val_preds, 'Validation Set: True vs. Predicted Capacity',
                         os.path.join(config.save_path, f'validation_plot_{exp_tag}.png'))
            plot_diagonal_results(val_labels, val_preds, 'Validation Set: Diagonal Plot',
                                  os.path.join(config.save_path, f'validation_diagonal_plot_{exp_tag}.png'))

        # 测试集评估
        if test_loader:
            _, test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, config.device)
            print(
                f"最终测试集指标: MSE={test_metrics['MSE']:.6f}, MAE={test_metrics['MAE']:.6f}, R2={test_metrics['R2']:.4f}")
            # 【修改5】: 创建结果DataFrame时不再包含'循环号'列
            test_results_df = pd.DataFrame({'True_Capacity': test_labels, 'Predicted_Capacity': test_preds})
            test_results_df.to_csv(os.path.join(config.save_path, f'test_predictions_{exp_tag}.csv'), index=False)
            plot_results(test_labels, test_preds, 'Test Set: True vs. Predicted Capacity',
                         os.path.join(config.save_path, f'test_plot_{exp_tag}.png'))
            plot_diagonal_results(test_labels, test_preds, 'Test Set: Diagonal Plot',
                                  os.path.join(config.save_path, f'test_diagonal_plot_{exp_tag}.png'))

        print(f"\n评估完成，预测值和图像已保存。")

if __name__ == '__main__':
    main()