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


# --- 1. 配置参数 (已修改) ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 (修改为三个输入路径) ---
        self.path_A_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'  # A文件: 弛豫段电压序列 (1200点/循环)
        # self.path_B_scalar = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/End'  # B文件: 弛豫末端电压 (1点/循环)
        self.path_C_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'  # C文件: 其他特征和目标 (1行/循环)
        self.save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/attention-LSTM/20'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        # 这里手动分配电池编号
        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 11, 13, 14, 15, 18, 21, 22, 23, 24]
        # self.val_batteries = [5, 10, 17, 19]
        # self.test_batteries = [6, 12, 16, 20]

        # self.train_batteries = [1, 2, 3, 4]
        # self.val_batteries = [5]
        # self.test_batteries = [6]

        # self.train_batteries = [7, 8, 9, 11]
        # self.val_batteries = [10]
        # self.test_batteries = [12]

        # self.train_batteries = [15, 13, 18, 14]
        # self.val_batteries = [17]
        # self.test_batteries = [16]
        #
        self.train_batteries = [21, 22, 23, 24]
        self.val_batteries = [19]
        self.test_batteries = [20]

        self.features_from_C = [
            # 'ICA峰值位置(V)',
            # '恒流充电时间(s)',
            '恒压充电时间(s)',
            # '恒流与恒压时间比值',
            # '2.8~3.4V放电时间(s)',
            '3.3~3.6V充电时间(s)',
            # '弛豫末端电压'
        ]

        # 文件A的输入特征维度 (例如，'弛豫段电压1'到'弛豫段电压6'就是6维)
        self.sequence_feature_dim = 7

        # --- 模型超参数 (为Attention-LSTM调整) ---
        # self.meta_cycle_len = 7      # <--- 不再需要，可以删除或注释掉
        self.sequence_length = 1  # <--- 注意：对于LSTM，这个值通常>1效果更好
        #      但你当前的数据预处理逻辑是固定为1的，所以我们先保持不变
        self.scalar_feature_dim = len(self.features_from_C)

        # 这里的 d_model 可以理解为 LSTM 的 hidden_size
        self.d_model = 256  # LSTM隐藏层维度
        self.d_ff = 1024  # 预测头的中间层维度

        # self.cycle_len = 2000        # <--- 不再需要，可以删除或注释掉
        self.dropout = 0.2
        self.weight_decay = 0.0001

        # --- 训练参数 ---
        self.epochs = 500
        self.batch_size = 128
        self.learning_rate = 0.001
        self.patience = 15
        self.seed = 2025  # 这个种子将不再被直接使用，而是作为参考
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


# --- 3. 新的多模态模型定义 (Attention-LSTM 版本) ---

# 首先，我们定义一个 Attention 模块
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, sequence_length, hidden_dim)

        # 计算注意力分数
        energy = torch.tanh(self.attn(lstm_output))  # (batch, seq_len, hidden_dim)

        # attention_scores shape: (batch, seq_len, 1)
        attention_scores = self.v(energy)

        # attention_weights shape: (batch_size, sequence_length, 1)
        # 使用 softmax 在序列长度维度上归一化，得到权重
        weights = torch.softmax(attention_scores, dim=1)

        # context_vector shape: (batch_size, hidden_dim)
        # 将权重应用于LSTM的输出，得到加权的上下文向量
        context_vector = torch.sum(weights * lstm_output, dim=1)

        return context_vector, weights


# 然后，我们定义新的主模型 AttentionLSTM
class AttentionLSTM(nn.Module):
    def __init__(self, configs):
        super(AttentionLSTM, self).__init__()
        self.configs = configs

        # 1. LSTM 层处理序列数据
        # configs.d_model 可以被看作是 lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=configs.sequence_feature_dim,
            hidden_size=configs.d_model,
            num_layers=2,  # 可以设为超参数，这里用2层
            batch_first=True,
            dropout=configs.dropout if 2 > 1 else 0,  # 只有多层LSTM才能用dropout
            bidirectional=False  # 是否使用双向LSTM
        )

        # 2. Attention 层
        # 注意力层的输入维度需要和LSTM的输出维度一致
        self.attention = Attention(configs.d_model)

        # 3. 标量数据编码器
        # 将标量特征编码，使其维度可以和序列特征的上下文向量拼接
        self.scalar_encoder = nn.Linear(configs.scalar_feature_dim, configs.d_model // 2)

        # 4. 预测头
        # 输入维度是 attention上下文向量(d_model)和标量编码(d_model//2)的拼接
        self.prediction_head = nn.Sequential(
            nn.Linear(configs.d_model + configs.d_model // 2, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, 1)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        # cycle_number 在这个模型中不会被使用，但为了保持接口一致性，我们仍然接收它

        # x_seq: (batch, seq_len, feature_dim)
        # x_scalar: (batch, scalar_dim)

        # --- 1. LSTM处理序列 ---
        # lstm_out shape: (batch, seq_len, hidden_size)
        # h_n, c_n shape: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x_seq)

        # --- 2. Attention机制计算上下文向量 ---
        # context_vector shape: (batch, hidden_size)
        context_vector, _ = self.attention(lstm_out)

        # --- 3. 编码标量特征 ---
        # scalar_embedding shape: (batch, d_model // 2)
        scalar_embedding = self.scalar_encoder(x_scalar)

        # --- 4. 拼接特征并预测 ---
        # combined_features shape: (batch, d_model + d_model // 2)
        combined_features = torch.cat((context_vector, scalar_embedding), dim=1)

        prediction = self.prediction_head(combined_features)

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
            # print(f"正在处理电池 {battery_id}...")
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

            # 2. 处理序列数据
            # 按'循环号'分组，并将'弛豫段电压'点聚合为一个Numpy数组
            # 三个特征列名
            # 根据 config 中的 sequence_feature_dim 参数动态生成特征列名
            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(
                name='voltage_sequence')
            # print("重命名之后的列名:", sequence_df.columns)  # 加上这行来查看
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
            raise ValueError(
                f"您手动选择的特征 '{col}' 不存在于加载的数据中。请检查 Config.features_from_C 中的拼写和列名。")

    # 更新config中的特征维度
    config.scalar_feature_dim = len(scalar_feature_cols)
    # print(f"已手动选择 {config.scalar_feature_dim} 个标量特征: {scalar_feature_cols}")

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
            all_cycle_indices.append(batch_cycle_idx.cpu().numpy())  # <--- 新增：收集当前批次的循环号

    avg_loss = total_loss / len(dataloader)

    # 拼接所有批次的预测和标签
    predictions = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    cycle_indices = np.concatenate(all_cycle_indices).flatten()  # <--- 新增：将所有循环号拼接成一个数组

    mse = mean_squared_error(labels, predictions)
    mape = mean_absolute_percentage_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)

    metrics = {'MSE': mse, 'MAPE': mape, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
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
def main():
    # 忽略一些不必要的警告
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    config = Config()

    # 确保主保存路径存在
    os.makedirs(config.save_path, exist_ok=True)
    print(f"所有实验的总保存路径: {config.save_path}")
    print(f"使用设备: {config.device}")

    # --- 新增：为多次实验设置的变量 ---
    num_runs = 5
    all_runs_metrics = []
    best_run_val_loss = float('inf')
    best_run_dir = None
    best_run_number = -1
    all_runs_PER_BATTERY_metrics = []

    # --- 开始多次实验循环 ---
    for run_number in range(1, num_runs + 1):
        # --- 1. 设置当前轮的随机种子和保存路径 ---
        current_seed = random.randint(0, 99999)
        set_seed(current_seed)

        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)

        print(f"\n{'=' * 30}")
        print(f" 开始第 {run_number}/{num_runs} 次实验 | 随机种子: {current_seed} ")
        print(f" 本次实验结果将保存到: {run_save_path}")
        print(f"{'=' * 30}")

        # --- 2. 数据加载和预处理 (每轮都一样，但为保持独立性放在循环内) ---
        try:
            train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
        except (FileNotFoundError, ValueError) as e:
            print(f"数据加载失败: {e}")
            continue  # 如果数据加载失败，跳过此次运行

        # 保存scaler字典到当前运行的文件夹
        joblib.dump(scalers, os.path.join(run_save_path, 'scalers.pkl'))

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True)

        print(f"数据加载完成。训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        # --- 3. 模型初始化和训练 ---
        model = AttentionLSTM(config).to(config.device)
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
                    # 模型保存在当前轮次的文件夹中
                    torch.save(model.state_dict(), os.path.join(run_save_path, 'best_model.pth'))
                    print(f"  - 验证损失降低，保存模型到 {os.path.join(run_save_path, 'best_model.pth')}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= config.patience:
                        print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                        break
            print("\n训练完成。")
            metrics_df = pd.DataFrame(metrics_log)
            metrics_df.to_csv(os.path.join(run_save_path, 'training_metrics_log.csv'), index=False)

        # --- 4. 评估最佳模型 (已重构，支持按电池分别评估和绘图) ---
        if config.mode in ['both', 'validate']:
            print('\n加载本轮最佳模型进行最终评估...')
            model_path = os.path.join(run_save_path, 'best_model.pth')
            if not os.path.exists(model_path):
                print(f"错误: 找不到已训练的模型 '{model_path}'。")
                continue

            model.load_state_dict(torch.load(model_path, map_location=config.device))
            scalers = joblib.load(os.path.join(run_save_path, 'scalers.pkl'))
            scaler_target = scalers['target']

            # 评估整个测试集
            _, _, test_preds, test_labels, test_cycle_nums = evaluate(model, test_loader, criterion,
                                                                      config.device)

            # 反归一化
            test_preds_orig = scaler_target.inverse_transform(test_preds.reshape(-1, 1)).flatten()
            test_labels_orig = scaler_target.inverse_transform(test_labels.reshape(-1, 1)).flatten()

            # 这会钳制任何模型误预测的负值，使其归零。
            test_preds_orig = np.clip(test_preds_orig, a_min=0.0, a_max=None)

            # ---【核心修改】开始：针对每个测试电池分别计算指标和绘图 ---
            print("\n--- 本轮评估结果 (按单电池) ---")

            # 因为 test_loader 的 shuffle=False，预测顺序与 test_dataset.df 严格一致
            # 我们可以直接从 test_dataset 中获取电池ID
            eval_df = pd.DataFrame({
                'battery_id': test_dataset.df['battery_id'].values,
                'cycle': test_cycle_nums,
                'true': test_labels_orig,
                'pred': test_preds_orig
            })

            per_battery_metrics_list = []
            for batt_id in config.test_batteries:
                batt_df = eval_df[eval_df['battery_id'] == batt_id]
                if batt_df.empty:
                    print(f"  - 电池 {batt_id}: 未找到数据，跳过。")
                    continue

                batt_true = batt_df['true'].values
                batt_pred = batt_df['pred'].values

                # 1. 计算单独指标
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
                # ---【新增逻辑】开始：将本次的单独电池指标也添加到“跨实验汇总”的总列表中 ---
                batt_metrics_with_run_info = batt_metrics_dict.copy()
                batt_metrics_with_run_info['run'] = run_number
                batt_metrics_with_run_info['seed'] = current_seed
                all_runs_PER_BATTERY_metrics.append(batt_metrics_with_run_info)
                # ---【新增逻辑】结束 ---

                # 2. 绘制单独的曲线图
                plot_results(batt_true, batt_pred,
                             f'Run {run_number} Battery {batt_id}: True vs Predicted Capacity',
                             os.path.join(run_save_path, f'test_plot_battery_{batt_id}.png'))

                # 3. 绘制单独的对角图
                plot_diagonal_results(batt_true, batt_pred,
                                      f'Run {run_number} Battery {batt_id}: Diagonal Plot',
                                      os.path.join(run_save_path, f'test_diagonal_plot_battery_{batt_id}.png'))

            # 4. 保存每个电池的指标汇总
            per_batt_df = pd.DataFrame(per_battery_metrics_list)
            per_batt_df.to_csv(os.path.join(run_save_path, 'test_per_battery_metrics.csv'), index=False)
            print(f"  -> 单独指标和图表已保存至: {run_save_path}")
            # --- 【核心修改】结束 ---

            # --- 评估结果 (所有测试电池汇总) ---
            print("\n--- 本轮评估结果 (所有测试电池汇总) ---")
            final_test_metrics = {
                'MAE': mean_absolute_error(test_labels_orig, test_preds_orig),
                'MAPE': mean_absolute_percentage_error(test_labels_orig, test_preds_orig),
                'MSE': mean_squared_error(test_labels_orig, test_preds_orig),
                'RMSE': np.sqrt(mean_squared_error(test_labels_orig, test_preds_orig)),
                'R2': r2_score(test_labels_orig, test_preds_orig)
            }
            # 【新增】保存总体指标CSV
            pd.DataFrame([final_test_metrics]).to_csv(os.path.join(run_save_path, 'test_overall_metrics.csv'),
                                                      index=False)

            # 记录本轮实验的总体指标 (用于5次run的对比)
            current_run_summary = {'run': run_number, 'seed': current_seed, **final_test_metrics}
            all_runs_metrics.append(current_run_summary)

            print(
                f"测试集(汇总): MSE={final_test_metrics['MSE']:.6f}, MAE={final_test_metrics['MAE']:.6f}, RMSE={final_test_metrics['RMSE']:.6f}, R2={final_test_metrics['R2']:.4f}")

            # 保存本轮次的 *总体* 预测结果和图表
            test_results_df = pd.DataFrame(
                {'Original_Cycle_Index': test_cycle_nums, 'True_Capacity_Ah': test_labels_orig,
                 'Predicted_Capacity_Ah': test_preds_orig})
            test_results_df.to_csv(os.path.join(run_save_path, 'test_ALL_predictions.csv'), index=False)

            # (注意：之前的总体绘图函数已被上面的循环内单独绘图替代，故注释掉)
            # plot_results(test_labels_orig, test_preds_orig, ...)
            # plot_diagonal_results(test_labels_orig, test_preds_orig, ...)

            print(f"本轮所有预测和图表已保存。")

            # --- 5. 检查是否为最佳轮次 ---
            if best_val_loss_this_run < best_run_val_loss:
                best_run_val_loss = best_val_loss_this_run
                best_run_dir = run_save_path
                best_run_number = run_number
                print(f"*** 新的最佳表现！验证集损失: {best_val_loss_this_run:.6f} ***")

    # --- 循环结束后 ---
    print(f"\n\n{'=' * 50}")
    print(" 所有实验均已完成。")
    print(f"{'=' * 50}")

    # 1. 保存所有轮次的指标汇总
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 ---")
        print(summary_df)
        print(f"\n汇总指标已保存到: {summary_path}")

    # 2. 将最佳轮次的结果复制到主目录
    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (验证集损失最低: {best_run_val_loss:.6f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")

        # 遍历最佳运行目录中的所有文件
        for filename in os.listdir(best_run_dir):
            source_file = os.path.join(best_run_dir, filename)
            destination_file = os.path.join(config.save_path, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)  # copy2 会同时复制元数据

        print("最佳结果复制完成。")
    else:
        print("未能确定最佳实验轮次。")

    print("\n正在生成所有实验的“分电池”详细汇总报告...")
    if all_runs_PER_BATTERY_metrics:
        # 将收集到的所有分电池指标列表转换为DataFrame
        per_batt_summary_df = pd.DataFrame(all_runs_PER_BATTERY_metrics)

        # 调整列顺序，方便阅读
        all_cols = list(per_batt_summary_df.columns)
        # 定义核心列的理想顺序
        core_cols = ['Battery_ID', 'run', 'seed', 'MAE', 'MAPE', 'MSE', 'RMSE', 'R2']
        # 生成最终排序列（确保所有核心列在前，其他列在后）
        ordered_cols = [col for col in core_cols if col in all_cols] + [col for col in all_cols if col not in core_cols]
        per_batt_summary_df = per_batt_summary_df[ordered_cols]

        # 按电池ID和运行次数排序
        per_batt_summary_df = per_batt_summary_df.sort_values(by=['Battery_ID', 'run'])

        # 保存为您想要的汇总文件
        summary_path_per_batt = os.path.join(config.save_path, 'all_runs_per_battery_summary.csv')
        per_batt_summary_df.to_csv(summary_path_per_batt, index=False)
        print(f"“分电池”详细汇总报告已保存到: {summary_path_per_batt}")
    else:
        print("未能生成“分电池”详细汇总报告，因为没有收集到数据。")


if __name__ == '__main__':
    main()