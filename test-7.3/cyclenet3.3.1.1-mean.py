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
        self.save_path = '/home/scuee_user06/myh/电池/data/cyclenet_result-150mean/20'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        # 定义用于交叉验证的电池池
        # self.cv_batteries = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 21, 22, 23, 24, 19]
        # # 定义固定的测试集
        # self.test_batteries = [5, 12, 14, 20]
        #
        # # 定义用于交叉验证的电池池
        # self.cv_batteries = [1, 2, 3, 4, 6]
        # # 定义固定的测试集
        # self.test_batteries = [5]
        #
        # 定义用于交叉验证的电池池
        # self.cv_batteries = [7, 8, 9, 10, 11]
        # # 定义固定的测试集
        # self.test_batteries = [12]
        #
        # # 定义用于交叉验证的电池池
        # self.cv_batteries = [13, 15, 16, 17, 18]
        # # 定义固定的测试集
        # self.test_batteries = [14]

        # # 定义用于交叉验证的电池池
        self.cv_batteries = [21, 22, 23, 24, 19]
        # 定义固定的测试集
        self.test_batteries = [20]

        self.features_from_C = [
            # 'ICA峰值位置(V)',
            # '恒流充电时间(s)',
            '恒压充电时间(s)',
            # '恒流与恒压时间比值',
            # '2.8~3.4V放电时间(s)',
            '3.3~3.6V充电时间(s)',
            '弛豫末端电压'
        ]

        # 文件A的输入特征维度 (例如，'弛豫段电压1'到'弛豫段电压6'就是6维)
        self.sequence_feature_dim = 6

        # --- 模型超参数 ---
        self.meta_cycle_len = 6  # 定义一个元周期长度，比如假设电池每100个循环有一个宏观上的周期性变化
        self.sequence_length = 1 # A文件的序列长度 (弛豫段电压的点数)
        self.scalar_feature_dim = len(self.features_from_C) + 1  # B和C文件合并后的标量特征数量 (请根据实际情况调整)
        self.d_model = 256  # 隐藏层维度
        self.d_ff = 1024  # MLP编码器和预测头的中间层维度
        self.cycle_len = 2000  # 最大循环次数 (应大于任何电池的最大循环号)
        self.dropout = 0.2  # Dropout概率，0.1表示随机丢弃10%的神经元
        self.use_revin = False  # 是否使用可逆实例归一化
        self.weight_decay = 0.0001  # 增加权重衰减，1e-4或1e-5是常用的初始值
        self.cycle_amount = 150

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
        self.sequence_encoder = nn.Linear(configs.sequence_length * configs.sequence_feature_dim, configs.d_model // 2)
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

    def forward(self, x_seq, x_scalar, cycle_number):
        # x_seq: (batch, seq_len, 3) -> (batch, 1, 3)
        # x_scalar: (batch, scalar_dim) -> (batch, 10)
        # cycle_number: (batch,) -> 真实的循环号，如 1, 2, ..., 1000

        # --- 1. 特征编码与合并 ---
        # 展平序列输入
        x_seq_flat = x_seq.view(x_seq.size(0), -1)

        seq_embedding = self.sequence_encoder(x_seq_flat)
        scalar_embedding = self.scalar_encoder(x_scalar)

        # 合并成一个统一的特征向量
        combined_features = torch.cat((seq_embedding, scalar_embedding), dim=1)  # 形状: (batch, d_model)

        # --- 2. CycleNet核心：分解周期性 ---
        # 计算在元周期中的位置
        cycle_index = cycle_number % self.configs.meta_cycle_len

        # 从合并的特征中减去学习到的周期成分
        # combined_features 需要增加一个维度以匹配 RecurrentCycle 的输出
        # length=1 因为我们每个样本只有一个组合特征向量
        decycled_features = combined_features - self.cycle_queue(cycle_index, length=1).squeeze(1)

        # --- 3. 预测 ---
        # 将去除了周期性的特征送入预测头
        prediction = self.prediction_head(decycled_features)

        # 注意：这里我们不再需要把周期加回去，因为我们的目标（容量）和输入特征不在一个域，
        # 加上周期没有物理意义。我们只是利用周期分解来帮助模型更好地学习非周期性的退化趋势。

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
            df_c = pd.read_csv(path_c, sep=',')  # 文件C：标量2 + 目标

            # 1. 合并标量特征
            # 现在标量特征只来源于文件C
            df_c.rename(columns=lambda x: x.strip(), inplace=True)
            scalar_df = df_c  # 直接将df_c作为scalar_df

            # 2. 处理序列数据
            # 按'循环号'分组，并将'弛豫段电压'点聚合为一个Numpy数组
            # 三个特征列名
            # 根据 config 中的 sequence_feature_dim 参数动态生成特征列名
            feature_cols = [f'弛豫段电压{i}' for i in range(1, config.sequence_feature_dim + 1)]
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
    full_df = full_df[full_df['循环号'] <= config.cycle_amount].copy()
    # #容量归一化
    # full_df['最大容量(Ah)'] = full_df['最大容量(Ah)'] / 3.5
    # print("已将'最大容量(Ah)'特征列的所有值除以3.5。")

    target_col = '循环号'
    sequence_col = 'voltage_sequence'

    # 由于不再使用文件B，直接从config中获取文件C的特征列表
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

    # 在训练集上拟合缩放器
    # 序列数据需要先展平才能fit
    all_train_sequences = np.vstack(train_df[sequence_col].values)
    scaler_seq.fit(all_train_sequences)

    # 标量数据直接fit
    scaler_scalar.fit(train_df[scalar_feature_cols])

    # 应用缩放
    for df in [train_df, val_df, test_df]:
        # x已经是(seq_len, 3)的形状，可以直接transform
        df[sequence_col] = df[sequence_col].apply(lambda x: scaler_seq.transform(x))
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
    plt.ylabel('Cycle Number', fontsize=12)
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

    plt.xlabel('True Cycle Number', fontsize=12)
    plt.ylabel('Predicted Cycle Number', fontsize=12)
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
    # --- 外部设置 ---
    warnings.filterwarnings('ignore')
    matplotlib.use('Agg')
    config = Config()
    set_seed(config.seed)

    # 创建一个总的实验根目录
    base_save_path = config.save_path
    os.makedirs(base_save_path, exist_ok=True)
    print(f"所有交叉验证结果将保存在根目录: {base_save_path}")
    print(f"使用设备: {config.device}")

    # --- 修改：初始化列表以存储所有折的指标、预测值和真实值 ---
    all_folds_test_metrics = []
    all_folds_test_preds = []
    all_folds_test_labels = []

    # --- 新增：为验证集创建等效列表 ---
    all_folds_val_metrics = []
    all_folds_val_preds = []
    all_folds_val_labels = []

    # --- K折交叉验证循环 ---
    cv_pool = config.cv_batteries
    for i, val_battery_id in enumerate(cv_pool):
        fold_num = i + 1
        print(f"\n{'=' * 20} FOLD {fold_num}/{len(cv_pool)} {'=' * 20}")

        # --- 1. 动态设置当前折的训练集和验证集 ---
        config.val_batteries = [val_battery_id]
        config.train_batteries = [b_id for b_id in cv_pool if b_id != val_battery_id]

        print(f"当前折验证集: {config.val_batteries}")
        print(f"当前折训练集: {config.train_batteries}")
        print(f"固定测试集: {config.test_batteries}")

        # --- 2. 为当前折创建独立的保存路径 ---
        exp_tag = get_exp_tag(config)
        fold_save_path = os.path.join(base_save_path, f'fold_{fold_num}_{exp_tag}')
        os.makedirs(fold_save_path, exist_ok=True)
        print(f"本折结果保存至: {fold_save_path}")

        # --- 3. 加载和预处理数据 ---
        try:
            train_dataset, val_dataset, test_dataset, scalers = load_and_preprocess_data(config)
        except (FileNotFoundError, ValueError) as e:
            print(f"数据加载失败: {e}")
            continue

        joblib.dump(scalers, os.path.join(fold_save_path, 'scalers.pkl'))

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True)

        # --- 4. 训练模型 ---
        model = CycleNetForSOH(config).to(config.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        grad_scaler = GradScaler() if config.use_gpu and config.device.type == 'cuda' else None

        best_val_loss = float('inf')
        epochs_no_improve = 0

        print("\n开始训练模型...")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, config.device, grad_scaler)
            val_loss, _, _, _, _ = evaluate(model, val_loader, criterion, config.device)
            print(f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(fold_save_path, f'best_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break

        # --- 5. 评估、保存和收集结果 ---
        print('\n加载本折最佳模型进行最终评估...')
        model_path = os.path.join(fold_save_path, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"错误: 找不到已训练的模型 '{model_path}'。")
            continue

        model.load_state_dict(torch.load(model_path, map_location=config.device))

        _, val_metrics, val_preds, val_labels, val_cycle_nums = evaluate(model, val_loader, criterion, config.device)
        _, test_metrics, test_preds, test_labels, test_cycle_nums = evaluate(model, test_loader, criterion,
                                                                             config.device)

        # --- 修改：收集本折的测试指标、预测值和真实值 ---
        all_folds_test_metrics.append(test_metrics)
        all_folds_test_preds.append(test_preds)
        all_folds_test_labels.append(test_labels)

        # --- 新增：收集本折的验证指标、预测值和真实值 ---
        all_folds_val_metrics.append(val_metrics)
        all_folds_val_preds.append(val_preds)
        all_folds_val_labels.append(val_labels)

        print(f"\n--- FOLD {fold_num} 评估结果 ---")
        print(f"验证集指标: R2={val_metrics['R2']:.4f}")
        print(f"固定测试集指标: R2={test_metrics['R2']:.4f}")

        # 保存本折的详细预测CSV文件
        val_results_df = pd.DataFrame(
            {'Original_Cycle_Index': val_cycle_nums, 'True_Cycle': val_labels, 'Predicted_Cycle': val_preds})
        test_results_df = pd.DataFrame(
            {'Original_Cycle_Index': test_cycle_nums, 'True_Cycle': test_labels, 'Predicted_Cycle': test_preds})
        val_results_df.to_csv(os.path.join(fold_save_path, 'validation_predictions.csv'), index=False)
        test_results_df.to_csv(os.path.join(fold_save_path, 'test_predictions.csv'), index=False)

        # 绘制本折的图表
        plot_diagonal_results(val_labels, val_preds, f'Fold {fold_num} Validation Set: Diagonal Plot',
                              os.path.join(fold_save_path, 'validation_diagonal_plot.png'))
        plot_diagonal_results(test_labels, test_preds, f'Fold {fold_num} Test Set: Diagonal Plot',
                              os.path.join(fold_save_path, 'test_diagonal_plot.png'))
    # --- 交叉验证循环结束后，进行最终的汇总、保存和绘图 ---
    if not all_folds_test_metrics:
        print("\n没有一折成功完成，无法计算最终结果。")
        return

    print(f"\n{'=' * 20} 交叉验证最终汇总 {'=' * 20}")
    base_save_path = config.save_path  # 获取根保存路径

    # --- 1. 验证集结果汇总 (通过拼接) ---
    print("\n--- 正在汇总所有验证集结果 (拼接)... ---")
    # 拼接所有验证集的预测和标签
    agg_val_preds = np.concatenate(all_folds_val_preds)
    agg_val_labels = np.concatenate(all_folds_val_labels)

    # 基于拼接后的结果计算总体验证指标
    final_val_mse = mean_squared_error(agg_val_labels, agg_val_preds)
    final_val_mae = mean_absolute_error(agg_val_labels, agg_val_preds)
    final_val_rmse = np.sqrt(final_val_mse)
    final_val_r2 = r2_score(agg_val_labels, agg_val_preds)
    final_val_metrics = {'MSE': final_val_mse, 'MAE': final_val_mae, 'RMSE': final_val_rmse, 'R2': final_val_r2}

    print("总体验证集指标 (所有折叠拼接后):")
    print(pd.Series(final_val_metrics))

    # 保存拼接后的验证集预测结果
    agg_val_df = pd.DataFrame({'True_Cycle': agg_val_labels, 'Predicted_Cycle': agg_val_preds})
    agg_val_csv_path = os.path.join(base_save_path, 'final_aggregated_validation_predictions.csv')
    agg_val_df.to_csv(agg_val_csv_path, index=False)
    print(f"拼接后的验证集预测已保存至: {agg_val_csv_path}")

    # 绘制拼接后的验证集图表
    plot_diagonal_results(agg_val_labels, agg_val_preds, 'Aggregated Validation Set: Diagonal Plot (All Folds)',
                          os.path.join(base_save_path, 'final_aggregated_validation_diagonal_plot.png'))
    print("已保存拼接后的验证集图表。")

    # --- 2. 测试集结果汇总 (通过平均预测) ---
    print("\n--- 正在汇总固定测试集结果 (平均预测)... ---")
    # 对所有折的预测值求平均
    # all_folds_test_preds 是一个列表，其中每个元素是当前折的预测数组
    # 使用 np.mean(axis=0) 来计算每个样本点的平均预测值
    mean_test_preds = np.mean(np.array(all_folds_test_preds), axis=0)

    # 因为测试集是固定的，所以所有折的真实标签都一样，取第一个即可
    final_test_labels = all_folds_test_labels[0]

    # 基于平均预测计算最终测试指标
    final_test_mse = mean_squared_error(final_test_labels, mean_test_preds)
    final_test_mae = mean_absolute_error(final_test_labels, mean_test_preds)
    final_test_rmse = np.sqrt(final_test_mse)
    final_test_r2 = r2_score(final_test_labels, mean_test_preds)
    final_test_metrics = {'MSE': final_test_mse, 'MAE': final_test_mae, 'RMSE': final_test_rmse,
                          'R2': final_test_r2}

    print("\n最终测试集指标 (基于平均预测):")
    print(pd.Series(final_test_metrics))

    # 保存平均后的测试集预测结果
    mean_test_df = pd.DataFrame({'True_Cycle': final_test_labels, 'Predicted_Cycle': mean_test_preds})
    mean_test_csv_path = os.path.join(base_save_path, 'final_mean_test_predictions.csv')
    mean_test_df.to_csv(mean_test_csv_path, index=False)
    print(f"平均后的测试集预测已保存至: {mean_test_csv_path}")

    # 绘制基于平均预测的测试集图表
    plot_diagonal_results(final_test_labels, mean_test_preds, 'Test Set with Mean Prediction: Diagonal Plot',
                          os.path.join(base_save_path, 'final_mean_test_diagonal_plot.png'))
    print("已保存基于平均预测的测试集图表。")

    # --- 3. 保存每个折叠在测试集上的详细指标 ---
    print("\n--- 正在保存各折在固定测试集上的指标详情... ---")
    metrics_df = pd.DataFrame(all_folds_test_metrics)
    # 计算均值和标准差以供参考
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    # 创建一个汇总DataFrame
    summary_df = metrics_df.copy()
    summary_df.loc['mean'] = mean_metrics
    summary_df.loc['std'] = std_metrics
    summary_df.index = [f'Fold_{i + 1}' for i in range(len(all_folds_test_metrics))] + ['mean', 'std']

    summary_path = os.path.join(base_save_path, 'final_cv_test_metrics_summary.csv')
    summary_df.to_csv(summary_path)

    print("各折在固定测试集上的指标详情:")
    print(summary_df.to_string())
    print(f"\n交叉验证的详细指标及均值已保存到: {summary_path}")


if __name__ == '__main__':
    main()
