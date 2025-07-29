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


# 忽略一些不必要的警告
warnings.filterwarnings('ignore')


# --- 1. 配置参数  ---
class Config:
    def __init__(self):
        # --- 数据和路径设置 ---
        self.data_path = '/home/scuee_user06/myh/电池/data/selected_feature/test'  # 存放24个电池CSV数据的文件夹路径
        self.save_path = '/home/scuee_user06/myh/电池/data/cyclenet_result'  # 保存模型、结果和图像的文件夹路径

        # --- 数据集划分 ---
        # 这里手动分配电池编号
        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22]
        # self.val_batteries = [5, 11, 17, 23]
        # self.test_batteries = [6, 12, 18, 24]
        self.train_batteries = [1]
        self.val_batteries = [5]
        self.test_batteries = [6]

        # --- 模型超参数 ---
        self.model_type = 'mlp'  # 模型类型: 'linear' 或 'mlp'
        self.seq_len = 10  # 输入序列长度 (用前10个循环的数据预测下一个)
        self.pred_len = 1  # 预测序列长度 (预测未来1个循环)
        self.enc_in = 11  # 输入特征维度 (根据您的数据列)
        self.d_model = 128  # MLP模型的隐藏层维度
        self.cycle = 1200  # 最大循环次数
        self.use_revin = False  # 是否使用可逆实例归一化 (Reversible Instance Normalization)

        # --- 训练参数 ---
        self.epochs = 10
        self.batch_size = 2048
        self.learning_rate = 0.01
        self.patience = 10  # Early stopping的耐心值
        self.seed = 2025  # 固定随机种子
        self.mode = 'validate'  # 可选 'train', 'validate', 'both'

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


# --- 3. CycleNet 模型定义 (来自用户提供的 CycleNet.py) ---
class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x - self.cycleQueue(cycle_index, self.seq_len)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = y + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean
        return y


# --- 4. 数据集定义 ---
# 加载csv
class BatteryDataset(Dataset):
    def __init__(self, data, feature_cols, target_col, seq_len, pred_len):
        # 重置索引
        data = data.reset_index(drop=True)

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.features = data[self.feature_cols].values
        self.target = data[self.target_col].values
        self.cycle_index = data['循环号'].values

        # 创建一个索引映射，以防止序列跨越不同的电池
        self.indices = []
        for battery_id in data['battery_id'].unique():
            battery_indices = data[data['battery_id'] == battery_id].index
            start = battery_indices[0]
            end = battery_indices[-1]
            # 确保有足够的长度来创建至少一个序列
            if len(battery_indices) >= seq_len + pred_len:
                # `i` 是序列的起始点在全局DataFrame中的索引
                for i in range(start, end - seq_len - pred_len + 2):
                    self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        pred_start_idx = end_idx + self.pred_len - 1

        x = self.features[start_idx:end_idx]
        y = self.target[pred_start_idx: pred_start_idx + 1]  # 预测单个值
        cycle_idx = self.cycle_index[start_idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(cycle_idx,
                                                                                                        dtype=torch.long)

# --- 5. 数据加载和预处理 ---
# 加载CSV
def load_and_preprocess_data(config):
    """加载所有电池数据，进行预处理和划分"""
    all_files = [f for f in os.listdir(config.data_path) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"在 '{config.data_path}' 文件夹中没有找到任何 .csv 文件。")

    df_list = []
    for file in all_files:
        try:
            battery_id = int(''.join(filter(str.isdigit, file)))
            file_path = os.path.join(config.data_path, file)
            # 假设列是用制表符分隔的
            df = pd.read_csv(file_path, sep=',', encoding='gbk')
            df['battery_id'] = battery_id
            df_list.append(df)
        except Exception as e:
            print(f"读取或处理文件 {file} 时出错: {e}")
            continue

    if not df_list:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(df_list, ignore_index=True)

    # 定义特征和目标
    feature_cols = [col for col in full_df.columns if col not in ['最大容量(Ah)', 'battery_id', '电池编号']]
    # 确保'循环号'在特征列中
    if '循环号' not in feature_cols:
        feature_cols.insert(0, '循环号')
    config.enc_in = len(feature_cols)  # 更新特征数量

    target_col = '最大容量(Ah)'

    # 划分数据集
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)]
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)]
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)]

    # 特征缩放
    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    # 创建Dataset
    train_dataset = BatteryDataset(train_df, feature_cols, target_col, config.seq_len, config.pred_len)
    val_dataset = BatteryDataset(val_df, feature_cols, target_col, config.seq_len, config.pred_len)
    test_dataset = BatteryDataset(test_df, feature_cols, target_col, config.seq_len, config.pred_len)

    return train_dataset, val_dataset, test_dataset, scaler, feature_cols


def inverse_transform_capacity(y, scaler, feature_cols):
    # y 一维容量数组，scaler为已保存的StandardScaler
    idx = feature_cols.index('最大容量(Ah)') if '最大容量(Ah)' in feature_cols else -1
    tmp = np.zeros((len(y), len(feature_cols)))
    tmp[:, idx] = y
    y_inv = scaler.inverse_transform(tmp)[:, idx]
    return y_inv


# --- 6. 训练函数 ---
def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, scaler):
    model.train()
    total_loss = 0
    for batch_x, batch_y, batch_cycle_idx in dataloader:
        batch_x, batch_y, batch_cycle_idx = batch_x.to(device), batch_y.to(device), batch_cycle_idx.to(device)

        optimizer.zero_grad()

        # 如果使用GPU和AMP，则启用 autocast
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(batch_x, batch_cycle_idx)[:, :, 0].squeeze(-1)
                loss = criterion(outputs, batch_y.squeeze(-1))

            # Scaler进行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # 如果不使用GPU，则按常规流程
            outputs = model(batch_x, batch_cycle_idx)[:, :, 0].squeeze(-1)
            loss = criterion(outputs, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(dataloader)

# --- 7. 验证/测试函数 ---
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y, batch_cycle_idx in dataloader:
            batch_x, batch_y, batch_cycle_idx = batch_x.to(device), batch_y.to(device), batch_cycle_idx.to(device)

            outputs = model(batch_x, batch_cycle_idx)[:, :, 0].squeeze(-1)
            loss = criterion(outputs, batch_y.squeeze(-1))

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy().squeeze(-1))

    avg_loss = total_loss / len(dataloader)

    # 将所有批次的预测和标签连接起来
    predictions = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    # 计算评估指标
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

    return avg_loss, metrics, predictions, labels

# 获取参数信息，组合成标签字符串
def get_exp_tag(config):
    train_ids = '-'.join([str(i) for i in config.train_batteries])
    val_ids = '-'.join([str(i) for i in config.val_batteries])
    test_ids = '-'.join([str(i) for i in config.test_batteries])
    tag = (
        f"train{train_ids}_val{val_ids}_test{test_ids}_"
        f"ep{config.epochs}_bs{config.batch_size}_lr{config.learning_rate}_"
        f"dm{config.d_model}_sl{config.seq_len}"
    )
    return tag


# --- 8. 可视化函数 ---
def plot_results(labels, preds, title, save_path):
    """绘制真实值与预测值的对比图"""
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='True Labels')
    plt.plot(preds, label='Predictions', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('max capacity(Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# --- 9. 主执行函数 ---
def main():
    # 1. 初始化配置
    config = Config()

    # 2. 设置随机种子
    set_seed(config.seed)

    # 3. 根据参数创建唯一的实验文件夹路径

    # 使用您已有的函数生成唯一标签
    exp_tag = get_exp_tag(config)

    # 将原始的 save_path 作为根目录
    base_save_path = config.save_path

    # 更新 config.save_path，使其指向本次实验的专属文件夹
    # 后续所有代码都会使用这个新的、唯一的路径
    config.save_path = os.path.join(base_save_path, exp_tag)

    # 创建这个专属文件夹（如果不存在）
    os.makedirs(config.save_path, exist_ok=True)

    print(f"本次实验结果将保存到: {config.save_path}")

    # # 4. 设置matplotlib支持中文
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    print(f"使用设备: {config.device}")

    # 5. 加载数据，并保存归一化器
    try:
        # 注意load_and_preprocess_data需return feature_cols
        train_dataset, val_dataset, test_dataset, scaler, feature_cols = load_and_preprocess_data(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"数据加载失败: {e}")
        return
    joblib.dump(scaler, os.path.join(config.save_path, 'scaler.pkl'))

    # 为DataLoader增加 num_workers 和 pin_memory
    # num_workers 的值可以设为 4, 8, 16 等，取决于你的CPU核心数，可以多尝试几个值以找到最佳性能
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=16,  # 显著提升性能
                              pin_memory=True)  # 加速数据到GPU的传输

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=16,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=16,
                             pin_memory=True)
    print(f"数据加载完成。训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}, 测试集样本数: {len(test_dataset)}")

    # 6. 初始化模型、损失函数、优化器、调度器
    model = Model(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    # 只有在使用GPU时才需要GradScaler 的初始化
    scaler = GradScaler() if config.device.type == 'cuda' else None

    # 7. 反归一化函数
    def inverse_transform_capacity(y, scaler, feature_cols):
        idx = feature_cols.index('最大容量(Ah)') if '最大容量(Ah)' in feature_cols else -1
        tmp = np.zeros((len(y), len(feature_cols)))
        tmp[:, idx] = y
        y_inv = scaler.inverse_transform(tmp)[:, idx]
        return y_inv

    # 8. 模式选择
    metrics_log = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if config.mode in ['both', 'train']:
        print("\n开始训练模型...")
        for epoch in range(config.epochs):
            # 将 scaler 作为参数传递给 train_epoch 函数
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, config.device, scaler)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, config.device)

            print(f"Epoch {epoch + 1}/{config.epochs} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | 验证 R2: {val_metrics['R2']:.4f}")

            log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                         **{'val_' + k: v for k, v in val_metrics.items()}}
            metrics_log.append(log_entry)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.save_path, f'best_model_{exp_tag}.pth'))
                print(f"  - 验证损失降低，保存模型到 {os.path.join(config.save_path, f'best_model_{exp_tag}.pth')}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break
        print("\n训练完成。")
        # 保存训练日志
        metrics_df = pd.DataFrame(metrics_log)
        metrics_df.to_csv(os.path.join(config.save_path, f'training_metrics_log_{exp_tag}.csv'), index=False)
        if config.mode == 'train':
            print('只训练模型并退出...')
            return

    if config.mode in ['both', 'validate']:
        print('加载最佳模型进行最终评估...')
        model.load_state_dict(torch.load(os.path.join(config.save_path, f'best_model_{exp_tag}.pth')))
        scaler = joblib.load(os.path.join(config.save_path, 'scaler.pkl'))

        # 在验证集上评估
        _, val_metrics, val_preds, val_labels = evaluate(model, val_loader, criterion, config.device)
        # 在测试集上评估
        _, test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, config.device)

        # --- 修改：由于目标值从未被归一化，我们不需要反归一化 ---
        # 直接使用evaluate函数返回的真实值和预测值即可
        val_labels_inv = val_labels
        val_preds_inv = val_preds
        test_labels_inv = test_labels
        test_preds_inv = test_preds

        # 保存所有指标 (后续代码不变)
        final_metrics = pd.DataFrame([
            {'set': 'validation', **val_metrics},
            {'set': 'test', **test_metrics}
        ])
        final_metrics.to_csv(os.path.join(config.save_path, 'final_evaluation_metrics.csv'), index=False)
        print(f"最终验证集指标: MSE={val_metrics['MSE']:.6f}, MAE={val_metrics['MAE']:.6f}, RMSE={val_metrics['RMSE']:.6f}, R2={val_metrics['R2']:.4f}")
        print(f"最终测试集指标: MSE={test_metrics['MSE']:.6f}, MAE={test_metrics['MAE']:.6f}, RMSE={test_metrics['RMSE']:.6f}, R2={test_metrics['R2']:.4f}")

        # 保存预测值（反归一化）
        val_results_df = pd.DataFrame({'True_Capacity': val_labels_inv, 'Predicted_Capacity': val_preds_inv})
        test_results_df = pd.DataFrame({'True_Capacity': test_labels_inv, 'Predicted_Capacity': test_preds_inv})
        val_results_df.to_csv(os.path.join(config.save_path, f'validation_predictions_{exp_tag}.csv'), index=False)
        test_results_df.to_csv(os.path.join(config.save_path, f'test_predictions_{exp_tag}.csv'), index=False)
        print(f"验证集和测试集的预测值已保存。")

        # 可视化结果（反归一化）
        plot_results(val_labels_inv, val_preds_inv, f'verification set: True vs predict',
                     os.path.join(config.save_path, f'validation_plot_{exp_tag}.png'))
        plot_results(test_labels_inv, test_preds_inv, f'test set: True vs predict',
                     os.path.join(config.save_path, f'test_plot_{exp_tag}.png'))
        print(f"结果对比图已保存。")


if __name__ == '__main__':
    main()
