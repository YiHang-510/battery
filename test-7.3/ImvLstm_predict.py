import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import random
import os
import matplotlib

# 设置matplotlib以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

# --- 模型定义 ---
class IMVTensorLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super(IMVTensorLSTM, self).__init__()
        self.input_dim = input_dim
        self.n_units = n_units
        self.output_dim = output_dim

        # LSTM-like gates
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.zeros(input_dim, n_units))
        self.b_i = nn.Parameter(torch.zeros(input_dim, n_units))
        self.b_f = nn.Parameter(torch.zeros(input_dim, n_units))
        self.b_o = nn.Parameter(torch.zeros(input_dim, n_units))

        # Attention mechanisms
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.zeros(input_dim, 1))
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        h_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=device)
        c_tilda_t = torch.zeros(batch_size, self.input_dim, self.n_units, device=device)

        outputs = []
        for t in range(seq_len):
            xt_t = x[:, t, :].unsqueeze(1)

            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", xt_t, self.U_j) + self.b_j)
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", xt_t, self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", xt_t, self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", xt_t, self.U_o) + self.b_o)

            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            h_tilda_t = o_tilda_t * torch.tanh(c_tilda_t)

            outputs.append(h_tilda_t)

        outputs = torch.stack(outputs, dim=1)

        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)

        hg = torch.cat([g_n, h_tilda_t], dim=2)

        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas / torch.sum(betas, dim=1, keepdim=True)

        mean = torch.sum(betas * mu, dim=1)

        return mean


# --- 数据集创建与预处理 ---
class BatteryDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx],
                                                                                   dtype=torch.float32)


def create_sliding_windows(data, window_size, step):
    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


# --- 训练和评估函数 ---
def train_model(model, train_loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()
        predictions = model(features).squeeze(-1)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device, scaler_y):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features).squeeze(-1)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    all_preds_rescaled = scaler_y.inverse_transform(all_preds.reshape(-1, 1))
    all_targets_rescaled = scaler_y.inverse_transform(all_targets.reshape(-1, 1))

    mse = mean_squared_error(all_targets_rescaled, all_preds_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets_rescaled, all_preds_rescaled)
    r2 = r2_score(all_targets_rescaled, all_preds_rescaled)

    return mse, rmse, mae, r2, all_preds_rescaled, all_targets_rescaled

# --- 固定随机种子的函数 ---
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确保cuDNN的确定性，可能会牺牲一些性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 数据准备函数 ---
def prepare_data(full_data, feature_cols, target_col, window_size, step, train_ratio):
    cols_for_scaling = [target_col] + feature_cols
    data_to_process = full_data[cols_for_scaling]

    all_battery_ids = full_data['电池编号'].unique()
    all_battery_ids.sort()

    train_size = int(len(all_battery_ids) * train_ratio)
    train_battery_ids = all_battery_ids[:train_size]
    test_battery_ids = all_battery_ids[train_size:]

    print(f"\n总共 {len(all_battery_ids)} 个电池。")
    print(f"训练电池编号 ({len(train_battery_ids)}个): {train_battery_ids}")
    print(f"测试电池编号 ({len(test_battery_ids)}个): {test_battery_ids}")

    train_data_raw = data_to_process[full_data['电池编号'].isin(train_battery_ids)]
    test_data_raw = data_to_process[full_data['电池编号'].isin(test_battery_ids)]

    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_raw)
    test_data_scaled = scaler.transform(test_data_raw)

    scaler_y = MinMaxScaler()
    scaler_y.fit(train_data_raw[[target_col]])

    X_train, y_train = [], []
    train_data_scaled_df = pd.DataFrame(train_data_scaled, columns=cols_for_scaling, index=train_data_raw.index)
    train_data_scaled_df['电池编号'] = full_data.loc[train_data_raw.index, '电池编号']

    for battery_id in train_battery_ids:
        battery_data = train_data_scaled_df[train_data_scaled_df['电池编号'] == battery_id][cols_for_scaling].values
        X_b, y_b = create_sliding_windows(battery_data, window_size, step)
        if len(X_b) > 0:
            X_train.append(X_b)
            y_train.append(y_b)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_test, y_test = [], []
    test_data_scaled_df = pd.DataFrame(test_data_scaled, columns=cols_for_scaling, index=test_data_raw.index)
    test_data_scaled_df['电池编号'] = full_data.loc[test_data_raw.index, '电池编号']

    for battery_id in test_battery_ids:
        battery_data = test_data_scaled_df[test_data_scaled_df['电池编号'] == battery_id][cols_for_scaling].values
        X_b, y_b = create_sliding_windows(battery_data, window_size, step)
        if len(X_b) > 0:
            X_test.append(X_b)
            y_test.append(y_b)

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    X_train_features = X_train[:, :, 1:]
    X_test_features = X_test[:, :, 1:]

    train_dataset = BatteryDataset(X_train_features, y_train)
    test_dataset = BatteryDataset(X_test_features, y_test)

    return train_dataset, test_dataset, scaler_y, X_train_features.shape[2]


# --- 主程序 ---
if __name__ == '__main__':
    # --- 参数设置 ---
    # !!! 控制开关: 'train_and_test', 'train', 'test' !!!
    EXECUTION_MODE = 'train_and_test'
    SEED = 217 # 随机种子

    DATA_FILE = r'/home/scuee_user06/myh/电池/data/feature_results/all_batteries_data.csv'
    MODEL_SAVE_PATH = r'/home/scuee_user06/myh/电池/data/result/imv_lstm_model.pth'
    METRICS_SAVE_PATH = r'/home/scuee_user06/myh/电池/data/result/evaluation_metrics.csv'
    RESULT_DIR = r'/home/scuee_user06/myh/电池/data/result'

    WINDOW_SIZE = 10
    STEP = 1
    N_UNITS = 64
    LEARNING_RATE = 0.001
    EPOCHS = 200
    BATCH_SIZE = 32
    TRAIN_RATIO = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=============================================")
    print(f"执行模式: {EXECUTION_MODE}")
    print(f"使用设备: {device}")
    print(f"=============================================")

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"错误: 数据文件 '{DATA_FILE}' 未找到。")

    full_data = pd.read_csv(DATA_FILE, encoding='gbk')

    feature_cols = [
        'ICA峰值', 'ICA峰值位置(V)', '2.8~3.4V放电面积(Ah)',
        '恒流充电时间(s)', '恒压充电时间(s)', '恒流与恒压时间比值',
        '2.8~3.4V放电时间(s)', '3.3~3.6V充电时间(s)'
    ]
    target_col = '最大容量(Ah)'

    required_cols = ['电池编号', target_col] + feature_cols
    for col in required_cols:
        if col not in full_data.columns:
            raise ValueError(f"错误: 数据文件中缺少必需的列: '{col}'")

    # --- 数据准备 ---
    train_dataset, test_dataset, scaler_y, input_dim = prepare_data(
        full_data, feature_cols, target_col, WINDOW_SIZE, STEP, TRAIN_RATIO
    )

    # --- 初始化模型 ---
    output_dim = 1
    model = IMVTensorLSTM(input_dim=input_dim, output_dim=output_dim, n_units=N_UNITS).to(device)

    # --- 训练流程 ---
    if EXECUTION_MODE in ['train', 'train_and_test']:
        print("\n--- 开始训练流程 ---")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        for epoch in range(EPOCHS):
            train_loss = train_model(model, train_loader, optimizer, criterion, scheduler, device)
            print(f"Epoch {epoch + 1}/{EPOCHS}, 训练损失: {train_loss:.6f}, 学习率: {scheduler.get_last_lr()[0]:.6f}")

        print("\n训练完成!")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"模型已保存至: {MODEL_SAVE_PATH}")

    # --- 测试流程 ---
    if EXECUTION_MODE in ['test', 'train_and_test']:
        print("\n--- 开始测试流程 ---")

        if EXECUTION_MODE == 'test':
            if not os.path.exists(MODEL_SAVE_PATH):
                raise FileNotFoundError(
                    f"错误: 模型文件 '{MODEL_SAVE_PATH}' 未找到。请先运行 'train' 或 'train_and_test' 模式。")
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print(f"已从 '{MODEL_SAVE_PATH}' 加载模型。")

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print("\n开始评估...")
        mse, rmse, mae, r2, predictions, targets = evaluate_model(model, test_loader, device, scaler_y)

        print("\n--- 评估结果 ---")
        print(f"MSE (均方误差):      {mse:.6f}")
        print(f"RMSE (均方根误差):   {rmse:.6f}")
        print(f"MAE (平均绝对误差):  {mae:.6f}")
        print(f"R² (R-squared):      {r2:.6f}")
        print("--------------------")

        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R2'],
            'Value': [mse, rmse, mae, r2]
        })
        metrics_df.to_csv(METRICS_SAVE_PATH, index=False)
        print(f"评估指标已保存至: {METRICS_SAVE_PATH}")

        plt.figure(figsize=(14, 7))
        plt.plot(targets, label='Actual SOH', color='blue', marker='o', linestyle='-', markersize=4)
        plt.plot(predictions, label='Predicted SOH', color='red', marker='x', linestyle='--', markersize=4)
        plt.title('SOH predict result', fontsize=16)
        plt.xlabel('Test Sample Point', fontsize=12)
        plt.ylabel('max capacity (Ah)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULT_DIR, '电池SOH预测结果对比.png'), dpi=300)
        plt.show()

    # --- 流程结束检查 ---
    if EXECUTION_MODE == 'train':
        print("\n'train' 模式已完成。模型已保存，未执行评估。")
    elif EXECUTION_MODE not in ['train', 'test', 'train_and_test']:
        raise ValueError(f"无效的 EXECUTION_MODE: '{EXECUTION_MODE}'. 请选择 'train', 'test', 或 'train_and_test'.")

