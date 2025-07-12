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
import json
import optuna

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
        X.append(data[i:i+window_size, 1:]) # 特征不包含第一列的目标
        y.append(data[i+window_size, 0])   # 目标是窗口末端下一个点的'最大容量(Ah)'
    return np.array(X), np.array(y)


# --- 训练和评估函数 (修改后用于Optuna) ---
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, device, trial, epochs):
    """训练并返回在验证集上的最佳RMSE"""
    best_val_rmse = float('inf')

    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(features).squeeze(-1)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 在验证集上评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features).squeeze(-1)
                val_loss += criterion(predictions, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        current_val_rmse = np.sqrt(avg_val_loss)

        if current_val_rmse < best_val_rmse:
            best_val_rmse = current_val_rmse

        # 向Optuna报告中间结果，用于剪枝
        trial.report(current_val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_rmse


# --- Optuna的目标函数 ---
def objective(trial, full_data, feature_cols, target_col, device):
    # 1. 定义超参数搜索空间
    params = {
        'window_size': trial.suggest_categorical('window_size', [10, 15, 20]),
        'n_units': trial.suggest_categorical('n_units', [32, 64, 128]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    }

    # 2. 准备数据 (每次试验都重新准备，因为window_size会变)
    # 数据集划分: 60% 训练, 20% 验证, 20% 测试
    all_battery_ids = sorted(full_data['电池编号'].unique())
    train_end_idx = int(len(all_battery_ids) * 0.6)
    val_end_idx = train_end_idx + int(len(all_battery_ids) * 0.2)

    train_ids = all_battery_ids[:train_end_idx]
    val_ids = all_battery_ids[train_end_idx:val_end_idx]

    cols_for_scaling = [target_col] + feature_cols
    data_to_process = full_data[cols_for_scaling]

    train_val_raw = data_to_process[full_data['电池编号'].isin(train_ids + val_ids)]
    scaler = MinMaxScaler()
    scaler.fit(train_val_raw)  # 在训练+验证集上fit scaler

    # 创建训练集
    train_data_raw = data_to_process[full_data['电池编号'].isin(train_ids)]
    train_data_scaled = scaler.transform(train_data_raw)
    X_train, y_train = [], []
    for battery_id in train_ids:
        battery_data_scaled = scaler.transform(data_to_process[full_data['电池编号'] == battery_id])
        X_b, y_b = create_sliding_windows(battery_data_scaled, params['window_size'], 1)
        if len(X_b) > 0:
            X_train.append(X_b);
            y_train.append(y_b)
    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    train_dataset = BatteryDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    # 创建验证集
    val_data_raw = data_to_process[full_data['电池编号'].isin(val_ids)]
    val_data_scaled = scaler.transform(val_data_raw)
    X_val, y_val = [], []
    for battery_id in val_ids:
        battery_data_scaled = scaler.transform(data_to_process[full_data['电池编号'] == battery_id])
        X_b, y_b = create_sliding_windows(battery_data_scaled, params['window_size'], 1)
        if len(X_b) > 0:
            X_val.append(X_b);
            y_val.append(y_b)
    X_val, y_val = np.concatenate(X_val), np.concatenate(y_val)
    val_dataset = BatteryDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # 3. 创建模型和优化器
    input_dim = len(feature_cols)
    model = IMVTensorLSTM(input_dim, 1, params['n_units']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # T_max可以设为固定值

    # 4. 训练和验证
    validation_rmse = train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, device,
                                         trial, epochs=50)

    return validation_rmse

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
    # --- 全局参数设置 ---
    # !!! 控制开关: 'optimize_and_test', 'optimize_only', 'test_only' !!!
    EXECUTION_MODE = 'optimize_and_test'

    SEED = 217
    N_TRIALS = 30   #optune寻优次数
    EPOCHS_FINAL = 200     #训练轮次
    PARAMS_SAVE_PATH = r'/home/scuee_user06/myh/电池/data/result/best_hyperparameters.json'
    DATA_FILE = r'/home/scuee_user06/myh/电池/data/feature_results/all_batteries_data.csv'
    MODEL_SAVE_PATH = r'/home/scuee_user06/myh/电池/data/result/imv_lstm_model.pth'
    METRICS_SAVE_PATH = r'/home/scuee_user06/myh/电池/data/result/evaluation_metrics.csv'
    RESULT_DIR = r'/home/scuee_user06/myh/电池/data/result'

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=============================================")
    print(f"执行模式: {EXECUTION_MODE}")
    print(f"使用设备: {device}, 随机种子: {SEED}")
    print(f"=============================================")

    full_data = pd.read_csv(DATA_FILE, encoding='gbk')
    feature_cols = ['ICA峰值', 'ICA峰值位置(V)', '2.8~3.4V放电面积(Ah)', '恒流充电时间(s)', '恒压充电时间(s)',
                    '恒流与恒压时间比值', '2.8~3.4V放电时间(s)', '3.3~3.6V充电时间(s)']
    target_col = '最大容量(Ah)'

    best_params = {}

    # --- 1. 寻优流程 ---
    if EXECUTION_MODE in ['optimize_only', 'optimize_and_test']:
        print("\n--- 开始Optuna超参数寻优 ---")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, full_data, feature_cols, target_col, device), n_trials=N_TRIALS)

        best_params = study.best_params
        print("\n寻优结束!")
        print(f"最佳试验的验证集RMSE: {study.best_value:.6f}")
        print("最佳超参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        with open(PARAMS_SAVE_PATH, 'w') as f:
            json.dump(best_params, f)
        print(f"最佳参数已保存至: {PARAMS_SAVE_PATH}")

        if EXECUTION_MODE == 'optimize_only':
            print("\n'optimize_only' 模式完成。")
            exit()

    # --- 2. 最终训练与测试流程 ---
    print("\n--- 开始最终训练与测试流程 ---")

    # 在 'test_only' 模式下加载已保存的参数
    if EXECUTION_MODE == 'test_only':
        if not os.path.exists(PARAMS_SAVE_PATH):
            raise FileNotFoundError(f"错误: 找不到参数文件 {PARAMS_SAVE_PATH}。请先运行 'optimize_and_test' 模式。")
        with open(PARAMS_SAVE_PATH, 'r') as f:
            best_params = json.load(f)
        print(f"已从 {PARAMS_SAVE_PATH} 加载最佳参数。")

    # 准备最终的数据集
    all_battery_ids = sorted(full_data['电池编号'].unique())
    train_val_end_idx = int(len(all_battery_ids) * 0.8)
    train_val_ids = all_battery_ids[:train_val_end_idx]
    test_ids = all_battery_ids[train_val_end_idx:]

    print(f"\n最终训练集 (train+val) 电池ID: {train_val_ids}")
    print(f"最终测试集 电池ID: {test_ids}")

    cols_for_scaling = [target_col] + feature_cols
    data_to_process = full_data[cols_for_scaling]
    train_val_raw = data_to_process[full_data['电池编号'].isin(train_val_ids)]
    scaler = MinMaxScaler().fit(train_val_raw)
    scaler_y = MinMaxScaler().fit(train_val_raw[[target_col]])

    # 创建最终模型
    final_model = IMVTensorLSTM(len(feature_cols), 1, best_params['n_units']).to(device)

    # 训练或加载最终模型
    if EXECUTION_MODE == 'optimize_and_test':
        print("\n--- 使用最佳参数训练最终模型 ---")
        X_train_final, y_train_final = [], []
        for battery_id in train_val_ids:
            battery_data_scaled = scaler.transform(data_to_process[full_data['电池编号'] == battery_id])
            X_b, y_b = create_sliding_windows(battery_data_scaled, best_params['window_size'], 1)
            if len(X_b) > 0:
                X_train_final.append(X_b);
                y_train_final.append(y_b)
        X_train_final, y_train_final = np.concatenate(X_train_final), np.concatenate(y_train_final)
        final_train_loader = DataLoader(BatteryDataset(X_train_final, y_train_final),
                                        batch_size=best_params['batch_size'], shuffle=True)

        optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINAL)

        for epoch in range(EPOCHS_FINAL):
            final_model.train()
            epoch_loss = 0
            for features, targets in final_train_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                predictions = final_model(features).squeeze(-1)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(
                    f"最终模型训练中... Epoch {epoch + 1}/{EPOCHS_FINAL}, Loss: {epoch_loss / len(final_train_loader):.6f}")

        torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
        print(f"\n最佳模型已保存至: {MODEL_SAVE_PATH}")

    elif EXECUTION_MODE == 'test_only':
        if not os.path.exists(MODEL_SAVE_PATH):
            raise FileNotFoundError(f"错误: 找不到模型文件 {MODEL_SAVE_PATH}。请先运行 'optimize_and_test' 模式。")
        final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"\n已从 {MODEL_SAVE_PATH} 加载最终模型。")

    # --- 3. 在测试集上评估最终模型 ---
    print("\n--- 在独立测试集上评估最终模型 ---")
    X_test_final, y_test_final = [], []
    for battery_id in test_ids:
        battery_data_scaled = scaler.transform(data_to_process[full_data['电池编号'] == battery_id])
        X_b, y_b = create_sliding_windows(battery_data_scaled, best_params['window_size'], 1)
        if len(X_b) > 0:
            X_test_final.append(X_b);
            y_test_final.append(y_b)
    X_test_final, y_test_final = np.concatenate(X_test_final), np.concatenate(y_test_final)
    final_test_loader = DataLoader(BatteryDataset(X_test_final, y_test_final), batch_size=best_params['batch_size'],
                                   shuffle=False)

    final_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in final_test_loader:
            features, targets = features.to(device), targets.to(device)
            all_preds.append(final_model(features).squeeze(-1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = scaler_y.inverse_transform(np.concatenate(all_preds).reshape(-1, 1))
    targets = scaler_y.inverse_transform(np.concatenate(all_targets).reshape(-1, 1))

    mse, rmse, mae, r2 = mean_squared_error(targets, predictions), np.sqrt(
        mean_squared_error(targets, predictions)), mean_absolute_error(targets, predictions), r2_score(targets,
                                                                                                       predictions)

    print("\n--- 最终评估结果 ---")
    print(f"MSE (均方误差):      {mse:.6f}")
    print(f"RMSE (均方根误差):   {rmse:.6f}")
    print(f"MAE (平均绝对误差):  {mae:.6f}")
    print(f"R² (R-squared):      {r2:.6f}")

    pd.DataFrame({'Metric': ['MSE', 'RMSE', 'MAE', 'R2'], 'Value': [mse, rmse, mae, r2]}).to_csv(METRICS_SAVE_PATH,
                                                                                                 index=False)
    print(f"最终评估指标已保存至: {METRICS_SAVE_PATH}")

    plt.figure(figsize=(14, 7))
    plt.plot(targets, label='Actual SOH', color='blue', marker='o', linestyle='-', markersize=4)
    plt.plot(predictions, label='Predicted SOH', color='red', marker='x', linestyle='--', markersize=4)
    plt.title('predict result in test dataset', fontsize=16)
    plt.xlabel('Test Sample Point', fontsize=12)
    plt.ylabel('max capacity (Ah)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, '测试集结果'))
    plt.show()

