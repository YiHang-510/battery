from Model import ExpNet
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import os
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Model import IMVTensorLSTM  # 你的IMV-LSTM模型类
from torch.utils.data import TensorDataset, DataLoader

save_path = r'/home/scuee_user06/myh/电池/rusult'
data_dir = r'/home/scuee_user06/myh/电池/data/cycle'
result_save_dir = f"{save_path}/expnet_cell_result"
os.makedirs(result_save_dir, exist_ok=True)
matplotlib.use('Agg')

def set_seed(seed=217):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(217)



# 1. 合并所有文件
file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
all_cycles = []
all_capacity = []

for fname in file_list:
    fpath = os.path.join(data_dir, fname)
    df = pd.read_csv(fpath, encoding='gbk')
    cycles = df['循环号'].to_numpy()
    capacity = df['放电容量(Ah)'].to_numpy()
    all_cycles.append(cycles)
    all_capacity.append(capacity)
cycles = np.concatenate(all_cycles)
capacity = np.concatenate(all_capacity)
soh = capacity / 3.5  # 标称容量归一化

# 2. shuffle整体数据（保证训练和验证均有来自不同文件的数据）
N = len(cycles)
indices = np.arange(N)
np.random.shuffle(indices)
cycles = cycles[indices]
soh = soh[indices]

split = int(0.7 * N)
train_c, val_c = cycles[:split], cycles[split:]
train_soh, val_soh = soh[:split], soh[split:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_c_tensor = torch.tensor(train_c, dtype=torch.float32, device=device)
train_soh_tensor = torch.tensor(train_soh, dtype=torch.float32, device=device)
val_c_tensor = torch.tensor(val_c, dtype=torch.float32, device=device)
val_soh_tensor = torch.tensor(val_soh, dtype=torch.float32, device=device)

# 3. 初始化
epochs = 4000
learning_rate = 1e-2

model = ExpNet(n_terms=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs + 100)

# 4. 训练
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(train_c_tensor)
    loss = criterion(pred, train_soh_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()
    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(val_c_tensor)
        val_loss = criterion(val_pred, val_soh_tensor).item()
    train_losses.append(loss.item())
    val_losses.append(val_loss)
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6e}")

# 5. 可视化 Loss 曲线
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig(f'{save_path}/exp-loss.png', dpi=200, bbox_inches='tight')
plt.close()

# 6. 可视化 SOH 拟合结果
model.eval()
with torch.no_grad():
    all_c = torch.tensor(cycles, dtype=torch.float32, device=device)
    all_pred = model(all_c).cpu().numpy()

file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
file_num = len(all_cycles)
fig_cols = 3
fig_rows = math.ceil(file_num / fig_cols)
plt.figure(figsize=(fig_cols * 5, fig_rows * 4))

val_indices = indices[split:]  # 验证集在打乱前的原始索引

metrics_list = []

start = 0
for i, (file_cycles, file_capacity) in enumerate(zip(all_cycles, all_capacity)):
    end = start + len(file_cycles)
    file_idx_range = np.arange(start, end)
    val_mask = np.isin(file_idx_range, val_indices)
    val_cycles = file_cycles[val_mask]
    val_capacity = file_capacity[val_mask]
    val_soh = val_capacity / 3.5

    if len(val_cycles) == 0:
        start = end
        continue

    val_c_tensor = torch.tensor(val_cycles, dtype=torch.float32, device=device)
    with torch.no_grad():
        val_pred = model(val_c_tensor).cpu().numpy()

    # 保存当前电池SOH预测结果
    df = pd.DataFrame({
        "cycle": val_cycles,
        "true_soh": val_soh,
        "pred_soh": val_pred.flatten()
    })
    out_name = os.path.splitext(file_list[i])[0] + "_soh_fit.csv"
    df.to_csv(os.path.join(result_save_dir, out_name), index=False, encoding='utf-8-sig')

    # 计算指标
    mae = mean_absolute_error(val_soh, val_pred)
    mse = mean_squared_error(val_soh, val_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(val_soh, val_pred)

    # 保存单个电池的指标
    metrics_list.append({
        "cell_name": os.path.splitext(file_list[i])[0],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    ax = plt.subplot(fig_rows, fig_cols, i + 1)
    ax.plot(val_cycles, val_soh, 'o', label='True SOH', alpha=0.7)
    ax.plot(val_cycles, val_pred, 'r.', label='Predicted SOH', alpha=0.7)
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('SOH')
    ax.set_title(file_list[i])  # 显示文件名
    ax.legend()
    start = end

# 转为DataFrame
metrics_df = pd.DataFrame(metrics_list)

# 计算平均
mean_row = metrics_df.mean(numeric_only=True)
mean_row['cell_name'] = "Mean"
metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

# 保存
metrics_df.to_csv(f"{save_path}/soh_fit_metrics.csv", index=False, encoding='utf-8-sig')

plt.tight_layout()
plt.savefig(f'{save_path}/exp-SOH-val-eachfile.png', dpi=200, bbox_inches='tight')
plt.close()
print('验证集各文件SOH拟合输出完成')
print('经验指数网络输出完成')



