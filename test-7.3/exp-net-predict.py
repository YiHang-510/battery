import os
import random
import re  # 引入正则表达式库，用于从文件名提取电池编号
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Model import ExpNet  # 假设ExpNet在Model.py中

matplotlib.use('Agg')


# --- 1. 配置参数类 (新增) ---
class Config:
    def __init__(self):
        # --- 路径设置 ---
        self.data_dir = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = r'/home/scuee_user06/myh/电池/data/expnet_result'

        # --- 数据集划分 (核心修改点) ---
        # 在这里手动分配电池编号
        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22]
        # self.val_batteries = [5, 11, 17, 23]
        # self.test_batteries = [6, 12, 18, 24]  # 假设这些文件存在

        # self.train_batteries = [1, 2, 3, 4]
        # self.val_batteries = [5]
        # self.test_batteries = [6]  # 假设这些文件存在

        # self.train_batteries = [7, 8, 9, 10]
        # self.val_batteries = [11]
        # self.test_batteries = [12]  # 假设这些文件存在

        # self.train_batteries = [13, 14, 15, 16]
        # self.val_batteries = [17]
        # self.test_batteries = [18]  # 假设这些文件存在

        self.train_batteries = [19, 20, 21, 22]
        self.val_batteries = [23]
        self.test_batteries = [24]  # 假设这些文件存在

        # --- 模型和训练超参数 ---
        self.n_terms = 4  # ExpNet的项数
        self.epochs = 10000
        self.learning_rate = 1e-2
        self.patience = 2000  # 用于早停
        self.nominal_capacity = 1  # 用于SOH归一化

        # --- 其他设置 ---
        self.seed = 2025
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# --- 2. 数据加载和划分函数 (重写) ---
def load_and_split_data(config):
    """
    按电池ID加载、合并和划分数据。
    """
    file_list = sorted([f for f in os.listdir(config.data_dir) if f.endswith('.csv')])

    all_df_list = []
    for fname in file_list:
        try:
            # 从文件名中提取电池编号 (例如: 'battery_1.csv' -> 1)
            match = re.search(r'\d+', fname)
            if not match:
                continue
            battery_id = int(match.group())

            fpath = os.path.join(config.data_dir, fname)
            df = pd.read_csv(fpath, encoding='gbk')
            df['battery_id'] = battery_id  # 添加电池ID列
            all_df_list.append(df)
        except Exception as e:
            print(f"读取或处理文件 {fname} 时出错: {e}")

    if not all_df_list:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_df_list, ignore_index=True)

    # ▼▼▼ 新增代码开始 ▼▼▼
    # 目标：为每个电池移除最后一个循环号的数据点。
    # 1. 使用 groupby 和 transform 为每一行数据找到其所属电池的最大循环号。
    #    这会生成一个与 full_df 等长的 Series，每一行的值都是该行所属电池的最大循环号。
    max_cycles_per_battery = full_df.groupby('battery_id')['循环号'].transform('max')

    # 2. 保留那些“循环号”小于其对应电池“最大循环号”的行。
    original_rows = len(full_df)
    full_df = full_df[full_df['循环号'] < max_cycles_per_battery].copy()  # 使用 .copy() 避免 SettingWithCopyWarning
    removed_rows = original_rows - len(full_df)

    print(f"\n已为每个电池移除最后一个循环号的数据点，共移除 {removed_rows} 个数据点。")
    # ▲▲▲ 新增代码结束 ▲▲▲

    # 计算SOH
    full_df['soh'] = full_df['最大容量(Ah)'] / config.nominal_capacity

    # 根据配置的电池ID列表划分数据集
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)]
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)]
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)]

    print(f"数据划分完成:")
    print(f"  - 训练集电池: {config.train_batteries} ({len(train_df)}个数据点)")
    print(f"  - 验证集电池: {config.val_batteries} ({len(val_df)}个数据点)")
    print(f"  - 测试集电池: {config.test_batteries} ({len(test_df)}个数据点)")

    return train_df, val_df, test_df


# --- 全新的绘图函数，用于生成测试集子图网格 ---
def plot_test_set_grid(df, test_batteries, nominal_capacity, save_path, ncols=2):
    """
    为测试集中的电池绘制结果图。
    - 如果只有一个电池，则绘制单张大图。
    - 如果有多个电池，则将它们排列在一个网格中。
    """
    num_plots = len(test_batteries)
    if num_plots == 0:
        print("测试集中没有电池，无法绘图。")
        return

    # =================== 核心修改：根据绘图数量选择不同逻辑 ===================

    if num_plots == 1:
        # --- 情况一：只有一个测试电池，绘制单张图 ---
        battery_id = test_batteries[0]
        subset = df[df['battery_id'] == battery_id].sort_values(by='循环号')

        if subset.empty:
            print(f"电池 {battery_id} 没有数据。")
            return

        plt.figure(figsize=(10, 6))  # 使用适合单张图的尺寸

        # 反归一化
        true_capacity = subset['soh'] * nominal_capacity
        pred_capacity = subset['pred_soh'] * nominal_capacity

        # 绘图
        plt.plot(subset['循环号'], true_capacity, 'o', color='blue', linestyle='-', label='True Capacity', markersize=4, alpha=0.6)
        plt.plot(subset['循环号'], pred_capacity, '-', color='red', linestyle='--', label='Predicted Capacity', linewidth=2, markersize=4)

        plt.title(f'Test Set: True vs. Predicted Capacity (Battery {battery_id})', fontsize=16)
        plt.xlabel('Cycle', fontsize=12)
        plt.ylabel('Capacity (Ah)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.ylim(2.2, 3.5)
        plt.tight_layout()  # 自动调整布局

    else:
        # --- 情况二：有多个测试电池，绘制网格图 (保留原逻辑) ---
        nrows = math.ceil(num_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows), constrained_layout=True)
        axes = axes.flatten()

        for i, battery_id in enumerate(test_batteries):
            ax = axes[i]
            subset = df[df['battery_id'] == battery_id].sort_values(by='循环号')

            if subset.empty:
                ax.set_title(f'Test Battery {battery_id}\n(No Data Found)')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                continue

            true_capacity = subset['soh'] * nominal_capacity
            pred_capacity = subset['pred_soh'] * nominal_capacity

            ax.plot(subset['循环号'], true_capacity, 'o', color='blue', label='True Capacity', alpha=0.6)
            ax.plot(subset['循环号'], pred_capacity, '-', color='red', label='Predicted Capacity', linewidth=2)

            ax.set_title(f'Test Battery {battery_id}', fontsize=14)
            ax.set_xlabel('Cycle', fontsize=10)
            ax.set_ylabel('Capacity (Ah)', fontsize=10)
            ax.legend()
            ax.grid(True)
            ax.set_ylim(2.2, 3.5)

        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Test Set: True vs. Predicted Capacity for Each Battery', fontsize=20)

    # 保存最终生成的图像
    plt.savefig(save_path, dpi=1200)
    plt.close()

# ▼▼▼【新功能】新增的可视化函数 ▼▼▼
def plot_diagonal_grid(df, test_batteries, save_path, ncols=2):
    """
    为测试集中的电池绘制真实值与预测值的对角图。
    - 如果只有一个电池，则绘制单张大图。
    - 如果有多个电池，则将它们排列在一个网格中。
    """
    num_plots = len(test_batteries)
    if num_plots == 0:
        print("测试集中没有电池，无法绘制对角图。")
        return

    # 确定SOH的绘图范围，以便所有子图比例一致
    min_val = min(df['soh'].min(), df['pred_soh'].min()) * 0.98
    max_val = max(df['soh'].max(), df['pred_soh'].max()) * 1.02

    if num_plots == 1:
        # --- 情况一：只有一个测试电池，绘制单张图 ---
        battery_id = test_batteries[0]
        subset = df[df['battery_id'] == battery_id]
        if subset.empty:
            print(f"电池 {battery_id} 没有数据。")
            return

        plt.figure(figsize=(8, 8))
        ax = plt.gca() # 获取当前坐标轴

        # 绘制散点图
        ax.scatter(subset['soh'], subset['pred_soh'], alpha=0.7, label='Predictions')
        # 绘制理想的对角线 (y=x)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit')

        ax.set_xlabel('True SOH', fontsize=12)
        ax.set_ylabel('Predicted SOH', fontsize=12)
        ax.set_title(f'True vs. Predicted SOH (Battery {battery_id})', fontsize=16)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box') # 使x和y轴比例相同
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

    else:
        # --- 情况二：有多个测试电池，绘制网格图 ---
        nrows = math.ceil(num_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows), constrained_layout=True)
        axes = axes.flatten()

        for i, battery_id in enumerate(test_batteries):
            ax = axes[i]
            subset = df[df['battery_id'] == battery_id]

            if subset.empty:
                ax.set_title(f'Test Battery {battery_id}\n(No Data Found)')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                continue

            # 绘制散点图
            ax.scatter(subset['soh'], subset['pred_soh'], alpha=0.7)
            # 绘制理想的对角线 (y=x)
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            ax.set_xlabel('True SOH', fontsize=10)
            ax.set_ylabel('Predicted SOH', fontsize=10)
            ax.set_title(f'Test Battery {battery_id}', fontsize=14)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal', adjustable='box') # 使x和y轴比例相同
            ax.grid(True)

        # 隐藏多余的子图
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Test Set: True vs. Predicted SOH for Each Battery', fontsize=20)

    # 保存最终生成的图像
    plt.savefig(save_path, dpi=300)
    plt.close()
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# --- 4. 主执行逻辑 ---
def main():
    # 初始化配置
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    set_seed(config.seed)

    # 加载并划分数据
    train_df, val_df, test_df = load_and_split_data(config)

    # 创建Tensors
    train_c = torch.tensor(train_df['循环号'].values, dtype=torch.float32, device=config.device)
    train_soh = torch.tensor(train_df['soh'].values, dtype=torch.float32, device=config.device)
    val_c = torch.tensor(val_df['循环号'].values, dtype=torch.float32, device=config.device)
    val_soh = torch.tensor(val_df['soh'].values, dtype=torch.float32, device=config.device)

    # 初始化模型、损失函数和优化器
    model = ExpNet(n_terms=config.n_terms).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 训练模型
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("\n开始训练模型...")
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(train_c)
        loss = criterion(pred, train_soh)

        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_c)
            val_loss = criterion(val_pred, val_soh).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")

        # 保存最佳模型并实现早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.save_path, 'best_expnet_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                break

    print("训练完成。")

    # 绘制Loss曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(config.save_path, 'loss_curve.png'), dpi=1200)
    plt.close()

    # --- 在测试集上评估最佳模型 (新增) ---
    print("\n加载最佳模型并在测试集上进行评估...")
    model.load_state_dict(torch.load(os.path.join(config.save_path, 'best_expnet_model.pth')))
    model.eval()

    test_c = torch.tensor(test_df['循环号'].values, dtype=torch.float32, device=config.device)

    with torch.no_grad():
        test_pred_soh = model(test_c).cpu().numpy()

    test_df['pred_soh'] = test_pred_soh

    # ▼▼▼【修改与新增部分】▼▼▼

    # --- 计算测试集总体指标 ---
    true_soh = test_df['soh']
    pred_soh = test_df['pred_soh']

    mae = mean_absolute_error(true_soh, pred_soh)
    mse = mean_squared_error(true_soh, pred_soh)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_soh, pred_soh)

    print("\n--- 测试集评估结果 ---")
    print(f"  - MAE:  {mae:.4f}")
    print(f"  - MSE:  {mse:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R2:   {r2:.4f}")

    # ---【新功能】将指标保存到CSV文件 ---
    metrics_data = {
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'R2': [r2]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_save_path = os.path.join(config.save_path, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_save_path, index=False, encoding='gbk')

    print(f"\n评估指标已成功保存到: {metrics_save_path}")

    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # 为每个测试电池保存预测结果
    for battery_id in config.test_batteries:
        cell_df = test_df[test_df['battery_id'] == battery_id]
        if not cell_df.empty:
            cell_df.to_csv(os.path.join(config.save_path, f'test_battery_{battery_id}_predictions.csv'), index=False, encoding='gbk')

    print("\n正在生成测试集结果网格图...")
    plot_test_set_grid(
        df=test_df,
        test_batteries=config.test_batteries,
        nominal_capacity=config.nominal_capacity,
        save_path=os.path.join(config.save_path, 'test_set_grid_plot.png'),
        ncols=2
    )

    # ▼▼▼【新功能】调用新的对角图函数 ▼▼▼
    print("\n正在生成真实值与预测值的对角网格图...")
    plot_diagonal_grid(
        df=test_df,
        test_batteries=config.test_batteries,
        save_path=os.path.join(config.save_path, 'test_set_diagonal_plot.png'),
        ncols=2  # 在这里设置网格的列数
    )
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    print(f"\n评估完成。所有结果已保存到: {config.save_path}")


if __name__ == '__main__':
    main()