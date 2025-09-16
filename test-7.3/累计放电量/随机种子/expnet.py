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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from Model import ExpNetTR
import shutil  # 导入 shutil 库用于文件操作
import torch.nn.functional as F

matplotlib.use('Agg')


# --- 1. 配置参数类 (新增) ---
class Config:
    def __init__(self):
        # --- 路径设置 ---
        self.data_dir = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/result-expnetTR-128/20'

        # --- 数据集划分 (核心修改点) ---
        # 在这里手动分配电池编号
        # self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 15, 16, 17, 18, 21, 22, 23, 24]
        # self.val_batteries = [5, 11, 13, 19]
        # self.test_batteries = [6, 12, 14, 20]  # 假设这些文件存在

        # self.train_batteries = [1, 2, 3, 4]
        # self.val_batteries = [5]
        # self.test_batteries = [6]

        # self.train_batteries = [7, 8, 9, 11]
        # self.val_batteries = [10]
        # self.test_batteries = [12]

        # self.train_batteries = [15, 16, 17, 18]
        # self.val_batteries = [13]
        # self.test_batteries = [14]
        #
        self.train_batteries = [21, 22, 23, 24]
        self.val_batteries = [19]
        self.test_batteries = [20]

        # --- 模型和训练超参数 ---
        self.n_terms = 512  # ExpNet的项数
        self.epochs = 20000
        self.learning_rate = 1e-3
        self.patience = 2000  # 用于早停
        self.nominal_capacity = 3.5  # 用于SOH归一化

        # --- 其他设置 ---
        self.seed = 2025  # 此种子将不再直接使用
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
            df = pd.read_csv(fpath)
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

    if removed_rows > 0:
        # 仅在实际移除数据时打印消息
        print(f"已为每个电池移除最后一个循环号的数据点，共移除 {removed_rows} 个数据点。")
    # ▲▲▲ 新增代码结束 ▲▲▲

    # 计算SOH
    full_df['soh'] = full_df['最大容量(Ah)'] / config.nominal_capacity

    # 根据配置的电池ID列表划分数据集
    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)]
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)]
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)]

    # print(f"数据划分完成:")
    # print(f"  - 训练集电池: {config.train_batteries} ({len(train_df)}个数据点)")
    # print(f"  - 验证集电池: {config.val_batteries} ({len(val_df)}个数据点)")
    # print(f"  - 测试集电池: {config.test_batteries} ({len(test_df)}个数据点)")

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
        plt.plot(subset['累计放电容量(Ah)'], true_capacity, 'o', color='blue', linestyle='-', label='True Capacity',
                 markersize=4, alpha=0.6)
        plt.plot(subset['累计放电容量(Ah)'], pred_capacity, '-', color='red', linestyle='--',
                 label='Predicted Capacity', linewidth=2, markersize=4)

        plt.title(f'Test Set: True vs. Predicted Capacity (Battery {battery_id})', fontsize=16)
        plt.xlabel('Accumulated discharge capacity (Ah)', fontsize=12)
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

            ax.plot(subset['累计放电容量(Ah)'], true_capacity, 'o', color='blue', label='True Capacity', alpha=0.6)
            ax.plot(subset['累计放电容量(Ah)'], pred_capacity, '-', color='red', label='Predicted Capacity',
                    linewidth=2)

            ax.set_title(f'Test Battery {battery_id}', fontsize=14)
            ax.set_xlabel('Accumulated discharge capacity (Ah)', fontsize=10)
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
        ax = plt.gca()  # 获取当前坐标轴

        # 绘制散点图
        ax.scatter(subset['soh'], subset['pred_soh'], alpha=0.7, label='Predictions')
        # 绘制理想的对角线 (y=x)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit')

        ax.set_xlabel('True SOH', fontsize=12)
        ax.set_ylabel('Predicted SOH', fontsize=12)
        ax.set_title(f'True vs. Predicted SOH (Battery {battery_id})', fontsize=16)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box')  # 使x和y轴比例相同
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
            ax.set_aspect('equal', adjustable='box')  # 使x和y轴比例相同
            ax.grid(True)

        # 隐藏多余的子图
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Test Set: True vs. Predicted SOH for Each Battery', fontsize=20)

    # 保存最终生成的图像
    plt.savefig(save_path, dpi=300)
    plt.close()


# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# --- 4. 主执行逻辑 (已完全重构) ---
def main():
    # 初始化配置
    config = Config()
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

    # 在循环外只加载一次数据
    print("\n正在加载和预处理数据...")
    try:
        train_df, val_df, test_df = load_and_split_data(config)
        print("数据加载完成。")
        print(f"  - 训练集: {len(train_df)}个数据点 | 验证集: {len(val_df)}个数据点 | 测试集: {len(test_df)}个数据点")
    except ValueError as e:
        print(f"数据加载失败: {e}")
        return

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

        # --- 2. 创建Tensors ---
        train_c = torch.tensor(train_df['累计放电容量(Ah)'].values, dtype=torch.float32, device=config.device)
        train_soh = torch.tensor(train_df['soh'].values, dtype=torch.float32, device=config.device)
        val_c = torch.tensor(val_df['累计放电容量(Ah)'].values, dtype=torch.float32, device=config.device)
        val_soh = torch.tensor(val_df['soh'].values, dtype=torch.float32, device=config.device)

        # --- 3. 初始化模型、损失函数和优化器 ---
        model = ExpNetTR(n_terms=config.n_terms).to(config.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        # --- 4. 训练模型 ---
        train_losses, val_losses = [], []
        best_val_loss_this_run = float('inf')
        epochs_no_improve = 0

        print("\n开始训练模型...")
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(train_c)
            loss = criterion(pred, train_soh)
            loss.backward()
            # 这会检查所有梯度，如果它们的总范数(大小)超过了 1.0，就按比例缩放回 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(val_c)
                val_loss = criterion(val_pred, val_soh).item()

            train_losses.append(loss.item())
            val_losses.append(val_loss)

            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1}/{config.epochs} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss_this_run:
                best_val_loss_this_run = val_loss
                torch.save(model.state_dict(), os.path.join(run_save_path, 'best_expnet_model.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    print(f"\n连续 {config.patience} 个 epoch 验证损失没有改善，提前停止训练。")
                    break

        print("训练完成。")

        # --- 5. 绘制Loss曲线并保存 ---
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title(f'Loss Curve (Run {run_number})')
        plt.savefig(os.path.join(run_save_path, 'loss_curve.png'), dpi=300)
        plt.close()

        # --- 6. 在测试集上评估最佳模型 (已重构，支持按电池分别计算指标) ---
        print("\n加载本轮最佳模型并在测试集上进行评估...")
        model.load_state_dict(torch.load(os.path.join(run_save_path, 'best_expnet_model.pth')))
        model.eval()

        test_c = torch.tensor(test_df['累计放电容量(Ah)'].values, dtype=torch.float32, device=config.device)
        with torch.no_grad():
            test_pred_soh = model(test_c).cpu().numpy()

        current_test_df = test_df.copy()
        current_test_df['pred_soh'] = test_pred_soh

        # ---【核心修改】开始：针对每个测试电池分别计算指标 ---
        print("\n--- 本轮测试集评估结果 (按单电池) ---")
        per_battery_metrics_list = []

        for batt_id in config.test_batteries:
            batt_df = current_test_df[current_test_df['battery_id'] == batt_id]
            if batt_df.empty:
                print(f"  - 电池 {batt_id}: 未找到数据，跳过。")
                continue

            batt_true = batt_df['soh'].values
            batt_pred = batt_df['pred_soh'].values

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
                f"  - 电池 {batt_id}: MAE={batt_metrics_dict['MAE']:.4f}, RMSE={batt_metrics_dict['RMSE']:.4f}, R2={batt_metrics_dict['R2']:.4f}")
            # ---【新增逻辑】开始：将本次的单独电池指标也添加到“跨实验汇总”的总列表中 ---
            batt_metrics_with_run_info = batt_metrics_dict.copy()
            batt_metrics_with_run_info['run'] = run_number
            batt_metrics_with_run_info['seed'] = current_seed
            all_runs_PER_BATTERY_metrics.append(batt_metrics_with_run_info)
            # ---【新增逻辑】结束 ---
            # ---【新增代码】开始：为单个电池绘制并保存两个图表 ---
            # (我们复用现有的网格图函数，但通过只传入单个电池ID，触发它们的“单图”模式)

            # 1. 绘制单独的曲线图
            plot_test_set_grid(df=current_test_df,
                               test_batteries=[batt_id],  # <-- 关键：只传入当前电池ID
                               nominal_capacity=config.nominal_capacity,
                               save_path=os.path.join(run_save_path, f'test_plot_battery_{batt_id}.png'),  # <-- 单独保存路径
                               ncols=1)

            # 2. 绘制单独的对角图
            plot_diagonal_grid(df=current_test_df,
                               test_batteries=[batt_id],  # <-- 关键：只传入当前电池ID
                               save_path=os.path.join(run_save_path, f'test_diagonal_plot_battery_{batt_id}.png'),
                               # <-- 单独保存路径
                               ncols=1)
            # ---【新增代码】结束 ---

        # 保存每个电池的指标汇总
        per_batt_df = pd.DataFrame(per_battery_metrics_list)
        per_batt_df.to_csv(os.path.join(run_save_path, 'test_per_battery_metrics.csv'), index=False)
        print(f"  -> 单独指标已保存至: {run_save_path}")
        # --- 【核心修改】结束 ---

        # --- 评估结果 (所有测试电池汇总) ---
        print("\n--- 本轮测试集评估结果 (所有测试电池汇总) ---")
        true_soh = current_test_df['soh']
        pred_soh = current_test_df['pred_soh']

        final_test_metrics = {
            'MAE': mean_absolute_error(true_soh, pred_soh),
            'MAPE': mean_absolute_percentage_error(true_soh, pred_soh),
            'MSE': mean_squared_error(true_soh, pred_soh),
            'RMSE': np.sqrt(mean_squared_error(true_soh, pred_soh)),
            'R2': r2_score(true_soh, pred_soh)
        }

        print(
            f"测试集(汇总): MSE={final_test_metrics['MSE']:.6f}, MAE={final_test_metrics['MAE']:.6f}, RMSE={final_test_metrics['RMSE']:.6f}, R2={final_test_metrics['R2']:.4f}")

        # 【新增】保存总体指标CSV
        pd.DataFrame([final_test_metrics]).to_csv(os.path.join(run_save_path, 'test_overall_metrics.csv'), index=False)

        # 记录本轮实验的总体指标 (用于5次run的对比)
        current_run_summary = {'run': run_number, 'seed': current_seed, **final_test_metrics}
        all_runs_metrics.append(current_run_summary)

        # --- 7. 保存本轮次的预测CSV和图表 (这部分已有逻辑不变) ---
        # 为每个测试电池保存预测结果CSV (已有逻辑)
        for battery_id in config.test_batteries:
            cell_df = current_test_df[current_test_df['battery_id'] == battery_id]
            if not cell_df.empty:
                cell_df.to_csv(os.path.join(run_save_path, f'test_battery_{battery_id}_predictions.csv'), index=False,
                               encoding='gbk')



        # --- 8. 检查是否为最佳轮次 (不变) ---
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
        print(summary_df.to_string())
        print(f"\n汇总指标已保存到: {summary_path}")

    # 2. 将最佳轮次的结果复制到主目录
    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (验证集损失最低: {best_run_val_loss:.6f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")

        for filename in os.listdir(best_run_dir):
            source_file = os.path.join(best_run_dir, filename)
            destination_file = os.path.join(config.save_path, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)

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