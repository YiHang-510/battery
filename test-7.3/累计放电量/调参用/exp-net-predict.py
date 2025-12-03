import os
import random
import re
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from Model import ExpNet  # 假设ExpNet在Model.py中
import itertools  # <-- 【新增】用于生成参数组合
import shutil  # <-- 【新增】用于清理旧目录

matplotlib.use('Agg')


# --- 1. 配置参数类 (基本不变) ---
class Config:
    def __init__(self):
        # --- 路径设置 ---
        self.data_dir = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'
        self.save_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/result-expnet/all_grid_search'  # <-- 修改了保存路径

        # --- 数据集划分 (不变) ---
        self.train_batteries = [1, 2, 3, 4, 7, 8, 9, 10, 15, 16, 17, 18, 21, 22, 23, 24]
        self.val_batteries = [5, 11, 13, 19]
        self.test_batteries = [6, 12, 14, 20]

        # --- 模型和训练超参数 (这些将成为默认值，会被搜索空间覆盖) ---
        self.n_terms = 4
        self.epochs = 20000
        self.learning_rate = 5e-3
        self.patience = 2000
        self.nominal_capacity = 3.5

        # --- 其他设置 (不变) ---
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


# --- 2. 数据加载和划分函数 (不变) ---
def load_and_split_data(config):
    """
    按电池ID加载、合并和划分数据。
    """
    file_list = sorted([f for f in os.listdir(config.data_dir) if f.endswith('.csv')])

    all_df_list = []
    for fname in file_list:
        try:
            match = re.search(r'\d+', fname)
            if not match:
                continue
            battery_id = int(match.group())

            fpath = os.path.join(config.data_dir, fname)
            df = pd.read_csv(fpath)
            df['battery_id'] = battery_id
            all_df_list.append(df)
        except Exception as e:
            print(f"读取或处理文件 {fname} 时出错: {e}")

    if not all_df_list:
        raise ValueError("未能成功加载任何电池数据。")

    full_df = pd.concat(all_df_list, ignore_index=True)

    max_cycles_per_battery = full_df.groupby('battery_id')['循环号'].transform('max')
    original_rows = len(full_df)
    full_df = full_df[full_df['循环号'] < max_cycles_per_battery].copy()
    removed_rows = original_rows - len(full_df)

    if removed_rows > 0:
        print(f"\n已为每个电池移除最后一个循环号的数据点，共移除 {removed_rows} 个数据点。")

    full_df['soh'] = full_df['最大容量(Ah)'] / config.nominal_capacity

    train_df = full_df[full_df['battery_id'].isin(config.train_batteries)]
    val_df = full_df[full_df['battery_id'].isin(config.val_batteries)]
    test_df = full_df[full_df['battery_id'].isin(config.test_batteries)]

    print(f"数据划分完成:")
    print(f"  - 训练集电池: {config.train_batteries} ({len(train_df)}个数据点)")
    print(f"  - 验证集电池: {config.val_batteries} ({len(val_df)}个数据点)")
    print(f"  - 测试集电池: {config.test_batteries} ({len(test_df)}个数据点)")

    return train_df, val_df, test_df


# --- 3. 绘图函数 (不变) ---
def plot_test_set_grid(df, test_batteries, nominal_capacity, save_path, ncols=2):
    # (此函数内容与原脚本相同，此处省略以节约篇幅)
    # ... (确保您已复制了原脚本中的 plot_test_set_grid 全部内容) ...
    num_plots = len(test_batteries)
    if num_plots == 0:
        return
    if num_plots == 1:
        battery_id = test_batteries[0]
        subset = df[df['battery_id'] == battery_id].sort_values(by='循环号')
        if subset.empty: return
        plt.figure(figsize=(10, 6))
        true_capacity = subset['soh'] * nominal_capacity
        pred_capacity = subset['pred_soh'] * nominal_capacity
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
        plt.tight_layout()
    else:
        nrows = math.ceil(num_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows), constrained_layout=True)
        axes = axes.flatten()
        for i, battery_id in enumerate(test_batteries):
            ax = axes[i]
            subset = df[df['battery_id'] == battery_id].sort_values(by='循环号')
            if subset.empty:
                ax.set_title(f'Test Battery {battery_id}\n(No Data Found)')
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
    plt.savefig(save_path, dpi=1200)
    plt.close()


def plot_diagonal_grid(df, test_batteries, save_path, ncols=2):
    # (此函数内容与原脚本相同，此处省略以节约篇幅)
    # ... (确保您已复制了原脚本中的 plot_diagonal_grid 全部内容) ...
    num_plots = len(test_batteries)
    if num_plots == 0:
        return
    min_val = min(df['soh'].min(), df['pred_soh'].min()) * 0.98
    max_val = max(df['soh'].max(), df['pred_soh'].max()) * 1.02
    if num_plots == 1:
        battery_id = test_batteries[0]
        subset = df[df['battery_id'] == battery_id]
        if subset.empty: return
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.scatter(subset['soh'], subset['pred_soh'], alpha=0.7, label='Predictions')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal Fit')
        ax.set_xlabel('True SOH', fontsize=12)
        ax.set_ylabel('Predicted SOH', fontsize=12)
        ax.set_title(f'True vs. Predicted SOH (Battery {battery_id})', fontsize=16)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
    else:
        nrows = math.ceil(num_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows), constrained_layout=True)
        axes = axes.flatten()
        for i, battery_id in enumerate(test_batteries):
            ax = axes[i]
            subset = df[df['battery_id'] == battery_id]
            if subset.empty:
                ax.set_title(f'Test Battery {battery_id}\n(No Data Found)')
                continue
            ax.scatter(subset['soh'], subset['pred_soh'], alpha=0.7)
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax.set_xlabel('True SOH', fontsize=10)
            ax.set_ylabel('Predicted SOH', fontsize=10)
            ax.set_title(f'Test Battery {battery_id}', fontsize=14)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True)
        for j in range(num_plots, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('Test Set: True vs. Predicted SOH for Each Battery', fontsize=20)
    plt.savefig(save_path, dpi=300)
    plt.close()


# --- 4. 【新增】重构的训练与评估函数 ---
def train_and_evaluate(config, data_tensors, trial_save_path):
    """
    使用给定配置运行一次完整的训练和评估。
    """

    # 从 data_tensors 解包数据
    train_c = data_tensors['train_c']
    train_soh = data_tensors['train_soh']
    val_c = data_tensors['val_c']
    val_soh = data_tensors['val_soh']
    test_df = data_tensors['test_df']
    test_c = data_tensors['test_c']

    # 初始化模型、损失函数和优化器
    # 确保每次试验都使用新种子，但种子本身是固定的，以保证试验可复现
    set_seed(config.seed)
    model = ExpNet(n_terms=config.n_terms).to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # 训练模型
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"开始训练... (nt={config.n_terms}, lr={config.learning_rate}, p={config.patience})")
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

        # 保存最佳模型并实现早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 【修改】保存到试验专用的子目录
            torch.save(model.state_dict(), os.path.join(trial_save_path, 'best_expnet_model.pth'))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"  - 提前停止于 Epoch {epoch + 1}")
                break

    # 绘制Loss曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title(f'Loss Curve (nt={config.n_terms}, lr={config.learning_rate})')
    plt.savefig(os.path.join(trial_save_path, 'loss_curve.png'), dpi=300)
    plt.close()

    # --- 在测试集上评估最佳模型 ---
    print("  - 加载最佳模型并评估测试集...")
    model.load_state_dict(torch.load(os.path.join(trial_save_path, 'best_expnet_model.pth')))
    model.eval()

    with torch.no_grad():
        test_pred_soh = model(test_c).cpu().numpy()

    # 必须使用 .copy() 以免影响其他试验中的 test_df
    current_test_df = test_df.copy()
    current_test_df['pred_soh'] = test_pred_soh

    # --- 计算测试集总体指标 ---
    true_soh = current_test_df['soh']
    pred_soh = current_test_df['pred_soh']

    mae = mean_absolute_error(true_soh, pred_soh)
    mape = mean_absolute_percentage_error(true_soh, pred_soh)
    mse = mean_squared_error(true_soh, pred_soh)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_soh, pred_soh)

    print(f"  - 测试结果: MAE={mae:.4f}, R2={r2:.4f}")

    # --- 保存指标和绘图 ---
    metrics_data = {'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2]}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_save_path = os.path.join(trial_save_path, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_save_path, index=False, encoding='gbk')

    for battery_id in config.test_batteries:
        cell_df = current_test_df[current_test_df['battery_id'] == battery_id]
        if not cell_df.empty:
            cell_df.to_csv(os.path.join(trial_save_path, f'test_battery_{battery_id}_predictions.csv'), index=False,
                           encoding='gbk')

    plot_test_set_grid(
        df=current_test_df,
        test_batteries=config.test_batteries,
        nominal_capacity=config.nominal_capacity,
        save_path=os.path.join(trial_save_path, 'test_set_grid_plot.png'),
        ncols=2
    )

    plot_diagonal_grid(
        df=current_test_df,
        test_batteries=config.test_batteries,
        save_path=os.path.join(trial_save_path, 'test_set_diagonal_plot.png'),
        ncols=2
    )

    # --- 【新增】返回结果字典，用于最终汇总 ---
    result_summary = {
        'n_terms': config.n_terms,
        'learning_rate': config.learning_rate,
        'patience': config.patience,
        'best_val_loss': best_val_loss,
        'test_mae': mae,
        'test_mape': mape,
        'test_rmse': rmse,
        'test_r2': r2,
        'trial_save_path': trial_save_path
    }
    return result_summary


# --- 5. 【重构】主执行逻辑 (Grid Search) ---
def main():
    # --- 1. 定义超参数搜索空间 ---
    SEARCH_SPACE = {
        'n_terms': [4, 8, 16, 32, 64, 128, 256, 512],
        'learning_rate': [0.01, 0.005, 0.001, 0.0005],
        'patience': [1000, 2000, 3000]
    }

    # 初始化默认配置
    base_config = Config()
    os.makedirs(base_config.save_path, exist_ok=True)

    # 清理之前可能存在的旧搜索结果
    # if os.path.exists(base_config.save_path):
    #     print(f"警告：正在清理旧的搜索目录: {base_config.save_path}")
    #     shutil.rmtree(base_config.save_path)
    #     os.makedirs(base_config.save_path)

    print("--- 超参数网格搜索开始 ---")
    print(f"搜索空间: {SEARCH_SPACE}")

    # --- 2. 加载一次数据 ---
    print("\n正在加载和预处理数据 (仅一次)...")
    train_df, val_df, test_df = load_and_split_data(base_config)

    # 创建一次Tensors
    data_tensors = {
        'train_c': torch.tensor(train_df['累计放电容量(Ah)'].values, dtype=torch.float32, device=base_config.device),
        'train_soh': torch.tensor(train_df['soh'].values, dtype=torch.float32, device=base_config.device),
        'val_c': torch.tensor(val_df['累计放电容量(Ah)'].values, dtype=torch.float32, device=base_config.device),
        'val_soh': torch.tensor(val_df['soh'].values, dtype=torch.float32, device=base_config.device),
        'test_df': test_df,
        'test_c': torch.tensor(test_df['累计放电容量(Ah)'].values, dtype=torch.float32, device=base_config.device)
    }
    print("数据和Tensors准备完毕。")

    # --- 3. 生成参数组合并开始循环 ---
    all_results = []

    # 从字典中提取键和值列表
    param_keys = SEARCH_SPACE.keys()
    param_values = SEARCH_SPACE.values()

    # 使用 itertools.product 生成所有组合的笛卡尔积
    all_combinations = list(itertools.product(*param_values))
    total_trials = len(all_combinations)
    print(f"\n总共需要运行 {total_trials} 次试验。\n")

    for i, combo in enumerate(all_combinations):

        # 创建本次试验的配置
        trial_config = Config()

        # 将当前组合的参数值设置到 trial_config 对象中
        param_dict = {}
        for key, value in zip(param_keys, combo):
            setattr(trial_config, key, value)
            param_dict[key] = value

        print(f"--- [试验 {i + 1}/{total_trials}] 开始: {param_dict} ---")

        # 创建本次试验的唯一保存目录
        trial_name = "_".join([f"{key[:2]}{val}" for key, val in param_dict.items()])  # e.g., "nt4_le0.005_pa2000"
        trial_save_path = os.path.join(base_config.save_path, f"trial_{trial_name}")
        os.makedirs(trial_save_path, exist_ok=True)

        # 运行训练和评估
        try:
            result_summary = train_and_evaluate(trial_config, data_tensors, trial_save_path)
            all_results.append(result_summary)
            print(f"--- [试验 {i + 1}/{total_trials}] 完成 ---")
        except Exception as e:
            print(f"!!!!!! [试验 {i + 1}/{total_trials}] 失败: {e} !!!!!!")
            # 记录失败
            all_results.append({
                **param_dict,
                'best_val_loss': 'FAILED',
                'test_mae': 'FAILED',
                'test_r2': 'FAILED'
            })

    # --- 4. 汇总所有结果 ---
    print("\n\n" + "=" * 50)
    print("所有试验均已完成。正在汇总结果...")
    print("=" * 50)

    results_df = pd.DataFrame(all_results)

    # 按测试集 MAE 排序 (确保 FAILED 的值不会被当作最小值)
    results_df['test_mae_numeric'] = pd.to_numeric(results_df['test_mae'], errors='coerce')
    results_df = results_df.sort_values(by='test_mae_numeric', ascending=True)
    results_df = results_df.drop(columns=['test_mae_numeric'])  # 删除临时排序列

    # 保存汇总CSV
    summary_save_path = os.path.join(base_config.save_path, 'hyperparameter_search_summary.csv')
    results_df.to_csv(summary_save_path, index=False)

    # --- 5. 打印最终报告 ---
    print("--- 超参数搜索汇总报告 ---")
    print(results_df.to_string())  # 打印所有结果
    print("\n--- 最佳超参数组合 (基于最低测试集 MAE) ---")

    if not results_df.empty and pd.to_numeric(results_df.iloc[0]['test_mae'], errors='coerce') is not np.nan:
        best_trial = results_df.iloc[0]
        print(f"  - n_terms:         {best_trial['n_terms']}")
        print(f"  - learning_rate:   {best_trial['learning_rate']}")
        print(f"  - patience:        {best_trial['patience']}")
        print(f"  - ==> 测试集 MAE:  {best_trial['test_mae']:.6f}")
        print(f"  - ==> 测试集 R2:     {best_trial['test_r2']:.6f}")
        print(f"\n  - 最佳模型及图表已保存于: {best_trial['trial_save_path']}")
    else:
        print("未能找到有效的最佳试验结果。")

    print(f"\n汇总报告已保存到: {summary_save_path}")


if __name__ == '__main__':
    main()