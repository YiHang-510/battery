import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib
import shutil  # 导入shutil库

# --- 前置准备：确保绘图后端和字体正常 ---
matplotlib.use('Agg')


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# =================================================================================
# 1. 模型定义 (从 cyclenet3.3-forcyclenum.py 和 exp-net-predict.py 移植)
# =================================================================================

# --- ExpNet 模型定义 ---
class ExpNet(nn.Module):
    def __init__(self, n_terms=16):
        super(ExpNet, self).__init__()
        # 每组都有a, b, d三个参数，共n_terms组
        # 乘一个小负数防止梯度爆炸
        self.b = nn.Parameter(torch.ones(n_terms) * -0.01)
        self.a = nn.Parameter(torch.ones(n_terms) * 1.0)
        self.d = nn.Parameter(torch.ones(n_terms))
        self.n_terms = n_terms

    def forward(self, c):
        # c: [batch_size,] 或 [batch_size, 1]
        c = c.view(-1, 1)  # [batch_size, 1]
        a = self.a.view(1, -1)  # [1, n_terms]
        b = self.b.view(1, -1)
        d = self.d.view(1, -1)
        # 广播，计算每组参数的输出
        out = a * torch.exp(b * c) + d  # [batch_size, n_terms]
        # 你可以选择sum或mean，也可以直接输出所有组
        out = out.sum(dim=1)  # [batch_size]
        return out


# --- CycleNet 模型定义 ---
class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]


class CycleNetForSOH(nn.Module):
    def __init__(self, configs):
        super(CycleNetForSOH, self).__init__()
        self.configs = configs
        self.sequence_encoder = nn.Linear(configs['sequence_length'] * configs['sequence_feature_dim'],
                                          configs['d_model'] // 2)
        self.scalar_encoder = nn.Linear(configs['scalar_feature_dim'], configs['d_model'] // 2)
        self.combined_feature_dim = configs['d_model']
        self.cycle_queue = RecurrentCycle(
            cycle_len=configs['meta_cycle_len'],
            channel_size=self.combined_feature_dim
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, configs['d_ff']),
            nn.ReLU(),
            nn.Dropout(configs['dropout']),
            nn.Linear(configs['d_ff'], 1)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        x_seq_flat = x_seq.view(x_seq.size(0), -1)
        seq_embedding = self.sequence_encoder(x_seq_flat)
        scalar_embedding = self.scalar_encoder(x_scalar)
        combined_features = torch.cat((seq_embedding, scalar_embedding), dim=1)
        cycle_index = cycle_number % self.configs['meta_cycle_len']
        decycled_features = combined_features - self.cycle_queue(cycle_index, length=1).squeeze(1)
        prediction = self.prediction_head(decycled_features)
        return prediction


# =================================================================================
# 2. 配置参数 (路径已修改为基础路径)
# =================================================================================
class Config:
    def __init__(self):
        # --- 1. 输入路径设置 ---
        # 原始数据路径
        self.data_path_sequence = r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x'
        self.data_path_features = r'/home/scuee_user06/myh/电池/data/selected_feature/statistic'

        # ▼▼▼【核心修改】这里设置为包含 run_1, run_2... 的父目录 ▼▼▼
        self.cyclenet_base_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/cyclenet/all'
        self.expnet_base_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/result-expnet-16/all'

        # --- 2. 输出路径设置 ---
        self.save_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/combine_prediction-16/all'

        # --- 3. 待测试电池ID ---
        self.test_battery_ids = [6, 12, 14, 20]
        # self.test_battery_ids = [20]

        # --- 4. 模型配置 (必须与训练时完全一致!) ---
        self.cyclenet_config = {
            'sequence_length': 1,
            'sequence_feature_dim': 7,
            'scalar_feature_dim': 2,
            'meta_cycle_len': 7,
            'd_model': 256,
            'd_ff': 1024,
            'dropout': 0.2,
            'features_from_C': [
                '恒压充电时间(s)',
                '3.3~3.6V充电时间(s)',
            ]
        }
        self.expnet_config = {
            'n_terms': 16,
            'nominal_capacity': 3.5
        }

        # --- 5. 其他设置 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================================================
# 3. 数据加载与预处理函数
# =================================================================================
def load_and_preprocess_single_battery(config, scalers, battery_id):
    """为单个测试电池加载并预处理数据"""
    # print(f"--- 步骤 1: 正在为电池 {battery_id} 加载和预处理数据... ---")
    try:
        path_a = os.path.join(config.data_path_sequence, f'relaxation_battery{battery_id}.csv')
        path_c = os.path.join(config.data_path_features, f'battery{battery_id}_SOH健康特征提取结果.csv')
        df_a = pd.read_csv(path_a)
        df_c = pd.read_csv(path_c)
        df_c.rename(columns=lambda x: x.strip(), inplace=True)
    except FileNotFoundError as e:
        print(f"警告: 电池 {battery_id} 的数据文件未找到，已跳过。错误: {e}")
        return None, None, None, None, None, None

    seq_conf = config.cyclenet_config
    feature_cols = [f'弛豫段电压{i}' for i in range(1, seq_conf['sequence_feature_dim'] + 1)]
    sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(name='voltage_sequence')
    sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == seq_conf['sequence_length']]

    full_df = pd.merge(sequence_df, df_c, on='循环号')

    true_capacity = full_df['最大容量(Ah)'].values
    true_soh = true_capacity / config.expnet_config['nominal_capacity']

    scaler_seq = scalers['sequence']
    scaler_scalar = scalers['scalar']

    scalar_feature_cols = seq_conf['features_from_C']
    full_df['voltage_sequence'] = full_df['voltage_sequence'].apply(lambda x: scaler_seq.transform(x))
    full_df[scalar_feature_cols] = scaler_scalar.transform(full_df[scalar_feature_cols])

    sequences = np.array(full_df['voltage_sequence'].tolist(), dtype=np.float32)
    scalars = full_df[scalar_feature_cols].values.astype(np.float32)
    cycle_indices = full_df['循环号'].values.astype(np.int64)

    x_seq_tensor = torch.from_numpy(sequences).to(config.device)
    x_scalar_tensor = torch.from_numpy(scalars).to(config.device)
    cycle_idx_tensor = torch.from_numpy(cycle_indices).to(config.device)
    cumulative_capacity = full_df['累计放电容量(Ah)'].values

    return x_seq_tensor, x_scalar_tensor, cycle_idx_tensor, true_soh, full_df['循环号'].values, cumulative_capacity


# =================================================================================
# 4. 预测函数
# =================================================================================
def predict_capacity_with_cyclenet(model, scalers, x_seq, x_scalar, cycle_idx):
    """使用CycleNet预测累计放电容量"""
    model.eval()
    with torch.no_grad():
        scaled_preds = model(x_seq, x_scalar, cycle_idx).cpu().numpy()
    scaler_target = scalers['target']
    predicted_capacity = scaler_target.inverse_transform(scaled_preds).flatten()
    return predicted_capacity


def predict_soh_with_expnet(model, predicted_capacity, device):
    """使用ExpNet和预测的容量来预测SOH"""
    model.eval()
    capacity_tensor = torch.tensor(predicted_capacity, dtype=torch.float32, device=device)
    with torch.no_grad():
        predicted_soh = model(capacity_tensor).cpu().numpy()
    return predicted_soh


# =================================================================================
# 5. 评估与可视化
# =================================================================================
def plot_diagonal_scatter(labels, preds, title, save_path):
    """绘制真实值与预测值的对角散点图。"""
    plt.figure(figsize=(8, 8))
    min_val = min(np.min(labels), np.min(preds)) * 0.98
    max_val = max(np.max(labels), np.max(preds)) * 1.02
    plt.scatter(labels, preds, alpha=0.6, label='True Value vs. Predicted Value')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Best prediction'   )
    plt.xlabel('True SOH', fontsize=12)
    plt.ylabel('Predicted SOH', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_and_visualize(save_path, battery_id, cycle_nums, true_soh, final_pred_soh, cumulative_capacity):
    """计算指标、保存结果并绘图"""
    mae = mean_absolute_error(true_soh, final_pred_soh)
    mape = mean_absolute_percentage_error(true_soh, final_pred_soh)
    mse = mean_squared_error(true_soh, final_pred_soh)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_soh, final_pred_soh)

    metrics_data = {'Battery_ID': [battery_id], 'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2]}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_save_path = os.path.join(save_path, 'fusion_model_evaluation_metrics.csv')

    # 追加模式写入指标
    if os.path.exists(metrics_save_path):
        metrics_df.to_csv(metrics_save_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        metrics_df.to_csv(metrics_save_path, index=False, encoding='utf-8')

    results_df = pd.DataFrame({'循环号': cycle_nums, '真实SOH': true_soh, '预测SOH': final_pred_soh})
    csv_path = os.path.join(save_path, f'battery_{battery_id}_fusion_prediction.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8')

    plt.figure(figsize=(12, 7))
    plt.plot(cumulative_capacity, true_soh, 'o-', label='True SOH', color='royalblue', markersize=4)
    plt.plot(cumulative_capacity, final_pred_soh, '^-', label='Predicted SOH', color='darkorange', markersize=4,
             alpha=0.8)
    plt.title(f'battery {battery_id}: True SOH vs. Predicted SOH', fontsize=16)
    plt.xlabel('Accumulated discharge capacity (Ah)', fontsize=12)
    plt.ylabel('SOH', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'battery_{battery_id}_soh_comparison_plot.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    scatter_plot_path = os.path.join(save_path, f'battery_{battery_id}_diagonal_scatter_plot.png')
    plot_diagonal_scatter(labels=true_soh, preds=final_pred_soh,
                          title=f'battery {battery_id}: True SOH vs. Predicted SOH', save_path=scatter_plot_path)


# =================================================================================
# 6. 主执行函数 (已重构)
# =================================================================================
def main():
    config = Config()
    os.makedirs(config.save_path, exist_ok=True)
    print(f"使用设备: {config.device}")

    # --- 新增：为多次实验设置的变量 ---
    num_runs = 5
    all_runs_metrics = []
    best_run_mae = float('inf')
    best_run_dir = None
    best_run_number = -1

    # --- 开始多次实验循环 ---
    for run_number in range(1, num_runs + 1):
        print(f"\n{'=' * 30}")
        print(f" 开始第 {run_number}/{num_runs} 次融合实验 ")
        print(f"{'=' * 30}")

        # --- 1. 动态构建当前轮次的路径 ---
        current_cyclenet_path = os.path.join(config.cyclenet_base_path, f'run_{run_number}')
        current_expnet_path = os.path.join(config.expnet_base_path, f'run_{run_number}')
        run_save_path = os.path.join(config.save_path, f'run_{run_number}')
        os.makedirs(run_save_path, exist_ok=True)

        # 清理上一轮可能存在的旧指标文件，确保每次都是重新计算
        metrics_summary_file = os.path.join(run_save_path, 'fusion_model_evaluation_metrics.csv')
        if os.path.exists(metrics_summary_file):
            os.remove(metrics_summary_file)

        # --- 2. 加载当前轮次的模型和缩放器 ---
        try:
            cyclenet_model_path = os.path.join(current_cyclenet_path, 'best_model.pth')
            expnet_model_path = os.path.join(current_expnet_path, 'best_expnet_model.pth')
            scalers_path = os.path.join(current_cyclenet_path, 'scalers.pkl')

            cyclenet_model = CycleNetForSOH(config.cyclenet_config).to(config.device)
            cyclenet_model.load_state_dict(torch.load(cyclenet_model_path, map_location=config.device))

            expnet_model = ExpNet(n_terms=config.expnet_config['n_terms']).to(config.device)
            expnet_model.load_state_dict(torch.load(expnet_model_path, map_location=config.device))

            scalers = joblib.load(scalers_path)
            print(f"Run {run_number}: 模型和缩放器加载成功！")
        except Exception as e:
            print(f"\n错误: 无法加载 Run {run_number} 的模型或缩放器文件: {e}")
            print("请确保之前的训练脚本已成功生成对应 run 文件夹。跳过此次实验。")
            continue

        # --- 3. 循环处理每个测试电池 ---
        for battery_id in config.test_battery_ids:
            print(f"\n--- 正在处理电池 {battery_id} ---")

            x_seq, x_scalar, cycle_idx, true_soh, cycle_nums, cumulative_capacity = load_and_preprocess_single_battery(
                config, scalers, battery_id)
            if x_seq is None:
                continue

            predicted_capacity = predict_capacity_with_cyclenet(cyclenet_model, scalers, x_seq, x_scalar, cycle_idx)
            # 强制执行物理约束：累计容量不能为负。这可以防止 ExpNet 爆炸。
            # 此时 predicted_capacity 是一个 numpy 数组, 所以我们使用 np.clip。
            predicted_capacity = np.clip(predicted_capacity, a_min=0.0, a_max=None)
            final_predicted_soh = predict_soh_with_expnet(expnet_model, predicted_capacity, config.device)

            evaluate_and_visualize(
                save_path=run_save_path,
                battery_id=battery_id,
                cycle_nums=cycle_nums,
                true_soh=true_soh,
                final_pred_soh=final_predicted_soh,
                cumulative_capacity=cumulative_capacity
            )
            print(f"电池 {battery_id} 处理完成。结果已保存至 {run_save_path}")

        # --- 4. 汇总本轮所有电池的指标 ---
        if os.path.exists(metrics_summary_file):
            run_metrics_df = pd.read_csv(metrics_summary_file)
            avg_metrics = run_metrics_df.mean().to_dict()  # 计算平均指标
            avg_metrics['run'] = run_number
            all_runs_metrics.append(avg_metrics)

            current_run_mae = avg_metrics.get('MAE', float('inf'))
            print(f"\n--- Run {run_number} 评估汇总 ---")
            print(f"  - 平均 MAE: {current_run_mae:.4f}")
            print(f"  - 平均 R²:  {avg_metrics.get('R2', 0):.4f}")

            # --- 5. 检查是否为最佳轮次 ---
            if current_run_mae < best_run_mae:
                best_run_mae = current_run_mae
                best_run_dir = run_save_path
                best_run_number = run_number
                print(f"*** 新的最佳表现！平均 MAE: {best_run_mae:.4f} ***")

    # --- 循环结束后 ---
    print(f"\n\n{'=' * 50}")
    print(" 所有融合实验均已完成。")
    print(f"{'=' * 50}")

    # 1. 保存所有轮次的指标汇总
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        # Reorder columns for better readability
        cols = ['run'] + [col for col in summary_df.columns if col != 'run']
        summary_df = summary_df[cols]
        summary_path = os.path.join(config.save_path, 'all_runs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 ---")
        print(summary_df.to_string())
        print(f"\n汇总指标已保存到: {summary_path}")

    # 2. 将最佳轮次的结果复制到主目录
    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (平均 MAE 最低: {best_run_mae:.4f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")

        for filename in os.listdir(best_run_dir):
            source_file = os.path.join(best_run_dir, filename)
            destination_file = os.path.join(config.save_path, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)

        print("最佳结果复制完成。")
    else:
        print("未能确定最佳实验轮次，可能是因为所有轮次的模型文件都未能成功加载。")

    print(f"\n评估完成。所有结果已保存到: {config.save_path}")


if __name__ == '__main__':
    main()