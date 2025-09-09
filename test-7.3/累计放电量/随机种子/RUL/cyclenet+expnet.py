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
"""这一步在 calculate_and_evaluate_RUL 函数中执行。脚本并不是通过模型来迭代预测未来的SOH值。

它只是在刚刚生成的“预测SOH”曲线上，从头到尾进行搜索 (np.where(final_pred_soh < FAILURE_THRESHOLD)[0])。

它找到第一个使得预测SOH值低于失败阈值（例如 0.75 SOH）的循环索引。这个索引对应的循环号（例如第1080圈）就被定义为**“预测的寿命终点 (Predicted EOL)”**。"""
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
        self.cyclenet_base_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/cyclenet/12'
        self.expnet_base_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/result-expnet/12'

        # --- 2. 输出路径设置 ---
        self.save_path = r'/home/scuee_user06/myh/电池/result-累计放电容量/RULcombine_prediction/12'

        # --- 3. 待测试电池ID ---
        # self.test_battery_ids = [6, 12, 14, 20]
        self.test_battery_ids = [12]

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
            'n_terms': 4,
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
# 5. 评估与可视化 (修改为RUL任务)
# =================================================================================

def plot_RUL_comparison(cycle_nums, true_rul_array, pred_rul_array, title, save_path):
    """绘制真实RUL与预测RUL的对比曲线图"""
    plt.figure(figsize=(12, 7))
    # 我们只绘制到真实故障点为止的数据
    plt.plot(cycle_nums, true_rul_array, 'o-', label='True RUL', color='royalblue', markersize=4)
    plt.plot(cycle_nums, pred_rul_array, '^-', label='Predicted RUL', color='darkorange', markersize=4, alpha=0.8)
    plt.title(title, fontsize=16)
    plt.xlabel('Cycle Number', fontsize=12)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"RUL对比图已保存至: {save_path}")


def plot_RUL_diagonal(true_rul_array, pred_rul_array, title, save_path):
    """绘制真实RUL与预测RUL的对角散点图"""
    plt.figure(figsize=(8, 8))
    min_val = 0  # RUL 最小为0
    max_val = max(np.max(true_rul_array), np.max(pred_rul_array)) * 1.02

    plt.scatter(true_rul_array, pred_rul_array, alpha=0.6, label='Predicted vs. True')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')

    plt.xlabel('True RUL (Cycles)', fontsize=12)
    plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"RUL对角图已保存至: {save_path}")


def calculate_and_evaluate_RUL(save_path, battery_id, cycle_nums, true_soh, final_pred_soh, cumulative_capacity):
    """
    核心函数：通过外推SOH曲线来计算和评估RUL。
    """
    print("\n--- 步骤 4: 通过SOH外推法计算和评估RUL... ---")

    # 1. 定义SOH故障阈值
    FAILURE_THRESHOLD = 0.75  # (即 3.5 * 0.7 = 2.45Ah)

    # 2. 查找真实故障点
    true_failure_indices = np.where(true_soh < FAILURE_THRESHOLD)[0]
    if len(true_failure_indices) == 0:
        print(f"  -> 警告: 电池 {battery_id} 的真实数据从未低于70% SOH。无法计算RUL指标，跳过评估。")
        return None  # 返回None，表示评估失败

    true_failure_idx = true_failure_indices[0]  # 第一个低于阈值的索引
    true_failure_cycle = cycle_nums[true_failure_idx]
    print(f"  - 真实故障点: 第 {true_failure_cycle} 圈 (在索引 {true_failure_idx} 处)")

    # 3. 查找预测故障点
    pred_failure_indices = np.where(final_pred_soh < FAILURE_THRESHOLD)[0]
    if len(pred_failure_indices) == 0:
        print(f"  -> 警告: 模型的 *预测曲线* 从未低于70% SOH。模型预测失败。")
        # 这种情况我们给一个极大的惩罚值，或者设为None
        pred_failure_cycle = cycle_nums[-1] + 100  # 假设它在数据范围外很远才失败
        print(f"  - 预测故障点: 未找到 (超出观测范围)")
    else:
        pred_failure_idx = pred_failure_indices[0]
        pred_failure_cycle = cycle_nums[pred_failure_idx]
        print(f"  - 预测故障点: 第 {pred_failure_cycle} 圈 (在索引 {pred_failure_idx} 处)")

    # 4. 计算RUL评估指标
    # RUL评估有两个主要指标：
    # 指标A: 寿命终点(EOL)误差，即两个故障点之间的绝对差值
    eol_error = abs(true_failure_cycle - pred_failure_cycle)

    # 指标B: 逐点的RUL误差 (MAE / RMSE)
    # 我们只在真实故障点之前的数据上进行比较 (即从 cycle 0 到 true_failure_idx)
    eval_range_slice = slice(0, true_failure_idx + 1)  # 包含故障点本身

    # 计算真实RUL向量 (例如: [1050, 1049, ..., 1, 0])
    true_rul_array = true_failure_cycle - cycle_nums[eval_range_slice]
    # 计算预测RUL向量
    pred_rul_array = pred_failure_cycle - cycle_nums[eval_range_slice]
    # RUL不能为负，对预测值进行钳制
    pred_rul_array = np.clip(pred_rul_array, a_min=0.0, a_max=None)

    mae_rul = mean_absolute_error(true_rul_array, pred_rul_array)
    rmse_rul = np.sqrt(mean_squared_error(true_rul_array, pred_rul_array))
    r2_rul = r2_score(true_rul_array, pred_rul_array)

    print("\n--- 融合模型最终RUL预测评估结果 ---")
    print(f"  - EOL 误差: {eol_error} (Cycles)")
    print(f"  - RUL MAE:  {mae_rul:.4f} (Cycles)")
    print(f"  - RUL RMSE: {rmse_rul:.4f} (Cycles)")
    print(f"  - RUL R²:   {r2_rul:.4f}")

    # 5. 保存RUL指标到CSV
    # --- ★★★ 关键修改：存储标量（纯数字），而不是列表 ---
    metrics_data = {
        'Battery_ID': battery_id,
        'EOL_Error_Cycles': eol_error,
        'RUL_MAE_Cycles': mae_rul,
        'RUL_RMSE_Cycles': rmse_rul,
        'RUL_R2': r2_rul,
        'True_EOL_Cycle': true_failure_cycle,
        'Pred_EOL_Cycle': pred_failure_cycle
    }

    # --- ★★★ 关键修改：从标量字典创建DataFrame时，必须将其包在列表中，Pandas才会将其视为单行数据 ---
    metrics_df = pd.DataFrame([metrics_data])

    # 按照您的要求，我们仍然使用这个文件名
    metrics_save_path = os.path.join(save_path, 'fusion_model_evaluation_metrics.csv')

    if os.path.exists(metrics_save_path):
        metrics_df.to_csv(metrics_save_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        metrics_df.to_csv(metrics_save_path, index=False, encoding='utf-8')
    print(f"\nRUL评估指标已保存至: {metrics_save_path}")

    # 6. 保存详细的RUL预测数据
    results_df = pd.DataFrame({
        '循环号': cycle_nums[eval_range_slice],
        '真实SOH': true_soh[eval_range_slice],
        '预测SOH': final_pred_soh[eval_range_slice],
        '真实RUL(cycles)': true_rul_array,
        '预测RUL(cycles)': pred_rul_array
    })
    csv_path = os.path.join(save_path, f'battery_{battery_id}_fusion_RUL_prediction.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"详细RUL预测结果已保存至: {csv_path}")

    # 7. 绘制新的RUL图表
    plot_RUL_comparison(
        cycle_nums[eval_range_slice], true_rul_array, pred_rul_array,
        f'Battery {battery_id}: True vs Predicted RUL (Extrapolation Method)',
        os.path.join(save_path, f'battery_{battery_id}_RUL_comparison_plot.png')
    )

    plot_RUL_diagonal(
        true_rul_array, pred_rul_array,
        f'Battery {battery_id}: True RUL vs Predicted RUL (Diagonal)',
        os.path.join(save_path, f'battery_{battery_id}_RUL_diagonal_scatter_plot.png')
    )

    # 返回评估指标用于汇总
    return metrics_data
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
        # --- 3. 循环处理每个测试电池 (★ 已修改为RUL外推流程 ★) ---
        run_battery_metrics_list = []  # 存储本轮所有电池的RUL指标

        for battery_id in config.test_battery_ids:
            print(f"\n--- 正在处理电池 {battery_id} (SOH外推RUL流程) ---")

            # 1. 加载和预处理数据 (返回6个值)
            x_seq, x_scalar, cycle_idx, true_soh, cycle_nums, cumulative_capacity = load_and_preprocess_single_battery(
                config, scalers, battery_id)

            if x_seq is None:
                continue

            # 2. 流程 步骤 1: CycleNet 预测累计放电容量
            predicted_capacity = predict_capacity_with_cyclenet(cyclenet_model, scalers, x_seq, x_scalar, cycle_idx)

            # 3. ★ 物理约束 (防爆) ★
            predicted_capacity = np.clip(predicted_capacity, a_min=0.0, a_max=None)

            # 4. 流程 步骤 2: ExpNet (SOH模型) 预测SOH曲线
            final_predicted_soh = predict_soh_with_expnet(expnet_model, predicted_capacity, config.device)

            # 5. ★ 物理约束 (SOH) ★
            # SOH不能为负 (尽管ExpNet不太可能预测为负, 但以防万一)
            # SOH也不能超过其起始点 (通常为1.x), 但我们主要关心故障点，所以只钳制底部
            final_predicted_soh = np.clip(final_predicted_soh, a_min=0.0, a_max=None)

            # 6. 评估和可视化 (使用全新的RUL计算函数)
            rul_metrics = calculate_and_evaluate_RUL(
                save_path=run_save_path,
                battery_id=battery_id,
                cycle_nums=cycle_nums,
                true_soh=true_soh,
                final_pred_soh=final_predicted_soh,
                cumulative_capacity=cumulative_capacity
            )

            if rul_metrics:  # 如果评估成功 (即电池达到了故障点)
                run_battery_metrics_list.append(rul_metrics)
                print(f"电池 {battery_id} RUL评估完成。结果已保存至 {run_save_path}")
            else:
                print(f"电池 {battery_id} RUL评估跳过。")

        # --- 4. 汇总本轮所有电池的RUL指标 ---
        if run_battery_metrics_list:
            # 将列表（每个元素是一个字典）转换为DataFrame以计算平均值
            run_metrics_df = pd.DataFrame(run_battery_metrics_list)
            # 移除Battery_ID列，以便对其他所有指标列计算均值
            avg_metrics = run_metrics_df.drop(columns=['Battery_ID']).mean().to_dict()
            avg_metrics['run'] = run_number
            all_runs_metrics.append(avg_metrics)  # 添加到全局列表

            current_run_mae = avg_metrics.get('RUL_MAE_Cycles', float('inf'))
            print(f"\n--- Run {run_number} RUL 评估汇总 ---")
            print(f"  - 平均 RUL MAE (Cycles): {current_run_mae:.4f}")
            print(f"  - 平均 EOL 误差 (Cycles): {avg_metrics.get('EOL_Error_Cycles', 0):.4f}")

            # --- 5. 检查是否为最佳轮次 (基于RUL MAE) ---
            if current_run_mae < best_run_mae:
                best_run_mae = current_run_mae
                best_run_dir = run_save_path
                best_run_number = run_number
                print(f"*** 新的最佳表现！平均 RUL MAE: {best_run_mae:.4f} Cycles ***")
        else:
            print(f"\n--- Run {run_number} 评估汇总 ---")
            print("  - 未能计算任何RUL指标 (可能测试电池均未达到故障标准)。")

    # --- 循环结束后 ---
    print(f"\n\n{'=' * 50}")
    print(" 所有融合实验均已完成。")
    print(f"{'=' * 50}")

    # 1. 保存所有轮次的(RUL)指标汇总
    if all_runs_metrics:
        summary_df = pd.DataFrame(all_runs_metrics)
        cols = ['run'] + [col for col in summary_df.columns if col != 'run']
        summary_df = summary_df[cols]
        summary_path = os.path.join(config.save_path, 'all_runs_RUL_summary.csv')  # 重命名
        summary_df.to_csv(summary_path, index=False)
        print("\n--- 五次实验性能汇总 (RUL平均指标) ---")
        print(summary_df.to_string())
        print(f"\nRUL汇总指标已保存到: {summary_path}")

    # 2. 将最佳轮次的结果复制到主目录
    if best_run_dir:
        print(f"\n表现最佳的实验是第 {best_run_number} 轮 (平均 RUL MAE 最低: {best_run_mae:.4f})。")
        print(f"正在将最佳结果从 {best_run_dir} 复制到主目录 {config.save_path} ...")

        for filename in os.listdir(best_run_dir):
            source_file = os.path.join(best_run_dir, filename)
            destination_file = os.path.join(config.save_path, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, destination_file)

        print("最佳结果复制完成。")
    else:
        print("未能确定最佳实验轮次，可能是因为所有轮次的模型文件都未能成功加载或测试电池未达标。")

    print(f"\n评估完成。所有结果已保存到: {config.save_path}")

    # (我们不再需要那个“跨实验分电池”的汇总，因为这个脚本的主要产物就是RUL指标，
    # 它们已经保存在 run_X 文件夹中的 fusion_model_evaluation_metrics.csv
    # 以及根目录的 all_runs_RUL_summary.csv 中)


if __name__ == '__main__':
    main()