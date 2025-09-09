# predict_pipeline.py

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, Dataset
import warnings
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 忽略不必要的警告
warnings.filterwarnings('ignore')
# 预先设置matplotlib，避免在无图形界面的服务器上出错
matplotlib.use('Agg')
# =================================================================================
# 模型定义区域
# 必须包含两个模型的结构定义，以便PyTorch加载权重文件。
# =================================================================================

# --- 来自 exp-net-predict.py 的模型定义 ---
# +++ 使用下面这段新的代码替换 +++
# +++ 使用下面这个最终版本的代码替换旧的 ExpNet 类定义 +++
class ExpNet(nn.Module):
    """
    ExpNet的模型结构 - 最终修正版
    根据错误信息推断，参数 d 的尺寸应为 n_terms。
    模型结构可能为 SOH = sum(a_i * exp(b_i * cycle) + d_i)。
    """
    def __init__(self, n_terms=4):
        super(ExpNet, self).__init__()
        self.n_terms = n_terms
        # 定义独立的参数a, b, d
        self.a = nn.Parameter(torch.randn(n_terms))
        self.b = nn.Parameter(torch.randn(n_terms))
        # -> 修改第1处：将 d 的尺寸从 (1) 修改为 (n_terms)
        self.d = nn.Parameter(torch.randn(n_terms))

    def forward(self, x):
        # 确保输入x的形状为 [batch_size, 1] 以便进行广播
        x = x.unsqueeze(-1)

        # 计算 SOH = sum(a * exp(b*x) + d)
        # a, b, d 的尺寸都是 [n_terms]
        # x 的尺寸是 [batch_size, 1]
        # a * exp(b * x) 的尺寸是 [batch_size, n_terms]
        # -> 修改第2处：将 +d 的操作移到求和符号内部
        terms = self.a * torch.exp(self.b * x) + self.d

        # 对所有项的结果求和
        return torch.sum(terms, dim=1)


# --- 来自 cyclenet3.3-forcyclenum.py 的模型定义 ---
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
    def __init__(self, d_model, sequence_feature_dim, scalar_feature_dim, d_ff, meta_cycle_len, dropout):
        super(CycleNetForSOH, self).__init__()
        self.meta_cycle_len = meta_cycle_len
        self.sequence_encoder = nn.Linear(1 * sequence_feature_dim, d_model // 2)
        self.scalar_encoder = nn.Linear(scalar_feature_dim, d_model // 2)
        self.combined_feature_dim = d_model
        self.cycle_queue = RecurrentCycle(cycle_len=meta_cycle_len, channel_size=self.combined_feature_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )

    def forward(self, x_seq, x_scalar, cycle_number):
        x_seq_flat = x_seq.view(x_seq.size(0), -1)
        seq_embedding = self.sequence_encoder(x_seq_flat)
        scalar_embedding = self.scalar_encoder(x_scalar)
        combined_features = torch.cat((seq_embedding, scalar_embedding), dim=1)
        cycle_index = cycle_number % self.meta_cycle_len
        decycled_features = combined_features - self.cycle_queue(cycle_index, length=1).squeeze(1)
        prediction = self.prediction_head(decycled_features)
        return prediction

# =================================================================================
# 数据加载与预处理区域
# =================================================================================

class PredictionDataset(Dataset):
    """用于加载预测数据的Dataset类。"""
    def __init__(self, dataframe, sequence_col, scalar_cols):
        self.df = dataframe.reset_index(drop=True)
        self.sequences = np.array(self.df[sequence_col].tolist(), dtype=np.float32)
        self.scalars = self.df[scalar_cols].values.astype(np.float32)
        self.cycle_indices = self.df['循环号'].values.astype(np.int64)
        self.battery_ids = self.df['battery_id'].values.astype(np.int64)
        if '最大容量(Ah)' in self.df.columns:
            self.true_soh = self.df['最大容量(Ah)'].values.astype(np.float32)
        else:
            self.true_soh = np.full(len(self.df), -1, dtype=np.float32) # 如果没有真实SOH，用-1填充

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_seq = torch.from_numpy(self.sequences[idx])
        x_scalar = torch.from_numpy(self.scalars[idx])
        cycle_idx = torch.tensor(self.cycle_indices[idx], dtype=torch.long)
        battery_id = torch.tensor(self.battery_ids[idx], dtype=torch.long)
        true_soh = torch.tensor(self.true_soh[idx], dtype=torch.float32)
        return x_seq, x_scalar, cycle_idx, battery_id, true_soh

def load_and_preprocess_for_prediction(battery_ids, data_paths, feature_config, scalers):
    """为预测任务加载并预处理指定电池的数据。"""
    all_battery_data = []
    print("开始加载并预处理数据...")

    for battery_id in battery_ids:
        try:
            print(f"  - 正在处理电池: {battery_id}...")
            path_a = os.path.join(data_paths['A_sequence'], f'relaxation_battery{battery_id}.csv')
            path_c = os.path.join(data_paths['C_features'], f'battery{battery_id}_SOH健康特征提取结果.csv')

            df_a = pd.read_csv(path_a)
            df_c = pd.read_csv(path_c)
            df_c.rename(columns=lambda x: x.strip(), inplace=True)

            feature_cols = [f'弛豫段电压{i}' for i in range(1, feature_config['sequence_dim'] + 1)]
            sequence_df = df_a.groupby('循环号')[feature_cols].apply(lambda x: x.values).reset_index(name='voltage_sequence')
            sequence_df = sequence_df[sequence_df['voltage_sequence'].apply(len) == feature_config['sequence_length']]

            final_df = pd.merge(sequence_df, df_c, on='循环号', how='left')
            final_df['battery_id'] = battery_id
            all_battery_data.append(final_df)

        except Exception as e:
            print(f"    警告: 无法处理电池 {battery_id}。错误: {e}")
            continue

    if not all_battery_data:
        raise ValueError("未能加载任何电池数据，请检查路径和电池ID。")

    full_df = pd.concat(all_battery_data, ignore_index=True)

    # 应用加载好的StandardScaler进行数据标准化
    scaler_seq = scalers['sequence']
    scaler_scalar = scalers['scalar']
    scalar_feature_cols = feature_config['scalar_features']

    # 确保所有需要的特征列都存在
    for col in scalar_feature_cols:
        if col not in full_df.columns:
            raise ValueError(f"错误：特征列 '{col}' 在数据中不存在，请检查配置文件。")

    full_df['voltage_sequence'] = full_df['voltage_sequence'].apply(lambda x: scaler_seq.transform(x))
    full_df.loc[:, scalar_feature_cols] = scaler_scalar.transform(full_df[scalar_feature_cols])

    print("数据加载与预处理完成。")
    return full_df, scalar_feature_cols

# =================================================================================
# 新增：绘图函数区域
# =================================================================================

def plot_results(labels, preds, title, save_path, y_label='数值'):
    """
    绘制真实值与预测值的折线对比图。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(labels, label='真实值', marker='o', linestyle='-', markersize=4, alpha=0.7)
    plt.plot(preds, label='预测值', marker='x', linestyle='--', markersize=4, alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_diagonal_results(labels, preds, title, save_path, xy_label='数值'):
    """
    绘制真实值与预测值的对角散点图。
    """
    plt.figure(figsize=(8, 8))
    # 找到x和y轴的共同范围
    min_val = min(np.min(labels), np.min(preds)) * 0.98
    max_val = max(np.max(labels), np.max(preds)) * 1.02

    # 绘制散点图
    plt.scatter(labels, preds, alpha=0.6, label='预测点')
    # 绘制y=x的完美预测线
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想情况 (y=x)')

    plt.xlabel(f'真实{xy_label}', fontsize=12)
    plt.ylabel(f'预测{xy_label}', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    # 设置坐标轴为相等比例，并限定范围
    plt.axis('equal')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.savefig(save_path, dpi=300)
    plt.close()

# =================================================================================
# 主程序入口
# =================================================================================

def main():
    """
    主函数，用于执行完整的串联预测流程。
    """
    # ### 1. 配置区域 ###
    # !!! 请务必根据您的实际情况修改以下路径 !!!
    config = {
        # 1.1 数据输入路径
        'data_paths': {
            'A_sequence': r'/home/scuee_user06/myh/电池/data/selected_feature/relaxation/Interval-singleraw-200x',
            'C_features': r'/home/scuee_user06/myh/电池/data/selected_feature/statistic',
        },
        # 1.2 模型和Scaler文件的路径
        'model_paths': {
            # 示例: '/home/.../cyclenet_result-forcyclenum/all/train...test.../best_model_...pth'
            'cycle_net': '/home/scuee_user06/myh/电池/data/modelfile/b6_cyclenet.pth',
            # 示例: '/home/.../expnet_result/all/best_expnet_model.pth'
            'exp_net': '/home/scuee_user06/myh/电池/data/modelfile/b6_expnet.pth',
            # 示例: '/home/.../cyclenet_result-forcyclenum/all/train...test.../scalers.pkl'
            'scalers': '/home/scuee_user06/myh/电池/data/modelfile/b6_cyclenet.pkl',
        },

        #/home/scuee_user06/myh/电池/data/modelfile
        # 1.3 预测结果输出路径
        'output_path': '/home/scuee_user06/myh/电池/data/combine_predict/6/final_soh_predictions.csv',

        # 1.3.1 (新增) 绘图结果输出路径
        'plot_paths': {
            'cycle_line_plot': '/home/scuee_user06/myh/电池/data/combine_predict/6/plot_cycle_line.png',
            'cycle_diagonal_plot': '/home/scuee_user06/myh/电池/data/combine_predict/6/plot_cycle_diagonal.png',
            'soh_line_plot': '/home/scuee_user06/myh/电池/data/combine_predict/6/plot_soh_line.png',
            'soh_diagonal_plot': '/home/scuee_user06/myh/电池/data/combine_predict/6/plot_soh_diagonal.png',
        },

        # 在 config = {...} 字典内部添加
        # 1.3.2 (新增) 评估指标文件的保存路径
        'metrics_path': '/home/scuee_user06/myh/电池/data/combine_predict/6/final_evaluation_metrics.csv',

        # 1.4 定义需要进行预测的电池编号
        # 'prediction_batteries': [6, 12, 14, 20], # 使用您脚本中的测试集作为示例
        'prediction_batteries': [6],
        # 1.5 模型超参数 (必须与训练时完全一致)
        'cycle_net_params': {
            'd_model': 256,
            'd_ff': 1024,
            'meta_cycle_len': 7,
            'dropout': 0.2,
            'sequence_feature_dim': 7,
            'scalar_features': [ # 与CycleNet训练时使用的特征完全一致
                '恒压充电时间(s)',
                '3.3~3.6V充电时间(s)',
            ]
        },
        'exp_net_params': {
            'n_terms': 4,
        },
        # 1.6 其他设置
        'batch_size': 256,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }

    # ### 2. 初始化 ###
    print(f"使用设备: {config['device']}")
    device = torch.device(config['device'])

    # 初始化CycleNet模型
    cn_params = config['cycle_net_params']
    cycle_model = CycleNetForSOH(
        d_model=cn_params['d_model'],
        sequence_feature_dim=cn_params['sequence_feature_dim'],
        scalar_feature_dim=len(cn_params['scalar_features']),
        d_ff=cn_params['d_ff'],
        meta_cycle_len=cn_params['meta_cycle_len'],
        dropout=cn_params['dropout']
    ).to(device)

    # 初始化ExpNet模型
    exp_params = config['exp_net_params']
    soh_model = ExpNet(n_terms=exp_params['n_terms']).to(device)

    # 加载预训练权重
    print("正在加载预训练模型...")
    try:
        cycle_model.load_state_dict(torch.load(config['model_paths']['cycle_net'], map_location=device))
        soh_model.load_state_dict(torch.load(config['model_paths']['exp_net'], map_location=device))
    except FileNotFoundError as e:
        print(f"错误: 模型文件未找到，请检查 'model_paths' 中的路径配置。详细信息: {e}")
        return
    cycle_model.eval() # 设置为评估模式
    soh_model.eval()   # 设置为评估模式
    print("模型加载成功。")

    # 加载预训练的Scaler
    print("正在加载Scaler...")
    try:
        scalers = joblib.load(config['model_paths']['scalers'])
    except FileNotFoundError as e:
        print(f"错误: acler文件 '{config['model_paths']['scalers']}' 未找到。该文件对于数据预处理至关重要。")
        return
    print("Scaler加载成功。")


    # ### 3. 数据准备 ###
    feature_info = {
        'sequence_dim': cn_params['sequence_feature_dim'],
        'sequence_length': 1, # 来自您的CycleNet脚本设置
        'scalar_features': cn_params['scalar_features']
    }
    prediction_df, scalar_cols = load_and_preprocess_for_prediction(
        config['prediction_batteries'],
        config['data_paths'],
        feature_info,
        scalers
    )
    pred_dataset = PredictionDataset(prediction_df, 'voltage_sequence', scalar_cols)
    pred_loader = DataLoader(pred_dataset, batch_size=config['batch_size'], shuffle=False)

    # ### 4. 执行推理 ###
    print("\n开始执行推理流程...")
    all_results = []
    # 在no_grad模式下运行，以提高效率并节省内存
    with torch.no_grad():
        for x_seq, x_scalar, true_cycle, battery_id, true_soh in pred_loader:
            x_seq = x_seq.to(device)
            x_scalar = x_scalar.to(device)
            true_cycle = true_cycle.to(device)

            # --- 步骤 1: 使用CycleNet预测循环号 ---
            # 注意: CycleNet的结构需要一个cycle_number输入来帮助模型去趋势化。
            # 这里我们传入真实的循环号，这与原始评估脚本中的做法一致，可以帮助模型做出更准确的预测。
            predicted_cycle_num = cycle_model(x_seq, x_scalar, true_cycle)

            # --- 步骤 2: 使用ExpNet预测SOH ---
            # 将上一步的输出作为这一步的输入
            predicted_soh = soh_model(predicted_cycle_num.squeeze(-1))

            # 收集当前批次的结果
            results_batch = {
                'battery_id': battery_id.cpu().numpy(),
                'true_cycle_number': true_cycle.cpu().numpy(),
                'true_soh': true_soh.cpu().numpy(),
                'predicted_cycle_number': predicted_cycle_num.cpu().numpy().flatten(),
                'predicted_soh': predicted_soh.cpu().numpy().flatten()
            }
            all_results.append(pd.DataFrame(results_batch))

    # ### 5. 保存结果 ###
    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
        # 确保输出目录存在
        output_dir = os.path.dirname(config['output_path'])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        final_results_df.to_csv(config['output_path'], index=False, encoding='utf-8-sig')
        print(f"\n推理完成。预测结果已保存至: {config['output_path']}")
        print("\n最终结果预览:")
        print(final_results_df.head())
    else:
        print("\n未能生成任何预测结果，请检查输入数据和配置。")

    # ### 6. (新增) 绘图 ###
    print("\n正在生成预测结果图...")

    # 准备用于绘图的数据
    true_cycles = final_results_df['true_cycle_number'].values
    pred_cycles = final_results_df['predicted_cycle_number'].values

    # 筛选出有真实SOH值的数据点进行绘图
    soh_plot_df = final_results_df[final_results_df['true_soh'] != -1]
    if not soh_plot_df.empty:
        true_soh = soh_plot_df['true_soh'].values
        pred_soh = soh_plot_df['predicted_soh'].values
    else:
        true_soh, pred_soh = None, None  # 如果没有真实SOH，则不绘制相关图形

    # 绘制循环号预测图
    plot_results(true_cycles, pred_cycles,
                 'cycle number prediction',
                 config['plot_paths']['cycle_line_plot'],
                 y_label='cycle number')
    plot_diagonal_results(true_cycles, pred_cycles,
                          'cycle number prediction',
                          config['plot_paths']['cycle_diagonal_plot'],
                          xy_label='cycle number')

    print(f"循环号预测图已保存至: {config['plot_paths']['cycle_line_plot']} 及对应的对角图文件。")

    # 绘制SOH预测图
    if true_soh is not None:
        plot_results(true_soh, pred_soh,
                     'SOH prediction',
                     config['plot_paths']['soh_line_plot'],
                     y_label='SOH (capacity Ah)')
        plot_diagonal_results(true_soh, pred_soh,
                              'SOH prediction',
                              config['plot_paths']['soh_diagonal_plot'],
                              xy_label='SOH')
        print(f"SOH预测图已保存至: {config['plot_paths']['soh_line_plot']} 及对应的对角图文件。")
    else:
        print("警告：未找到有效的真实SOH值，跳过SOH绘图。")

    # ### 5.5 (新增) 最终结果评估 ###
    print("\n正在对最终SOH预测结果进行评估...")

    # 筛选出拥有真实SOH值的数据点用于评估
    evaluation_df = final_results_df[final_results_df['true_soh'] != -1].copy()

    if not evaluation_df.empty:
        true_values = evaluation_df['true_soh'].values
        pred_values = evaluation_df['predicted_soh'].values

        # 计算各项评估指标
        mae = mean_absolute_error(true_values, pred_values)
        mse = mean_squared_error(true_values, pred_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, pred_values)

        print("--- SOH 预测结果评估 ---")
        print(f"  - MAE (平均绝对误差):  {mae:.6f}")
        print(f"  - MSE (均方误差):     {mse:.6f}")
        print(f"  - RMSE (均方根误差):  {rmse:.6f}")
        print(f"  - R2 Score (决定系数): {r2:.6f}")

        # 将指标保存到CSV文件
        metrics_data = {
            'Metric': ['MAE', 'MSE', 'RMSE', 'R2_Score'],
            'Value': [mae, mse, rmse, r2]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(config['metrics_path'], index=False, encoding='utf-8-sig')
        print(f"\n评估指标已成功保存至: {config['metrics_path']}")

    else:
        print("警告: 未找到有效的真实SOH值，跳过最终结果评估环节。")


if __name__ == '__main__':
    main()