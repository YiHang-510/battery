import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os


'''
步骤 1: 生成两个模型的预测结果
分别运行exp-net-predict.py和cycle-net-predict.py在相同的测试集（例如，电池6）上进行预测，并保存结果
确保每个测试电池都生成了类似这样的CSV文件：
battery6_expnet_preds.csv (包含列: cycle, true_soh, pred_soh)
battery6_cyclenet_preds.csv (包含列: cycle, true_soh, pred_soh)
'''


class KalmanFilter1D:
    """
    一个简单的一维卡尔曼滤波器。
    """

    def __init__(self, x0, P0, R, Q):
        """
        初始化滤波器。
        :param x0: 初始状态估计 (initial state estimate)
        :param P0: 初始误差协方差 (initial error covariance)
        :param R: 测量噪声协方差 (measurement noise covariance)
        :param Q: 过程噪声协方差 (process noise covariance)
        """
        self.x = x0  # 状态估计
        self.P = P0  # 误差协方差
        self.R = R  # 测量噪声
        self.Q = Q  # 过程噪声

    def predict(self, u=0):
        """
        预测步骤。
        这里我们使用一个简化的过程模型: x_k = x_{k-1} + u
        u 是来自ExpNet的外部控制输入，代表SOH的预期变化。
        """
        # 状态预测
        self.x = self.x + u
        # 误差协方差预测
        self.P = self.P + self.Q
        return self.x

    def update(self, z):
        """
        更新步骤。
        :param z: 当前的测量值 (来自CycleNet的SOH)
        """
        # 卡尔曼增益 (Kalman Gain)
        K = self.P / (self.P + self.R)

        # 更新状态估计
        self.x = self.x + K * (z - self.x)

        # 更新误差协方差
        self.P = (1 - K) * self.P
        return self.x


def plot_fusion_results(df, cell_name, save_dir):
    """可视化融合结果"""
    plt.figure(figsize=(14, 8))
    plt.plot(df['cycle'], df['true_soh'], 'k-', label='True SOH', linewidth=2)
    plt.plot(df['cycle'], df['expnet_soh'], 'b--', label='ExpNet SOH (Guideline)', alpha=0.7)
    plt.plot(df['cycle'], df['cyclenet_soh'], 'g.', label='CycleNet SOH (Measurement)', alpha=0.5)
    plt.plot(df['cycle'], df['fused_soh'], 'r-', label='Fused SOH (Kalman Filter)', linewidth=2.5)

    plt.title(f'SOH Prediction Fusion for {cell_name}', fontsize=16)
    plt.xlabel('Cycle', fontsize=12)
    plt.ylabel('SOH', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{cell_name}_fusion_result.png'), dpi=300)
    plt.close()


def calculate_metrics(df):
    """计算并打印所有模型的性能指标"""
    y_true = df['true_soh']

    models = ['expnet_soh', 'cyclenet_soh', 'fused_soh']
    metrics_summary = {}

    for model_name in models:
        y_pred = df[model_name]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        metrics_summary[model_name] = {'RMSE': rmse, 'R2': r2}

    print("--- Performance Metrics ---")
    for model_name, metrics in metrics_summary.items():
        print(f"{model_name:>15}: RMSE = {metrics['RMSE']:.6f}, R2 = {metrics['R2']:.6f}")
    print("--------------------------")
    return metrics_summary


if __name__ == '__main__':
    # --- 1. 定义文件路径和参数 ---
    # !! 修改为你实际的预测文件路径 !!
    expnet_pred_path = '/home/scuee_user06/myh/电池/data/input_for_kalman/expnet/test_battery_6_predictions.csv'
    cyclenet_pred_path = '/home/scuee_user06/myh/电池/data/input_for_kalman/cyclenet/test_predictions.csv'
    cell_name = 'Battery_6'
    save_dir = '/home/scuee_user06/myh/电池/data/result_for_kalman'
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 加载数据 ---
    try:
        df_exp = pd.read_csv(expnet_pred_path, encoding='gbk')
        df_cycle = pd.read_csv(cyclenet_pred_path, encoding='gbk')
    except FileNotFoundError:
        print("错误：请确保预测文件路径正确。")
        exit()

    # --- 3. 合并数据 ---
    # 假设两个文件的cycle列和true_soh列是一致的
    df_merged = pd.DataFrame({
        'cycle': df_exp['循环号'],
        'true_soh': df_exp['soh'],
        'expnet_soh': df_exp['pred_soh'],
        'cyclenet_soh': df_cycle['Predicted_Capacity']
    }).sort_values(by='cycle').reset_index(drop=True)

    # --- 4. 初始化卡尔曼滤波器 ---
    # 初始状态：使用第一个循环的真实SOH（或CycleNet预测值）
    x0 = df_merged['true_soh'].iloc[0]
    # 初始不确定度：可以设为一个较小的值
    P0 = 1e-4

    '''
    核心调优：
    R（测量噪声）和Q（过程噪声）是您唯一需要调整的参数。
    增大 R / 减小 Q：意味着您更不信任CycleNet的测量，更相信ExpNet的平滑趋势。结果会更平滑，但可能无法及时跟随SOH的真实突变。
    减小 R / 增大 Q：意味着您更相信CycleNet的测量。结果会更贴近CycleNet的曲线，能更好地捕捉局部变化，但也可能引入更多噪声。
    调节 R/Q 的比值是调优的关键。可以从 R=0.01, Q=1e-5 开始，然后尝试将R增大或减小一个数量级，观察结果变化。
    '''

    # !! 核心调优参数 !!
    # R: 测量噪声。值越小，代表我们越信任CycleNet的瞬时测量。
    # 如果CycleNet的预测曲线抖动很厉害，可以适当增大R。
    R = 0.01
    # Q: 过程噪声。值越小，代表我们越信任ExpNet的平滑趋势。
    # 如果你认为SOH的实际衰退过程本身就有一些随机性，可以适当增大Q。
    Q = 1e-5

    kf = KalmanFilter1D(x0=x0, P0=P0, R=R, Q=Q)

    # --- 5. 循环执行融合 ---
    fused_soh_list = []

    # 第一个点的SOH直接使用初始值
    fused_soh_list.append(x0)

    for i in range(1, len(df_merged)):
        # 预测步骤的外部输入 u: ExpNet预测的SOH在两个循环间的变化量
        soh_change_exp = df_merged['expnet_soh'].iloc[i] - df_merged['expnet_soh'].iloc[i - 1]
        kf.predict(u=soh_change_exp)

        # 更新步骤的测量值 z: 当前循环下CycleNet的预测值
        measurement = df_merged['cyclenet_soh'].iloc[i]
        fused_soh = kf.update(z=measurement)

        fused_soh_list.append(fused_soh)

    df_merged['fused_soh'] = fused_soh_list

    # --- 6. 评估和可视化 ---
    plot_fusion_results(df_merged, cell_name, save_dir)
    metrics = calculate_metrics(df_merged)

    # 保存融合后的数据
    df_merged.to_csv(os.path.join(save_dir, f'{cell_name}_fusion_data.csv'), index=False, encoding='gbk')

    print(f"\n融合完成！结果已保存到 '{save_dir}' 文件夹。")