
# main.py

from imv_lstm_net import IMVLSTMNet
from exp_net import ExpNetPredictor
from kalman_fusion import KalmanFusion
import numpy as np
import random
import torch
import matplotlib
import matplotlib.pyplot as plt

save_path = r'E:\code\Battery\result'
data_dir = r'E:\code\Battery\data\battery_features_norm_10_1525.csv'
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


# 数据准备代码略...
# X_input: [batch, seq_len, input_dim]  # LSTM用
# c_input: [batch,]  # 经验网络用，通常是循环编号或与capacity拟合最强的变量
# Y_true: [batch,]   # 真实容量

imv_lstm = IMVLSTMNet(input_dim=6, hidden_dim=64, num_layers=2, output_dim=1, model_path="your_lstm.pth")
exp_net = ExpNetPredictor(model_path="your_expnet.pth", n_terms=16)
kf = KalmanFusion(Q=1e-5, R=1e-2)

# 分别预测
y_pred_lstm = imv_lstm.predict(X_input)      # [batch,]
y_pred_exp = exp_net.predict(c_input)        # [batch,]

# 融合预测
final_preds = []
for i in range(len(y_pred_lstm)):
    fused = kf.update([y_pred_lstm[i], y_pred_exp[i]])
    final_preds.append(fused)
final_preds = np.array(final_preds)

print("Final fusion prediction:", final_preds)
