import numpy as np

class KalmanFusion:
    def __init__(self, Q=1e-5, R=1e-2):
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 观测噪声协方差
        self.x = None  # 融合状态
        self.P = 1  # 估计协方差

    def update(self, z):
        # z: list or np.array, 多个观测值
        if self.x is None:
            self.x = np.mean(z)
        # 预测
        self.P = self.P + self.Q
        # 融合观测
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (np.mean(z) - self.x)
        self.P = (1 - K) * self.P
        return self.x
