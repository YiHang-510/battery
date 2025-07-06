# imv_lstm_net.py

import torch
from Model import IMVTensorLSTM  # 假设你模型类还在 Model.py
import numpy as np

class IMVLSTMNet:
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, model_path=None):
        self.model = IMVTensorLSTM(input_dim, hidden_dim, num_layers, output_dim)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, x):
        # x: numpy array, shape [batch, seq_len, input_dim]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_pred = self.model(x_tensor)
        return y_pred.squeeze(-1).cpu().numpy()