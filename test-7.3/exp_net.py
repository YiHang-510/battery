# exp_net.py

import torch
import torch.nn as nn
from Model import ExpNet

class ExpNetPredictor:
    """经验网络推理器，适配main.py的调用"""
    def __init__(self, model_path, n_terms=16, device='cpu'):
        self.model = ExpNet(n_terms=n_terms).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def predict(self, c_array):
        # c_array: numpy array, shape [batch,] 或 [batch, 1]
        c_tensor = torch.tensor(c_array, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(c_tensor)
        return y_pred.cpu().numpy()
