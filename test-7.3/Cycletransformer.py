import torch
import torch.nn as nn
import math

class RecurrentCycle(torch.nn.Module):
    # Thanks for the contribution of wayhoww.
    # The new implementation uses index arithmetic with modulo to directly gather cyclic data in a single operation,
    # while the original implementation manually rolls and repeats the data through looping.
    # It achieves a significant speed improvement (2x ~ 3x acceleration).
    # See https://github.com/ACAT-SCUT/CycleNet/pull/4 for more details.
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

    def forward(self, index, length):
        gather_index = (index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)) % self.cycle_len
        return self.data[gather_index]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# RecurrentCycle 类的定义保持不变
# ...

class CycleTransformer(nn.Module):
    def __init__(self, configs):
        super(CycleTransformer, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        # 1. 保留 RecurrentCycle 模块
        self.cycleQueue = RecurrentCycle(cycle_len=self.cycle_len, channel_size=self.enc_in)

        # --- 从这里开始是主要的修改 ---

        # 2. 输入映射层：将输入的 channel_size 映射到 d_model
        self.input_projection = nn.Linear(self.enc_in, self.d_model)

        # 3. 添加位置编码
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=configs.dropout)

        # 4. 定义 Transformer Encoder 层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=configs.nhead,  # 新增超参数：注意力头数
            dim_feedforward=configs.d_ff,  # 新增超参数：前馈网络维度
            dropout=configs.dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=configs.e_layers  # 新增超参数：Encoder层数
        )

        # 5. 输出映射层：将 Transformer 的输出映射到最终预测的维度
        # 我们用一个简单的线性层将序列展平后进行预测
        self.output_projection = nn.Linear(self.d_model * self.seq_len, self.pred_len)

        # --- 修改结束 ---

    def forward(self, x, cycle_index):
        # x: (batch_size, seq_len, enc_in), cycle_index: (batch_size,)
        batch_size = x.shape[0]

        # Instance Normalization (RevIN)
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # 1. 剥离周期性成分
        x_cycle_removed = x - self.cycleQueue(cycle_index, self.seq_len)

        # --- Transformer 前向传播 ---

        # 2. 将输入映射到 d_model
        # (batch_size, seq_len, enc_in) -> (batch_size, seq_len, d_model)
        src = self.input_projection(x_cycle_removed)

        # 3. 调整维度以匹配 Transformer 输入 & 添加位置编码
        # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)

        # 4. 通过 Transformer Encoder
        # (seq_len, batch_size, d_model) -> (seq_len, batch_size, d_model)
        transformer_output = self.transformer_encoder(src)

        # 5. 恢复维度并映射到输出
        # (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        transformer_output = transformer_output.permute(1, 0, 2)

        # 将所有时间步的输出展平
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len * d_model)
        output_flat = transformer_output.reshape(batch_size, -1)

        # 预测非周期性部分
        # (batch_size, seq_len * d_model) -> (batch_size, pred_len)
        y_residual = self.output_projection(output_flat)

        # 将输出调整为 (batch_size, pred_len, channel_size)
        # 注意：这里假设我们为所有 channel 预测相同的值，
        # 如果要为每个 channel 单独预测，需要调整输出层
        y_residual = y_residual.unsqueeze(-1).repeat(1, 1, self.enc_in)

        # --- 结束 ---

        # 6. 加回周期性成分
        y = y_residual + self.cycleQueue((cycle_index + self.seq_len) % self.cycle_len, self.pred_len)

        # Instance Denormalization
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y