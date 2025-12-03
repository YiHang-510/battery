import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # 确保文件顶部有这个

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓多模态LSTM网络↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

class IMVTensorLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim

    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                      torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t * f_tilda_t + i_tilda_t * j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t * torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas / torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas * mu, dim=1)

        return mean, alphas, betas


class IMVFullLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]

    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.W_i = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.W_f = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.W_o = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim

    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim * self.n_units).cuda()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            inp = torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t * f_t + i_t * j_tilda_t.contiguous().view(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t * torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas / torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas * mu, dim=1)
        return mean, alphas, betas

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑多模态LSTM网络↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓经验指数网络↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

class ExpNet(nn.Module):
    def __init__(self, n_terms=16):
        super(ExpNet, self).__init__()
        # 每组都有a, b, d三个参数，共n_terms组
        # 乘一个小负数防止梯度爆炸
        # self.b = nn.Parameter(torch.ones(n_terms) * -0.01)
        # self.a = nn.Parameter(torch.ones(n_terms) * 1.0)
        # self.d = nn.Parameter(torch.ones(n_terms))
        # 不再使用 torch.ones，而是使用 torch.rand 来创建随机的初始值。
        # torch.rand() 会读取您 set_seed() 设置的种子，从而实现随机初始化。

        # 这会生成一个在 [-0.01, -0.001] 范围内的随机数
        # 它保证了 b 既是负数，又永远不会是 0。
        self.b = nn.Parameter(-torch.rand(n_terms) * 0.009 - 0.001)

        # 参数 a: 用0到1之间的随机数替代固定的 1.0
        self.a = nn.Parameter(torch.rand(n_terms))

        # 参数 d: 用0到1之间的随机数替代固定的 1.0
        self.d = nn.Parameter(torch.rand(n_terms))
        self.n_terms = n_terms

    def forward(self, c):
        # c: [batch_size,] 或 [batch_size, 1]
        c = c.view(-1, 1)      # [batch_size, 1]
        a = self.a.view(1, -1) # [1, n_terms]
        b = self.b.view(1, -1)
        d = self.d.view(1, -1)
        # 广播，计算每组参数的输出
        out = a * torch.exp(b * c) + d  # [batch_size, n_terms]
        # 你可以选择sum或mean，也可以直接输出所有组
        out = out.sum(dim=1)            # [batch_size]
        return out

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑经验指数网络↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


# #######卡尔曼滤波######
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



class ExpNetTR(nn.Module):
    """
    Trend (mixture of exponentials) + local Residual (Gaussian bumps).
    - 允许容量“再生”局部上升（由残差负责）
    - 趋势项稳、可解释；残差项可局部正/负，幅度有界，避免发散
    """
    def __init__(self, n_terms=16, n_bumps=8, use_logspace_tau=True):
        super().__init__()
        self.n_terms = n_terms
        self.n_bumps = n_bumps

        # ---- Trend：指数混合（不做单调硬约束，b自动学负值为主，也允许正值）
        # 权重 α -> softmax；衰减率 τ -> softplus；输入尺度 gamma -> softplus
        self.raw_alpha = nn.Parameter(0.01 * torch.randn(n_terms))
        if use_logspace_tau:
            # 覆盖更快到更慢（前段需要非常快的衰减项）
            max_log = 4.0  # 可试 4.5~5.0 -> tau_max ≈ e^4.5≈90 或 e^5≈148
            init_tau = torch.exp(torch.linspace(-2.5, max_log, steps=n_terms))  # ~[0.082, 90]
            self.raw_tau = nn.Parameter(torch.log(init_tau)+ 0.01 * torch.randn(n_terms))
        else:
            self.raw_tau = nn.Parameter(torch.randn(n_terms) * 0.1)
        self.raw_gamma = nn.Parameter(torch.tensor(0.0))        # 输入尺度

        # 趋势的上下界（可选）：不强制到 [0,1]，给线性输出更自由
        # 也可以改成 y = y_inf + (y0 - y_inf)*mix，提升可解释性
        self.trend_bias = nn.Parameter(torch.tensor(0.8) + 0.01 * torch.randn(()))       # 类似 y0
        self.trend_gain = nn.Parameter(torch.tensor(-0.5) + 0.01 * torch.randn(()))      # 类似 (y_inf - y0)，初始向下

        # ---- Residual：局部高斯凸起（允许正/负），专门刻画“再生/回落”
        # 中心 μ 放在 [0,1] 的等距初值；σ 用 softplus 保正；权重用 tanh 限幅更稳
        # mu = torch.linspace(0.05, 0.95, steps=n_bumps)          # 归一化 C 轴上的中心
        # self.mu = nn.Parameter(mu)                               # 可学习中心
        # self.raw_sigma = nn.Parameter(torch.full((n_bumps,), -1.0))  # softplus(-1)≈0.31

        # 残差：头/中/尾分配，更窄的头尾以刻画局部形状
        n_head = max(2, int(self.n_bumps * 0.35))  # ~35% 盯头部
        n_mid = max(1, int(self.n_bumps * 0.20))  # ~20% 过渡
        n_tail = self.n_bumps - n_head - n_mid

        # 头部用“对数间距”更密更靠近 0（0.001~0.15）
        mu_head = torch.exp(torch.linspace(math.log(1e-3), math.log(0.15), steps=n_head))
        # 中段均匀
        mu_mid = torch.linspace(0.15, 0.70, steps=n_mid)
        # 尾段更密，并允许略超 1 兜住边界效应
        mu_tail = torch.linspace(0.70, 1.02, steps=n_tail)

        self.mu = nn.Parameter(torch.cat([mu_head, mu_mid, mu_tail]))

        # 头部更窄，中段中等，尾部较窄
        self.raw_sigma = nn.Parameter(torch.cat([
            torch.full((n_head,), -2.3),  # σ≈softplus(-2.3)≈0.10
            torch.full((n_mid,), -1.3),  # σ≈0.27
            torch.full((n_tail,), -2.0),  # σ≈0.13
        ]))

        self.raw_beta  = nn.Parameter(0.01 * torch.randn(n_bumps))
        self.raw_res_scale = nn.Parameter(torch.tensor(-2.0))    # 残差总幅度缩放

        # ---- 可选：学习一个输入平移（适配不同起点）
        self.input_shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, c, return_components=False):
        # c: [B] or [B,1]，建议外部把 C 归一化到 [0,1]；若未归一，也能靠 gamma 学到尺度
        c = c.view(-1, 1)
        c_ = c - self.input_shift

        # Trend
        alpha = F.softmax(self.raw_alpha, dim=0)                 # [K]
        tau = torch.exp(self.raw_tau)  # ≥0，数值更稳
        # 可选再限幅，防数值病态：
        tau = tau.clamp_max(80.0)
        gamma = F.softplus(self.raw_gamma) + 1e-6                # >=0
        # 指数基函数：exp(-tau * gamma * c_ )，允许 c_ < 0 时更灵活
        expo = torch.exp(- (c_ * gamma) @ tau.view(1, -1))       # [B,K]
        mix  = (expo * alpha.view(1, -1)).sum(dim=1, keepdim=True)  # [B,1] in (0,1]
        trend = self.trend_bias + self.trend_gain * mix          # [B,1]

        # Residual：Gaussian bumps（允许正负，幅度受 tanh + scale 控制）
        sigma = F.softplus(self.raw_sigma) + 1e-6                # [M] >0
        beta  = torch.tanh(self.raw_beta)                        # [-1,1]
        res_scale = torch.sigmoid(self.raw_res_scale)            # (0,1) 小幅度优先
        # [B,M]
        gauss = torch.exp(-0.5 * ((c_ - self.mu.view(1, -1)) / sigma.view(1, -1))**2)
        residual = res_scale * (gauss * beta.view(1, -1)).sum(dim=1, keepdim=True)  # [B,1]

        y = (trend + residual).view(-1)                          # 不强制到 [0,1]

        if not return_components:
            return y
        else:
            comps = {
                "alpha": alpha, "tau": tau, "gamma": gamma,
                "trend_bias": self.trend_bias, "trend_gain": self.trend_gain,
                "mu": self.mu, "sigma": sigma, "beta": beta, "res_scale": res_scale,
                "trend": trend.view(-1), "residual": residual.view(-1)
            }
            return y, comps



class ExpNetKnee(nn.Module):
    def __init__(self, n_terms=8, n_bumps=8):
        super().__init__()
        K = n_terms
        # 两套指数混合
        self.raw_alpha1 = nn.Parameter(torch.zeros(K))
        self.raw_tau1   = nn.Parameter(torch.linspace(-3., 1.5, K).exp().log())   # 反softplus近似
        self.raw_alpha2 = nn.Parameter(torch.zeros(K))
        self.raw_tau2   = nn.Parameter(torch.linspace(-1., 2.0, K).exp().log())
        self.raw_gamma1 = nn.Parameter(torch.tensor(0.0))
        self.raw_gamma2 = nn.Parameter(torch.tensor(0.0))

        # 门控（拐点）
        self.raw_c0 = nn.Parameter(torch.tensor(0.7))   # 归一化 C 轴上的拐点初值
        self.raw_k  = nn.Parameter(torch.tensor(5.0))   # 拐点锐度

        # y_inf 参数化： y = y_inf + (y0 - y_inf) * mix
        self.y0     = nn.Parameter(torch.tensor(1.0))
        self.y_inf  = nn.Parameter(torch.tensor(0.0))

        # 残差（见改进2 会把 μ 向尾段加密）
        M = n_bumps
        mu = torch.linspace(0.05, 0.95, steps=M) ** 2.0  # ← 尾段加密（γ=2）
        self.mu = nn.Parameter(mu)
        self.raw_sigma = nn.Parameter(torch.full((M,), -1.0))
        self.raw_beta  = nn.Parameter(torch.zeros(M))
        self.raw_res_scale = nn.Parameter(torch.tensor(-2.0))

        self.input_shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, c, return_components=False):
        c = c.view(-1,1)
        c_ = c - self.input_shift

        sp = F.softplus
        # pre-phase
        a1 = F.softmax(self.raw_alpha1, dim=0)
        t1 = sp(self.raw_tau1) + 1e-6
        g1 = sp(self.raw_gamma1) + 1e-6
        m1 = torch.exp(- (c_ * g1) @ t1.view(1,-1)) @ a1.view(-1,1)  # [B,1]

        # post-phase
        a2 = F.softmax(self.raw_alpha2, dim=0)
        t2 = sp(self.raw_tau2) + 1e-6
        g2 = sp(self.raw_gamma2) + 1e-6
        m2 = torch.exp(- (c_ * g2) @ t2.view(1,-1)) @ a2.view(-1,1)  # [B,1]

        # gate：后相权重 p_post
        c0 = torch.clamp(self.raw_c0, 0.0, 1.0)
        k  = sp(self.raw_k)
        p_post = torch.sigmoid(k*(c_ - c0))       # [B,1]
        mix = (1 - p_post) * m1 + p_post * m2     # [B,1]

        trend = self.y_inf + (self.y0 - self.y_inf) * mix

        # residual (RBF)，尾段加密后更能贴住EoL形状
        sigma = sp(self.raw_sigma) + 1e-6
        beta  = torch.tanh(self.raw_beta)
        res_scale = torch.sigmoid(self.raw_res_scale)
        gauss = torch.exp(-0.5 * ((c_ - self.mu.view(1,-1))/sigma.view(1,-1))**2)
        residual = res_scale * (gauss * beta.view(1,-1)).sum(dim=1, keepdim=True)

        y = (trend + residual).view(-1)
        if not return_components:
            return y
        return y, dict(trend=trend.view(-1), residual=residual.view(-1), p_post=p_post.view(-1))
