import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# -------- 基础样式（期刊友好）--------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ===== 1) 配置输入与输出 =====

# 指向您的 validation_results.csv 文件
csvs = {
    "CCC-CCD": r"D:\任务归档\电池\研究\[二稿]小论文1号\RESULT\自己的实验\TM_PIRes\cc\validation_results.csv",
    "CCC-VCD": r"D:\任务归档\电池\研究\[二稿]小论文1号\RESULT\自己的实验\TM_PIRes\cv\validation_results.csv",
    "VCC-CCD": r"D:\任务归档\电池\研究\[二稿]小论文1号\RESULT\自己的实验\TM_PIRes\vc\validation_results.csv",
    "VCC-VCD": r"D:\任务归档\电池\研究\[二稿]小论文1号\RESULT\自己的实验\TM_PIRes\vv\validation_results.csv",
}

# 图表标题字典
titles = {
    "CCC-CCD": "",
    "CCC-VCD": "",
    "VCC-CCD": "",
    "VCC-VCD": "",
}

# [修改] 输出目录建议改为 ADC_estimation
out_dir = r"D:\任务归档\电池\研究\[二稿]小论文1号\DOCUMENT\fig\Results\ADC_estimation"
os.makedirs(out_dir, exist_ok=True)

# 颜色配置
c_real = "royalblue"
palette = {
    "CCC-CCD": "#b08bd3",
    "CCC-VCD": "#58c5c7",
    "VCC-CCD": "#d64b8c",
    "VCC-VCD": "#f07c54",
}


# ===== 2) 修改后的数据读取函数 (读取 ADC 数据) =====
def read_pair(name):
    path = csvs[name]
    df = pd.read_csv(path)

    # X轴：循环圈数
    x = df["cycle"].to_numpy()

    # [核心修改] Y轴：改为读取累计放电容量 (ADC)
    # true_q: 真实 ADC, pred_q: 预测 ADC
    real = df["true_q"].to_numpy()
    est = df["pred_q"].to_numpy()

    return x, real, est


# ===== 3) 绘图函数 (ADC 版本) =====
def save_one_small_fig(title, savename, x, real, est, color,
                       figsize=(3.95, 5.5), dpi=600):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1, 1.1], figure=fig)

    # --- 自动计算坐标轴范围 (因为ADC数值范围大，不能写死) ---
    data_min = min(real.min(), est.min())
    data_max = max(real.max(), est.max())
    margin = (data_max - data_min) * 0.05  # 留5%边距
    limit_min = data_min - margin
    limit_max = data_max + margin

    # ----------------------
    # 上图：ADC vs Cycle
    # ----------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, real, ls='none', marker='o', ms=1, color=c_real, label='Real ADC', alpha=0.8)
    ax1.plot(x, est, ls='none', marker='D', ms=1, color=color, label='Estimated ADC', alpha=0.8)

    # [修改标签]
    ax1.set_xlabel("Cycle", fontsize=10)
    ax1.set_ylabel("ADC (Ah)", fontsize=10)  # 或者是 Accumulated Capacity (Ah)
    ax1.set_title(title, fontsize=11, pad=4)
    ax1.tick_params(labelsize=10)

    # 设置 Y 轴范围
    ax1.set_ylim(limit_min, limit_max)
    # ax1.yaxis.set_major_locator(MaxNLocator(nbins=5)) # 自动调整刻度数量

    ax1.legend(loc="upper left", fontsize=9, frameon=True, handletextpad=0.2, borderpad=0.3)
    ax1.grid(True, ls='--', lw=0.5, alpha=0.5)

    # ----------------------
    # 下图：Predicted vs Real (对角图)
    # ----------------------
    ax2 = fig.add_subplot(gs[1, 0])

    # 1. 对角线
    ax2.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', lw=1, label='Perfect Prediction')

    # 2. 散点
    ax2.scatter(real, est, s=8, c=color, marker='o', alpha=0.6, edgecolors='none', label='Prediction')

    # 3. [修改标签]
    ax2.set_xlabel("Real ADC (Ah)", fontsize=10)
    ax2.set_ylabel("Estimated ADC (Ah)", fontsize=10)
    ax2.tick_params(labelsize=10)

    # 4. 范围与比例
    ax2.set_xlim(limit_min, limit_max)
    ax2.set_ylim(limit_min, limit_max)
    ax2.set_aspect('equal')
    ax2.grid(True, ls='--', lw=0.5, alpha=0.5)

    # RMSE
    rmse = np.sqrt(np.mean((real - est) ** 2))
    # RMSE 位置调整：放在左上角
    # ax2.text(0.05, 0.90, f"RMSE: {rmse:.2f}", transform=ax2.transAxes,
    #          fontsize=10, verticalalignment='top',
    #          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 保存
    base = os.path.join(out_dir, savename.replace(" ", "_") + "_ADC")
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ===== 4) 执行 =====
for name, path in csvs.items():
    print(f"正在处理: {name} ...")
    try:
        x, real, est = read_pair(name)
        save_one_small_fig(title=titles[name], savename=name,
                           x=x, real=real, est=est, color=palette.get(name, "#1f77b4"))
    except Exception as e:
        print(f"处理 {name} 时出错: {e}")

print("所有 ADC-Cycle 对角图已成功生成并保存。")