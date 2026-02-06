import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    "CCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\cc\validation_results.csv",
    "CCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\cv\validation_results.csv",
    "VCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\vc\validation_results.csv",
    "VCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\vv\validation_results.csv",
}

# 图表标题字典

titles = {
    "CCC-CCD": "",
    "CCC-VCD": "",
    "VCC-CCD": "",
    "VCC-VCD": "",
}

# 输出目录

out_dir = r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\Results\CDC_estimation_final"
os.makedirs(out_dir, exist_ok=True)

# 颜色配置
c_real = "royalblue"
palette = {
    "CCC-CCD": "#b08bd3",
    "CCC-VCD": "#58c5c7",
    "VCC-CCD": "#d64b8c",
    "VCC-VCD": "#f07c54",
}


# ===== 2) 数据读取函数 =====
def read_pair(name):
    path = csvs[name]
    df = pd.read_csv(path)

    x = df["true_q"].to_numpy()
    real = df["true_q"].to_numpy()
    est = df["pred_q"].to_numpy()
    true_soh = df["true_soh"].to_numpy()

    return x, real, est, true_soh


# ===== 3) 绘图函数 (布局调整版) =====
def save_one_small_fig(title, savename, x, real, est, true_soh, color,
                       figsize=(4, 3), dpi=600):
    fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=True)

    # 仅在 true_soh < 0.8 后开始绘制，且去掉最后一个点
    mask = true_soh < 0.8
    if mask.any():
        x = x[mask]
        real = real[mask]
        est = est[mask]
        true_soh = true_soh[mask]
    if len(x) > 0:
        x = x[:-1]
        real = real[:-1]
        est = est[:-1]
        true_soh = true_soh[:-1]

    # --- 自动计算范围 ---
    data_min_y = min(real.min(), est.min())
    data_max_y = max(real.max(), est.max())
    margin_y = (data_max_y - data_min_y) * 0.05
    limit_min_y = data_min_y - margin_y
    limit_max_y = data_max_y + margin_y

    # --- 主图绘制 ---
    ax1.plot(x, real, ls='none', marker='o', ms=2, color=c_real, label='Real ADC', alpha=1)
    ax1.plot(x, est, ls='none', marker='D', ms=2, color=color, label='Estimated ADC', alpha=1)

    ax1.set_xlabel("True Cumulative Discharge Capacity (Ah)", fontsize=16)
    ax1.set_ylabel("Predicted CDC (Ah)", fontsize=16)
    ax1.set_title(title, fontsize=17, pad=4)
    ax1.tick_params(labelsize=16)

    ax1.set_ylim(limit_min_y, limit_max_y)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.grid(True, ls='--', lw=0.5, alpha=0.5)

    # [修改 1] 图例位置改为右下角
    ax1.legend(loc="lower right", fontsize=12, frameon=True, handletextpad=0.2, borderpad=0.3)

    # ----------------------
    # 内嵌图 (Inset Plot)
    # ----------------------

    # [修改 2] 内嵌图位置改为左上角
    # 坐标格式: [left, bottom, width, height] (相对于主图的比例 0-1)
    # 左上角位置：left 约 0.18, bottom 约 0.55 (视具体宽高而定)
    ax_inset = fig.add_axes([0.11, 0.55, 0.42, 0.42])

    # 内嵌图范围
    data_min_inset = min(real.min(), est.min())
    data_max_inset = max(real.max(), est.max())
    plot_min_inset = data_min_inset - (data_max_inset - data_min_inset) * 0.05
    plot_max_inset = data_max_inset + (data_max_inset - data_min_inset) * 0.05

    # 1. 对角线
    ax_inset.plot([plot_min_inset, plot_max_inset], [plot_min_inset, plot_max_inset], 'k--', lw=0.8)

    # 2. 散点
    ax_inset.scatter(real, est, s=5, c=color, marker='o', alpha=0.7, edgecolors='none')

    # 3. 标签
    # ax_inset.set_xlabel("Real ADC", fontsize=8)
    # ax_inset.set_ylabel("Est. ADC", fontsize=8)
    ax_inset.tick_params(labelsize=7)

    ax_inset.yaxis.tick_right()  # 将刻度移动到右侧
    ax_inset.yaxis.set_label_position("right")  # 将标签移动到右侧

    # 4. 范围与比例
    ax_inset.set_xlim(plot_min_inset, plot_max_inset)
    ax_inset.set_ylim(plot_min_inset, plot_max_inset)
    ax_inset.set_aspect('equal')
    ax_inset.grid(True, ls=':', lw=0.3, alpha=0.6)

    # RMSE 文本 (在内嵌图左上角)
    # rmse = np.sqrt(np.mean((real - est) ** 2))
    # ax_inset.text(0.05, 0.95, f"RMSE: {rmse:.2f}", transform=ax_inset.transAxes,
    #               fontsize=7, verticalalignment='top',
    #               bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.7))

    # 保存
    base = os.path.join(out_dir, savename)
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ===== 4) 执行 =====
for name, path in csvs.items():
    print(f"正在处理: {name} ...")
    try:
        x, real, est, true_soh = read_pair(name)
        save_one_small_fig(title=titles[name], savename=name,
                           x=x, real=real, est=est, true_soh=true_soh,
                           color=palette.get(name, "#1f77b4"))
    except Exception as e:
        print(f"处理 {name} 时出错: {e}")

print("所有图表已更新：对角图在左上，图例在右下。")
