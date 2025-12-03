import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -------- 基础样式（期刊友好）--------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ===== 1) 配置输入与输出 =====
# csvs 字典保持不变，它的键将用于文件名
csvs = {
    "CCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\4\validation_results.csv",
    "CCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\12\validation_results.csv",
    "VCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\14\validation_results.csv",
    "VCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\20\validation_results.csv",
}

# 新建一个 titles 字典，用于图表标题
titles = {
    "CCC-CCD": "Battery4",
    "CCC-VCD": "Battery12",
    "VCC-CCD": "Battery14",
    "VCC-VCD": "Battery20",
}

out_dir = r"D:\任务归档\电池\研究\小论文1号\DOCUMENT\fig\Results\ADC_estimation"
os.makedirs(out_dir, exist_ok=True)

# 颜色（自定）
c_real = "royalblue"  # Real SOH
# palette 字典也保持不变
palette = {
    "CCC-CCD": "#b08bd3",
    "CCC-VCD": "#58c5c7",
    "VCC-CCD": "#d64b8c",
    "VCC-VCD": "#f07c54",
}

def read_pair(path):
    df = pd.read_csv(path)
    # 改成你的列名：cycle / real_soh / est_soh
    x = df["Cycle"].to_numpy()
    real = df["True_Value"].to_numpy()
    est  = df["Predicted_Value"].to_numpy()
    # 若是0~1小数，转百分比
    if real.max() <= 1.5 and est.max() <= 1.5:
        real, est = real*100.0, est*100.0
    return x, real, est

# ***** 函数定义修改处 *****
# 将原来的 name 参数拆分为 title 和 savename
def save_one_small_fig(title, savename, x, real, est, color,
                       figsize=(3.4, 3.2), dpi=600,
                       xticks=None, err_ylim="auto"):
    err = est - real
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], figure=fig)

    # 上：SOH
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, real, ls='none', marker='o', ms=2, color=c_real, label='Real Value (Ah)', alpha=0.95)
    ax1.plot(x, est,  ls='none', marker='D', ms=0.7, color=color,  label='Estimated Value (Ah)', alpha=0.95)
    ax1.set_ylabel("Accumulated Discharge Capacity", fontsize=11)
    # 使用 title 参数设置标题
    ax1.set_title(title, fontsize=13, pad=2)
    ax1.tick_params(labelsize=10)
    # ax1.grid(True, ls='--', lw=0.5, alpha=0.25)
    # ax1.legend(loc="lower right", fontsize=10, frameon=False, handletextpad=0.4, borderpad=0.2)

    # 下：误差
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.axhline(0.0, color='k', lw=0.8, alpha=0.7)
    ax2.plot(x, err, ls='none', marker='^', ms=1, color=color, alpha=0.95, label='Error (Ah)')
    ax2.set_xlabel("Cycles", fontsize=11)
    ax2.set_ylabel("Error", fontsize=11)
    ax2.tick_params(labelsize=10)
    # ax2.grid(True, ls='--', lw=0.5, alpha=0.25)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # 2. 合并它们
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    # 3. 在 ax1 上创建统一的图例
    ax1.legend(all_handles, all_labels, loc="lower right", fontsize=10,
               frameon=False, handletextpad=0.4, borderpad=0.2)

    # x 轴刻度（可固定）
    if xticks is not None:
        ax2.set_xticks(xticks)

    # 误差轴范围（均值±3σ，或自定义）
    if err_ylim == "auto":
        mu, sigma = float(np.mean(err)), float(np.std(err))
        ax2.set_ylim(mu - 3.0*sigma, mu + 3.0*sigma)
    elif isinstance(err_ylim, (tuple, list)) and len(err_ylim) == 2:
        ax2.set_ylim(err_ylim)

    # 保存
    # 使用 savename 参数设置文件名
    base = os.path.join(out_dir, savename.replace(" ", "_"))
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ===== 2) 逐个生成四张小图 =====
# ***** 循环调用修改处 *****
for name, path in csvs.items():
    x, real, est = read_pair(path)
    # name ("CCC-CCD"等) 用于获取颜色和作为文件名
    # titles[name] ("Battery4"等) 用于作为图表标题
    save_one_small_fig(title=titles[name], savename=name,
                       x=x, real=real, est=est, color=palette.get(name, "#1f77b4"),
                       # 如果需要固定x刻度，取消下一行注释：
                       # xticks=[0, 50, 100, 150, 200, 250, 300, 350, 400],
                       err_ylim="auto")