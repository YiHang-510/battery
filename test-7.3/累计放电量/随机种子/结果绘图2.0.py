import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

# -------- 基础样式（期刊友好）--------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# ===== 1) 配置输入与输出 =====

# 包含 x 轴数据 "True_Value" 的文件路径字典
csvs_adc = {
    "CCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\4\validation_results.csv",
    "CCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\12\validation_results.csv",
    "VCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\14\validation_results.csv",
    "VCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\TM_PIRes\20\validation_results.csv",
}

# 包含 y 轴数据 "真实SOH" 和 "预测SOH" 的文件路径字典
csvs = {
    "CCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\combine_TMPIRes_ExpNetTR_final\4\battery_4_fusion_prediction.csv",
    "CCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\combine_TMPIRes_ExpNetTR_final\12\battery_12_fusion_prediction.csv",
    "VCC-CCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\combine_TMPIRes_ExpNetTR_final\14\battery_14_fusion_prediction.csv",
    "VCC-VCD": r"D:\任务归档\电池\研究\小论文1号\RESULT\自己的实验\combine_TMPIRes_ExpNetTR_final\20\battery_20_fusion_prediction.csv",
}

# 图表标题字典
titles = {
    "CCC-CCD": "Battery4",
    "CCC-VCD": "Battery12",
    "VCC-CCD": "Battery14",
    "VCC-VCD": "Battery20",
}

out_dir = r"D:\任务归档\电池\研究\小论文1号\DOCUMENT\fig\Results\SOH_estimation"
os.makedirs(out_dir, exist_ok=True)

# 颜色
c_real = "royalblue"
palette = {
    "CCC-CCD": "#b08bd3",
    "CCC-VCD": "#58c5c7",
    "VCC-CCD": "#d64b8c",
    "VCC-VCD": "#f07c54",
}


# ===== 修改点 1: 函数定义和内部逻辑 =====
# 函数的参数从 path 改为 name，这样它就可以访问两个字典
def read_pair(name):
    # 根据 name 从 csvs 字典获取主数据文件路径
    path_main = csvs[name]
    # 根据 name 从 csvs_adc 字典获取x轴数据文件路径
    path_adc = csvs_adc[name]

    # 读取包含 y 轴数据的文件
    df = pd.read_csv(path_main)
    # 读取包含 x 轴数据的文件
    df1 = pd.read_csv(path_adc)

    # 从 df1 中提取 x 轴数据
    x = df1["True_Value"].to_numpy()
    # 从 df 中提取 y 轴数据
    real = df["真实SOH"].to_numpy()
    est = df["预测SOH"].to_numpy()

    # 若是0~1小数，转百分比
    # if real.max() <= 1.5 and est.max() <= 1.5:
    #     real, est = real * 100.0, est * 100.0
    return x, real, est


# 绘图函数保持不变
def save_one_small_fig(title, savename, x, real, est, color,
                       figsize=(3.95, 3.2), dpi=600,
                       xticks=None, err_ylim="auto"):
    err = est - real
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], figure=fig)

    # 上：SOH
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, real, ls='none', marker='o', ms=1, color=c_real, label='Real SOH', alpha=0.95)
    ax1.plot(x, est, ls='none', marker='D', ms=1, color=color, label='Estimated SOH', alpha=0.95)
    ax1.set_ylabel("SOH", fontsize=10)
    ax1.set_title(title, fontsize=10, pad=2)
    ax1.tick_params(labelsize=10)
    ax1.set_ylim(0.7, 1.0)

    ax1.yaxis.set_major_locator(MultipleLocator(0.05))

    # 下：误差
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.axhline(0.0, color='k', lw=0.8, alpha=0.7)
    ax2.plot(x, err, ls='none', marker='^', ms=1, color=color, alpha=0.95, label='Error')
    ax2.set_xlabel("Accumulated Discharge Capacity (Ah)", fontsize=10)
    ax2.set_ylabel("Error", fontsize=10)
    ax2.tick_params(labelsize=10)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax1.legend(all_handles, all_labels, loc="upper right", fontsize=10,
               frameon=False, handletextpad=0.4, borderpad=0.2)

    if xticks is not None:
        ax2.set_xticks(xticks)

    if err_ylim == "auto":
        mu, sigma = float(np.mean(err)), float(np.std(err))
        ax2.set_ylim(mu - 3.0 * sigma, mu + 3.0 * sigma)
    elif isinstance(err_ylim, (tuple, list)) and len(err_ylim) == 2:
        ax2.set_ylim(err_ylim)

    base = os.path.join(out_dir, savename.replace(" ", "_"))
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ===== 2) 逐个生成四张小图 =====
# 移除文件末尾多余的 '}'
for name, path in csvs.items():
    # ===== 修改点 2: 函数调用 =====
    # 将 path 改为 name 传入函数
    x, real, est = read_pair(name)

    save_one_small_fig(title=titles[name], savename=name,
                       x=x, real=real, est=est, color=palette.get(name, "#1f77b4"),
                       err_ylim="auto")

print("所有图像已成功生成并保存。")