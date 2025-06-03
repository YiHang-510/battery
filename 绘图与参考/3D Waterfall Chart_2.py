import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

def line_3d(x, y, z, x_label_indexs):
    """
    在y轴的每个点，向x轴的方向延伸出一个折线面：展示每个变量的时序变化。
    x: x轴，时间维，右边。
    y: y轴，变量维，左边。
    z: z轴，数值维。二维矩阵，y列x行。每一行是对应变量的一个时间序列。
    x_label_indexs: 需要标注的时间点。
    """
    x_num = len(x)
    y_num = len(y)
    if z.shape[0] != y_num or z.shape[1] != x_num:
        return -1

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')

    # 标签列表
    labels = ['Real Data', 'GPR', 'LSTM', 'DDPG', 'GDDPG']

    # 绘制折线面，并调换颜色顺序
    for i in range(y_num):
        axs.plot(Y[i], X[i], z[i], color=plt.cm.viridis((y_num - 1 - i) / y_num),
                 linestyle='-', linewidth=1, marker='o', markersize=1, alpha=0.7, label=labels[i])  # 设置图例标签
        axs.plot(Y[i], X[i], np.zeros_like(z[i]), color='gray', alpha=0.5)

        polygon = [
            [Y[i, 0], X[i, 0], 0],
            [Y[i, -1], X[i, -1], 0],
        ]
        for j in range(x_num - 1, -1, -1):
            polygon.append([Y[i, j], X[i, j], z[i, j]])
        axs.add_collection3d(Poly3DCollection([polygon], color=plt.cm.viridis((y_num - 1 - i) / y_num), alpha=0.25))

        for k in x_label_indexs:
            axs.text(Y[i, k] - 0.05, X[i, k], z[i, k] + 0.02, f'{z[i, k]:.2f}',
                     color='black', ha='center', size=7)

    for k in x_label_indexs:
        axs.plot(Y[:, k], X[:, k], z[:, k], linestyle='--', linewidth=0.8, color='gray')

    axs.grid(False)

    axs.set_xticks([0, 1, 2, 3, 4])
    axs.set_xticklabels(['Real Data', 'GPR', 'LSTM', 'DDPG', 'GDDPG'])

    # —— 删除 y 轴刻度 ——
    axs.set_yticks([])

    # # —— 添加 y 轴坐标轴标签 ——
    # axs.set_ylabel('t')

    # —— 在这里添加视角调整 ——
    axs.view_init(elev=20, azim=-30)

    # 将图例往下移动：bbox_to_anchor 中第二个值越小，图例越往下
    axs.legend(loc='upper left',
               bbox_to_anchor=(0, 0.90),
               fontsize=10)
    plt.savefig('3D Waterfall Chart.svg', format='svg')
    plt.show()

if __name__ == '__main__':
    x = np.arange(1, 21, 2)  # 原始时间点
    y = np.arange(5)  # 变量维度
    z = np.array([
        [0.20, 0.21, 0.34, 0.43, 0.52, 0.68, 0.73, 0.81, 0.81, 0.83],
        [0.21, 0.34, 0.42, 0.48, 0.60, 0.79, 0.75, 0.81, 0.81, 0.83],
        [0.22, 0.38, 0.44, 0.50, 0.61, 0.67, 0.75, 0.77, 0.78, 0.80],
        [0.23, 0.37, 0.38, 0.48, 0.52, 0.60, 0.66, 0.62, 0.59, 0.68],
        [0.22, 0.19, 0.25, 0.36, 0.45, 0.50, 0.44, 0.48, 0.52, 0.53]
    ])  # 新的原始数据
    line_3d(x, y, z, [1, 5, 9])