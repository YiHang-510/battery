import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

# --- 1. SCI 风格全局绘图设置 ---
sci_config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",  # 数学公式字体，与Times New Roman更搭配
    "font.size": 12,
    "axes.linewidth": 1.2,       # 坐标轴边框加粗
    "axes.labelsize": 16,        # 轴标签字号
    "xtick.labelsize": 12,       # 刻度字号
    "ytick.labelsize": 12,
    "xtick.direction": "in",     # 刻度线朝内
    "ytick.direction": "in",
    "xtick.top": True,           # 顶部开启刻度
    "ytick.right": True,         # 右侧开启刻度
    "xtick.major.width": 1.2,    # 主刻度线宽
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.8,    # 次刻度线宽
    "ytick.minor.width": 0.8,
}
mpl.rcParams.update(sci_config)

# --- 2. 数据读取与预处理 ---
try:
    i = 1  # 电池编号
    # 特征文件路径 (保持你的原始路径)
    capacity_file_name = f'D:\\任务归档\\电池\\研究\\data\\selected_feature\\statistic\\battery{i}_SOH健康特征提取结果.csv'
    df = pd.read_csv(capacity_file_name)

    # 验证必要的列
    required_columns = ['循环号', '最大容量(Ah)', '恒压充电时间(s)', '3.3~3.6V充电时间(s)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"错误：文件中缺少以下列: {missing_cols}")
        exit()

    # 排序并计算累计容量
    df = df.sort_values('循环号')
    if '累计放电容量(Ah)' not in df.columns:
        print("正在计算累计放电容量...")
        df['累计放电容量(Ah)'] = df['最大容量(Ah)'].cumsum()

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}")
    exit()
except Exception as e:
    print(f"发生错误: {e}")
    exit()

# 去除第一个点（不稳定数据）
plot_df = df.iloc[1:] 

# --- 3. 绘图核心部分 ---

# 定义 Nature 期刊风格配色
color_cv = '#00A087'    # 深墨绿
color_volt = '#3C5488'  # 深海蓝

# 创建画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=600, sharex=True)
plt.subplots_adjust(hspace=0.08) # 子图间距

x_data = plot_df['累计放电容量(Ah)']

# ==========================================
# 上图：恒压充电时间
# ==========================================
ax1.plot(x_data, plot_df['恒压充电时间(s)'], 
         color=color_cv, 
         linestyle='-',    # 使用实线代替 marker，更清晰
         linewidth=2.5,    # 线宽适中
         alpha=0.9,        # 轻微透明度增加质感
         zorder=10)        # 确保线在网格之上

ax1.set_ylabel('Time (s)', fontsize=17)

# 添加标签 (带白色背景框，防止遮挡)
# bbox参数让文字有个半透明白底，看起来非常高级
props = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='None')

ax1.text(0.97, 0.08, 'Constant Voltage Charging Time', transform=ax1.transAxes, 
         color=color_cv, fontsize=16, fontweight='bold',
         verticalalignment='bottom', horizontalalignment='right', 
         bbox=props, zorder=20)

# ==========================================
# 下图：3.3~3.6V充电时间
# ==========================================
ax2.plot(x_data, plot_df['3.3~3.6V充电时间(s)'], 
         color=color_volt, 
         linestyle='-', 
         linewidth=2.5, 
         alpha=0.9,
         zorder=10)

ax2.set_ylabel('Time (s)', fontsize=17)
ax2.set_xlabel('Cumulative Discharge Capacity (Ah)', fontsize=17)

# 添加标签 (带白色背景框)
ax2.text(0.97, 0.90, '3.3-3.6V Charging Time', transform=ax2.transAxes, 
         color=color_volt, fontsize=16, fontweight='bold',
         verticalalignment='top', horizontalalignment='right', 
         bbox=props, zorder=20)

# ==========================================
# 4. 公共美化细节 (关键步骤)
# ==========================================
for ax in [ax1, ax2]:
    # 浅灰色虚线网格，位于底层
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, color='#D3D3D3', alpha=0.6, zorder=0)
    
    # 开启次刻度 (Minor Ticks) - 增加精密感
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 调整刻度长度
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3, color='gray')

    # 设置x轴从0开始，避免线条贴边
    ax.set_xlim(left=0)

# --- 5. 保存图片 ---
# 建议保存为 _sci 后缀，以免覆盖原图
save_path_png = r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\scale_feature_track_sci.png"
save_path_pdf = r"D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\scale_feature_track_sci.pdf"

# bbox_inches="tight" 自动裁剪多余白边
fig.savefig(save_path_png, dpi=1200, bbox_inches="tight")
fig.savefig(save_path_pdf, bbox_inches="tight")

print(f"SCI风格绘图完成！已保存至: {save_path_png}")
plt.close()