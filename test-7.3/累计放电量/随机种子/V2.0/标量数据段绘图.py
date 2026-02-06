import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.text as mtext

# --- 设置全局字体为 Times New Roman ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'


# --- (字体设置结束) ---


# --- 辅助函数：create_continuous_series ---
# (保持不变)
def create_continuous_series(series):
    is_current_value_near_zero = (series.abs() < 1e-6)
    is_previous_value_not_near_zero = (series.shift(1).abs() >= 1e-6)
    reset_points = is_current_value_near_zero & is_previous_value_not_near_zero
    values_before_reset = series.shift(1)[reset_points]
    cumulative_offsets = values_before_reset.cumsum().fillna(0)
    offset_series = cumulative_offsets.reindex(series.index).ffill().fillna(0)
    return series + offset_series


# Output directory for all generated figures (relative to this script)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 1. 加载和准备数据 ---
# (保持您的路径)
for bat in range(1,2):
    csv_filename = f'D:\任务归档\电池\研究\data\selected_data\\battery{bat}\\battery{bat}_record.csv'

    try:
        df = pd.read_csv(csv_filename, encoding='gbk')
        df_cycle_100 = df[df['循环号'] == 100].copy()

        if df_cycle_100.empty:
            print("错误: 未找到 '循环号' 100 的数据。")
        else:
            print(f"成功加载并筛选了 {len(df_cycle_100)} 个数据点。")

            # --- 2. 创建 X 轴和连续容量数据 ---
            raw_x_data = np.arange(len(df_cycle_100))
            # (保持您的 /1000 缩放)
            x_data = raw_x_data / 1000

            df_cycle_100['充电容量(Ah)_continuous'] = create_continuous_series(df_cycle_100['充电容量(Ah)'])
            df_cycle_100['放电容量(Ah)_continuous'] = create_continuous_series(df_cycle_100['放电容量(Ah)'])

            # --- 3. 创建画布和四个坐标轴 ---
            fig, ax_current = plt.subplots(figsize=(8, 6))
            ax_voltage = ax_current.twinx()
            ax_charging = ax_current.twinx()
            ax_discharging = ax_current.twinx()

            # --- 4. 配置坐标轴位置 ---
            # (保持您的布局不变)
            ax_current.spines['left'].set_position(('axes', 0.0))
            ax_voltage.spines['left'].set_position(('axes', 0.0))
            ax_voltage.yaxis.tick_left()
            ax_voltage.yaxis.set_label_position("left")
            ax_voltage.spines['right'].set_color('none')

            # --- (保持Y轴标签被注释) ---
            # ax_current.set_ylabel('Y1: Current (A)', color='tab:blue', fontsize=20)
            # ax_voltage.set_ylabel('Y2: Voltage (V)', color='tab:red', fontsize=20)

            ax_current.tick_params(axis='y', colors='tab:blue', pad=8)
            ax_voltage.tick_params(axis='y', colors='tab:red', pad=-40)  # (保持您的 -40)
            ax_current.yaxis.set_label_coords(-0.14, 0.5)
            ax_voltage.yaxis.set_label_coords(-0.10, 0.5)
            ax_charging.yaxis.tick_right()
            ax_charging.yaxis.set_label_position("right")
            ax_discharging.yaxis.tick_right()
            ax_discharging.yaxis.set_label_position("right")
            ax_charging.spines['right'].set_position(('axes', 1.0))
            ax_discharging.spines['right'].set_position(('axes', 1.0))
            ax_charging.spines['left'].set_color('none')
            ax_discharging.spines['left'].set_color('none')

            # --- (保持Y轴标签被注释) ---
            # ax_charging.set_ylabel('Y3: Charging quantity (Ah)', color='tab:blue', fontsize=20)
            # ax_discharging.set_ylabel('Y4: Discharging quantity (Ah)', color='tab:green', fontsize=20)

            ax_charging.tick_params(axis='y', colors='tab:blue', pad=-35)
            ax_discharging.tick_params(axis='y', colors='tab:green', pad=8)
            ax_charging.yaxis.set_label_coords(1.10, 0.5)
            ax_discharging.yaxis.set_label_coords(1.14, 0.5)
            fig.subplots_adjust(left=0.20, right=0.80, bottom=0.13, top=0.85)
            for ax in [ax_current, ax_voltage, ax_charging, ax_discharging]:
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='x', labelsize=18, length=6, width=2)

            # (保持您的自定义刻度线参数)
            ax_current.tick_params(
                axis='y',
                which='both',
                direction='out',
                labelsize=20,
                length=6, width=2
            )
            ax_voltage.tick_params(
                axis='y',
                which='both',
                direction='in',
                labelsize=20,
                length=6, width=2
            )
            ax_charging.tick_params(
                axis='y',
                which='both',
                direction='in',
                labelsize=20,
                length=6, width=2
            )
            ax_discharging.tick_params(
                axis='y',
                which='both',
                direction='out',
                labelsize=20,
                length=6, width=2
            )

            # --- 5. 定义颜色和线型 ---
            # (保持不变)
            color_c, color_v, color_chg, color_dis = 'tab:blue', 'tab:red', 'tab:purple', 'tab:green'
            style_c, style_v, style_chg, style_dis = 'dotted', 'solid', 'dashed', 'solid'

            # --- 6. 绘图并分别设置 Y 轴 ---

            # (!!! 关键: 保持您原始的 ylim !!!)

            line_c = ax_current.plot(x_data, df_cycle_100['电流(A)'], color=color_c, linestyle=style_c, label='Current(A)',
                                     linewidth=3.5)
            ax_current.set_ylim(-5, 3)  # 保持不变
            ax_current.tick_params(axis='y', colors=color_c)

            line_v = ax_voltage.plot(x_data, df_cycle_100['电压(V)'], color=color_v, linestyle=style_v, label='Voltage(V)',
                                     linewidth=3)
            ax_voltage.set_ylim(2, 4.5)  # 保持不变
            ax_voltage.tick_params(axis='y', colors=color_v)

            line_chg = ax_charging.plot(x_data, df_cycle_100['充电容量(Ah)_continuous'], color=color_chg,
                                        linestyle=style_chg, label='Charging Capacity(Ah)', linewidth=3)
            ax_charging.set_ylim(0, 4)  # 保持不变
            ax_charging.tick_params(axis='y', colors=color_chg)

            line_dis = ax_discharging.plot(x_data, df_cycle_100['放电容量(Ah)_continuous'], color=color_dis,
                                           linestyle=style_dis, label='Discharging Capacity(Ah)', linewidth=3)
            ax_discharging.set_ylim(0, 4)  # 保持不变
            ax_discharging.tick_params(axis='y', colors=color_dis)

            # --- 7. 设置 X 轴和图例 ---

            # (保持您的 X 轴标签被注释)
            # ax_current.set_xlabel(r'X: Sampling steps ($\times 10^3$)', fontsize=20)

            ax_current.set_xlim(0, x_data.max())

            lines = line_c + line_v + line_chg + line_dis
            labels = [l.get_label() for l in lines]
            # ax_current.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)

            ax_current.set_xlim(0, x_data.max())

            # 收集图例信息
            lines = line_c + line_v + line_chg + line_dis
            labels = [l.get_label() for l in lines]

            # ==========================================
            # 步骤 A: 单独保存图例
            # ==========================================
            # 1. 确保主图里不画图例 (保持注释状态)
            MY_FONT_SIZE = 6.0      # 字体非常小
            MY_HANDLE_LENGTH = 4.5  # <--- 关键修改！字体越小，这个倍数要越大。
                                    # 原理：6pt * 4.5 = 27pt (宽度足够放下 "Y1")
                                    # 如果是 16pt字体，这个值只需要 1.5

            # 画布也必须非常小，否则图片会有巨大的白边
            FIG_WIDTH = 3.5         
            FIG_HEIGHT = 0.3        

            # ==========================================
            # 1. 自定义 Handler (保持居中逻辑)
            # ==========================================
            class CenteredTextHandler(HandlerBase):
                def __init__(self, text, color, fontsize):
                    self.text = text
                    self.color = color
                    self.fontsize = fontsize
                    super().__init__()

                def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                    # 强制使用我们指定的字号，不受外部干扰
                    t = mtext.Text(
                        x=width / 2, 
                        y=height / 2 - ydescent,
                        text=self.text,
                        color=self.color,
                        fontsize=self.fontsize, 
                        ha='center', va='center', 
                        fontfamily='Times New Roman'
                    )
                    return [t]

            # ==========================================
            # 2. 准备数据
            # ==========================================
            # 定义四种线条样式
            line_styles = [
                mlines.Line2D([], [], color='tab:blue', linestyle='dotted', linewidth=1.5),
                mlines.Line2D([], [], color='tab:red', linestyle='solid', linewidth=1.5),
                mlines.Line2D([], [], color='tab:purple', linestyle='dashed', linewidth=1.5),
                mlines.Line2D([], [], color='tab:green', linestyle='solid', linewidth=1.5)
            ]

            labels_text = ["Current (A)", "Voltage (V)", "Charging Capacity (Ah)", "Discharging Capacity (Ah)"]
            prefixes = ["Y1", "Y2", "Y3", "Y4"]
            colors = ['tab:blue', 'tab:red', 'tab:purple', 'tab:green']

            # ==========================================
            # 3. 构造图例句柄
            # ==========================================
            final_handles = []
            final_labels = []
            handler_map = {}

            for i in range(4):
                # 第一行：线条
                final_handles.append(line_styles[i])
                final_labels.append(labels_text[i])
                
                # 第二行：文字图标
                dummy = mpatches.Rectangle((0, 0), 1, 1, alpha=0)
                
                # 将字体大小传入 Handler
                handler_map[dummy] = CenteredTextHandler(
                    text=prefixes[i], 
                    color=colors[i], 
                    fontsize=MY_FONT_SIZE 
                )
                
                final_handles.append(dummy)
                final_labels.append(labels_text[i]) # 标签文字相同，保证对齐

            # ==========================================
            # 4. 绘图与保存
            # ==========================================
            # 强制重置字体配置，防止干扰
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Times New Roman']

            fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

            leg = fig.legend(
                final_handles, final_labels,
                loc='center',
                ncol=4,
                frameon=False,
                
                # ⭐⭐⭐ 这里的 fontsize 控制右边的 "Current (A)"
                fontsize=MY_FONT_SIZE, 
                
                handler_map=handler_map,
                
                # ⭐⭐⭐ 这里的 handlelength 控制左边图标的宽度
                handlelength=MY_HANDLE_LENGTH, 
                
                # 其他微调
                columnspacing=1.0,  # 列间距
                handletextpad=0.4,  # 图标和文字的间距
                labelspacing=0.1    # 行间距
            )

            # 再次给第二行文字上色
            text_objs = leg.get_texts()
            for i in range(4):
                text_objs[2*i + 1].set_color(colors[i])

            # 保存
            save_dir = r'D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT\fig\scale_feature'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'scale_feature_label.pdf')

            # bbox_inches='tight' 会自动裁掉多余白边，确保无论画布多大，结果都紧凑
            fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02, transparent=True)

            print(f"6号字体完美版已保存: {save_path}")
            plt.close(fig)
            # ==========================================
            # 步骤 B: 保存主图 (不带图例)
            # ==========================================

            # --- 8. 调整布局并保存主图 ---

            # (保持防止标签裁切的循环)
            for ax in [ax_current, ax_voltage, ax_charging, ax_discharging]:
                yticks = ax.get_yticks()
                yticklabels = ax.get_yticklabels()
                bottom_val = ax.get_ylim()[0]

                for val, label in zip(yticks, yticklabels):
                    label.set_clip_on(False)
                    # 简单的浮点数比较
                    if abs(val - bottom_val) < 1e-9:
                        label.set_verticalalignment('bottom')

            # (保持布局调整)
            fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.1)
            fig.patch.set_alpha(0)
            for ax in [ax_current, ax_voltage, ax_charging, ax_discharging]:
                ax.patch.set_alpha(0)

            # !!! 关键修改: 必须将 plt.savefig 改为 fig.savefig !!!
            # 这样 Python 才知道您指的是主图那个变量，而不是刚才活动的图例画布

            # 保存 SVG
            fig.savefig(
                os.path.join(OUTPUT_DIR, "charge_curve.svg"),
                format="svg",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0
            )

            output_filename = os.path.join(OUTPUT_DIR, f'b{bat}_cycle_100_plot_NoLegend.png')

            # 保存 PNG
            fig.savefig(
                output_filename,
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.01
            )

            print(f"主图已保存为 '{output_filename}' (无图例，无白边)")
    except FileNotFoundError:
        print(f"错误: 文件 {csv_filename} 未找到。")
    except KeyError as e:
        print(f"错误: 数据中缺少必要的列: {e}。")
