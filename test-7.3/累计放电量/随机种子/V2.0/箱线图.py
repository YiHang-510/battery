import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib

# --- 设置全局字体 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def plot_box_only_comparison():
    # ---------------- 配置区域 ----------------

    # #cc
    # csv_files = {
    #     'M1': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MCNN_BiRNN_AM\cc\run_2\test_details_all_predictions.csv',
    #     'M2': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MoE_Model\cc\run_3\test_details_all_predictions.csv',
    #     'M3': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\RCMHCRE\cc\run_2\test_details_all_predictions.csv',
    #     'M4': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\CNN-LSTM-TPA\cc\run_2\test_details_all_predictions.csv',
    #     'M5': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\自己的实验\TM_PIRes\cc\validation_results.csv'
    # }
    
    # #cv
    # csv_files = {
    #     'M1': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MCNN_BiRNN_AM\cv\run_2\test_details_all_predictions.csv',
    #     'M2': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MoE_Model\cv\run_3\test_details_all_predictions.csv',
    #     'M3': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\RCMHCRE\cv\run_4\test_details_all_predictions.csv',
    #     'M4': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\CNN-LSTM-TPA\cv\run_3\test_details_all_predictions.csv',
    #     'M5': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\自己的实验\TM_PIRes\cv\validation_results.csv'
    # }

    # #vc
    # csv_files = {
    #     'M1': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MCNN_BiRNN_AM\vc\run_2\test_details_all_predictions.csv',
    #     'M2': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MoE_Model\vc\run_5\test_details_all_predictions.csv',
    #     'M3': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\RCMHCRE\vc\run_5\test_details_all_predictions.csv',
    #     'M4': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\CNN-LSTM-TPA\vc\run_3\test_details_all_predictions.csv',
    #     'M5': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\自己的实验\TM_PIRes\vc\validation_results.csv'
    # }

    #vv
    csv_files = {
        'M1': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MCNN_BiRNN_AM\vv\run_1\test_details_all_predictions.csv',
        'M2': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\MoE_Model\vv\run_3\test_details_all_predictions.csv',
        'M3': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\RCMHCRE\vv\run_1\test_details_all_predictions.csv',
        'M4': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\对比实验-带预测真值\CNN-LSTM-TPA\vv\run_5\test_details_all_predictions.csv',
        'M5': r'D:\任务归档\电池\研究\二稿-小论文1号\RESULT\自己的实验\TM_PIRes\vv\validation_results.csv'
    }
    
    # 图片颜色设置
    my_colors = {
        'M1': '#8ECFC9',  
        'M2': '#FFBE7A',  
        'M3': '#FA7F6F',  
        'M4': '#82B0D2',  
        'M5': '#BEB8DC'   
    }

    size = 48
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # ---------------- 数据处理 ----------------
    data = []
    model_names = list(csv_files.keys())

    print("开始读取数据...")
    for model_name, file_path in csv_files.items():
        if not os.path.exists(file_path):
            print(f"[警告] 文件不存在: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"[错误] 读取文件 {file_path} 失败: {e}")
            continue
        
        # --- 兼容不同列名 ---
        cols_map = {c.lower(): c for c in df.columns}
        
        # 1. 确定真值列
        if 'true_soh' in cols_map:
            true_col = cols_map['true_soh']
        elif 'true' in cols_map:
            true_col = cols_map['true']
        else:
            print(f"[跳过] {model_name}: 文件中未找到 'true_soh' 或 'TRUE' 列")
            continue

        # 2. 确定预测值列
        if 'pred_soh' in cols_map:
            pred_col = cols_map['pred_soh']
        elif 'pred' in cols_map:
            pred_col = cols_map['pred']
        else:
            print(f"[跳过] {model_name}: 文件中未找到 'pred_soh' 或 'pred' 列")
            continue

        print(f"[{model_name}] 使用列名: True='{true_col}', Pred='{pred_col}'")

        true_vals = df[true_col].values
        pred_vals = df[pred_col].values
        
        # 确保长度一致
        min_len = min(len(true_vals), len(pred_vals))
        true_vals = true_vals[:min_len]
        pred_vals = pred_vals[:min_len]

        # 计算误差
        err = abs(pred_vals - true_vals)
        
        # 去掉第一个和最后一个点
        if len(err) > 2:
            err = err[1:-1]

        for val in err:
            data.append({'Model': model_name, 'Error': val})

    if not data:
        print("没有有效数据用于绘图，请检查文件路径和列名。")
        return

    df_plot = pd.DataFrame(data)

    # ---------------- 绘图逻辑 (仅箱线图) ----------------
    
    # 纯箱线图绘制
    sns.boxplot(
        x='Model', y='Error',
        hue='Model',        # 指定 hue 以便 palette 生效
        data=df_plot,
        ax=ax,
        width=0.5,          # 增加宽度，使其占据小提琴原本的空间
        dodge=False,        # x与hue一致时，关闭dodge防止偏移
        showfliers=False,   # 不显示异常值点
        palette=my_colors,  # 应用自定义颜色
        legend=False,       # 不显示图例
        
        # 设置箱体边框和填充
        # 注意：这里去掉了 facecolor='none'，允许颜色填充
        boxprops=dict(edgecolor='black', linewidth=2.2, alpha=0.9), 
        whiskerprops=dict(color='black', linewidth=2.2),
        capprops=dict(color='black', linewidth=2.2),
        medianprops=dict(color='black', linewidth=2.2)
    )

    # 绘制均值红线
    grouped = df_plot.groupby('Model')['Error'].mean().reset_index()
    for j, model in enumerate(model_names):
        if model in grouped['Model'].values:
            mean_val = grouped[grouped['Model'] == model]['Error'].values[0]
            # 这里的 0.25 对应 width=0.5 的一半，使红线宽度与箱体一致
            ax.plot([j - 0.25, j + 0.25], [mean_val, mean_val], color='red', linewidth=2.5)

    # ---------------- 样式调整 ----------------
    ax.set_xlabel('Model', fontsize=54)
    ax.set_ylabel('Absolute error', fontsize=54)
    ax.tick_params(labelsize=size)
    
    # 设置 Y 轴范围
    ax.set_ylim(0, 0.05)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 保存图片
    # 修改文件名以示区别
    os.makedirs('comparison_plot', exist_ok=True)
    save_path = r'D:\任务归档\电池\研究\二稿-小论文1号\DOCUMENT_v0.1\fig\Results\box_fig\boxplot_error_vv.pdf'
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"绘图完成，已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_box_only_comparison()