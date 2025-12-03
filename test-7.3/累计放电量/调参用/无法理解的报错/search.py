import os
import subprocess
import itertools
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm

# --- 1. 定义超参数搜索空间 ---
# 您可以在这里定义想要尝试的任何参数值
param_grid = {
    'd_model': [128, 256, 512],
    'd_ff': [256, 512, 1024],
    'dropout': [0.1, 0.2, 0.3],
    'weight_decay': [1e-4, 1e-5, 5e-5],
    'batch_size': [64, 128, 256, 512],
    'learning_rate': [0.001, 0.002, 0.005],
    'patience': [15, 25]
}


# --- 2. 定义单个实验的执行函数 ---
# 这个函数会被每个并行的进程调用
def run_experiment(params):
    """
    为一组给定的参数运行训练脚本。
    参数:
        params (tuple): 包含 (gpu_id, param_dict) 的元组。
    """
    gpu_id, param_dict = params

    # 构建命令行
    command = ['python', 'cyclenet.py']
    for key, value in param_dict.items():
        command.append(f'--{key}')
        command.append(str(value))

    # 设置环境变量，为这个子进程指定GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"--- [GPU {gpu_id}] 开始运行: {param_dict} ---")

    # 运行子进程
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)

        # 实验成功，读取结果
        exp_name = (f"dm{param_dict['d_model']}_dff{param_dict['d_ff']}_dp{param_dict['dropout']}_"
                    f"wd{param_dict['weight_decay']}_bs{param_dict['batch_size']}_lr{param_dict['learning_rate']}_"
                    f"p{param_dict['patience']}")

        base_save_path = '/home/scuee_user06/myh/电池/result-累计放电容量/cyclenet/hyperparam_search'  # 确保和训练脚本中的路径一致
        result_path = os.path.join(base_save_path, exp_name, 'final_evaluation_metrics_orig.csv')

        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            # 我们关心验证集的R2分数
            val_r2 = df[df['set'] == 'validation']['R2'].iloc[0]
            print(f"--- [GPU {gpu_id}] 运行成功: R2={val_r2:.4f} ---")
            return {**param_dict, 'validation_r2': val_r2}
        else:
            print(f"--- [GPU {gpu_id}] 运行警告: 找不到结果文件 {result_path} ---")
            return {**param_dict, 'validation_r2': -1}  # 表示失败

    except subprocess.CalledProcessError as e:
        print(f"--- [GPU {gpu_id}] 运行失败: {param_dict} ---")
        print(f"错误信息: {e.stderr}")
        return {**param_dict, 'validation_r2': -1}  # 表示失败


# --- 3. 主逻辑 ---
if __name__ == '__main__':
    # GPU数量
    NUM_GPUS = 3

    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"总共需要运行 {len(param_combinations)} 组实验。")

    # 将GPU ID (0, 1, 2) 循环分配给每个任务
    tasks_with_gpu = [(i % NUM_GPUS, params) for i, params in enumerate(param_combinations)]

    # 创建一个进程池，进程数等于GPU数量
    # 使用tqdm来显示进度条
    results = []
    with Pool(processes=NUM_GPUS) as pool:
        for result in tqdm(pool.imap_unordered(run_experiment, tasks_with_gpu), total=len(tasks_with_gpu)):
            if result:
                results.append(result)

    print("\n--- 所有实验已完成 ---")

    # 将结果保存到CSV文件
    if results:
        results_df = pd.DataFrame(results)
        # 按验证集R2分数降序排序
        results_df = results_df.sort_values(by='validation_r2', ascending=False)

        results_df.to_csv('hyperparameter_search_results.csv', index=False)

        print("搜索结果已保存到 'hyperparameter_search_results.csv'")
        print("\n--- 最佳参数组合 ---")
        print(results_df.iloc[0])
    else:
        print("没有成功的实验结果。")