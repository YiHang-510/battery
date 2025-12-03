# src/your_dl_project/data_processing.py

"""
数据处理模块
负责加载、清洗和预处理数据
"""

import os
import numpy as np
import pandas as pd


def load_raw_data(data_path):
    """
    从指定路径加载原始数据
    
    :param data_path: 原始数据文件路径
    :return: 加载的数据
    """
    # 示例：加载 CSV 文件
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
        print(f"成功加载数据，形状: {data.shape}")
        return data
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")


def preprocess_data(data):
    """
    预处理数据（清洗、归一化、特征工程等）
    
    :param data: 原始数据
    :return: 预处理后的数据
    """
    # 示例：简单的数据预处理
    # 这里你可以添加：
    # - 缺失值处理
    # - 数据标准化/归一化
    # - 特征编码
    # - 数据增强等
    
    print("正在预处理数据...")
    processed_data = data.copy()
    
    # 示例：删除缺失值
    processed_data = processed_data.dropna()
    
    print(f"预处理完成，数据形状: {processed_data.shape}")
    return processed_data


def save_processed_data(data, save_path):
    """
    保存预处理后的数据
    
    :param data: 预处理后的数据
    :param save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_path.endswith('.csv'):
        data.to_csv(save_path, index=False)
    elif save_path.endswith('.npy'):
        np.save(save_path, data)
    else:
        raise ValueError(f"不支持的保存格式: {save_path}")
    
    print(f"数据已保存到: {save_path}")


def prepare_dataset():
    """
    完整的数据准备流程
    从 data/raw 加载数据，处理后保存到 data/processed
    """
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 定义路径
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'your_dataset.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    
    # 检查原始数据是否存在
    if not os.path.exists(raw_data_path):
        print(f"警告: 原始数据文件不存在: {raw_data_path}")
        print("请将你的数据集放在 data/raw/ 目录下")
        return
    
    # 加载、预处理和保存数据
    data = load_raw_data(raw_data_path)
    processed_data = preprocess_data(data)
    save_processed_data(processed_data, processed_data_path)
    
    print("数据准备完成！")


if __name__ == "__main__":
    prepare_dataset()
