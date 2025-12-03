# src/your_dl_project/dataset.py

"""
自定义数据集类
适用于 PyTorch 的 Dataset 接口
如果你使用 TensorFlow，可以改为创建 tf.data.Dataset
"""

import os
import numpy as np
import pandas as pd

# 如果使用 PyTorch，取消下面的注释
# from torch.utils.data import Dataset


class CustomDataset:
    """
    自定义数据集类
    
    如果使用 PyTorch，请让这个类继承 torch.utils.data.Dataset
    并实现 __len__ 和 __getitem__ 方法
    """
    
    def __init__(self, data_path, transform=None):
        """
        初始化数据集
        
        :param data_path: 处理后的数据路径
        :param transform: 数据转换/增强操作
        """
        self.data_path = data_path
        self.transform = transform
        
        # 加载数据
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.npy'):
            self.data = np.load(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        print(f"数据集已加载，样本数: {len(self.data)}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        :param idx: 样本索引
        :return: (features, label) 或根据你的任务调整
        """
        # 示例：假设数据的最后一列是标签
        sample = self.data.iloc[idx] if isinstance(self.data, pd.DataFrame) else self.data[idx]
        
        # 根据你的任务分离特征和标签
        # features = sample[:-1]
        # label = sample[-1]
        
        # 如果有数据转换操作，应用它们
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def create_dataloaders(processed_data_path, batch_size=32, train_split=0.8):
    """
    创建训练和验证数据加载器
    
    :param processed_data_path: 处理后的数据路径
    :param batch_size: 批大小
    :param train_split: 训练集比例
    :return: train_loader, val_loader
    """
    # 这里是一个框架示例
    # 实际实现取决于你使用的深度学习框架
    
    print(f"创建数据加载器，批大小: {batch_size}")
    print(f"训练集比例: {train_split}")
    
    # PyTorch 示例（需要取消注释相关导入）:
    # dataset = CustomDataset(processed_data_path)
    # train_size = int(train_split * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # return train_loader, val_loader
    
    pass


if __name__ == "__main__":
    # 测试数据集加载
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    
    if os.path.exists(data_path):
        dataset = CustomDataset(data_path)
        print(f"数据集测试成功，包含 {len(dataset)} 个样本")
    else:
        print(f"数据文件不存在: {data_path}")
        print("请先运行 data_processing.py 生成处理后的数据")
