# src/your_dl_project/evaluate.py

"""
模型评估脚本
"""

import os
import yaml
import numpy as np
from .model import create_model
from .dataset import CustomDataset


def load_model(model_path, config):
    """
    加载训练好的模型
    
    :param model_path: 模型权重文件路径
    :param config: 模型配置
    :return: 加载的模型
    """
    model = create_model(config)
    
    # PyTorch 加载示例:
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    
    # TensorFlow 加载示例:
    # model = keras.models.load_model(model_path)
    
    print(f"模型已从 {model_path} 加载")
    return model


def evaluate_model(model, test_loader):
    """
    在测试集上评估模型
    
    :param model: 训练好的模型
    :param test_loader: 测试数据加载器
    :return: 评估指标字典
    """
    print("开始评估模型...")
    
    all_predictions = []
    all_targets = []
    
    # 评估循环
    # with torch.no_grad():  # PyTorch
    #     for data, target in test_loader:
    #         output = model(data)
    #         pred = output.argmax(dim=1)
    #         all_predictions.extend(pred.cpu().numpy())
    #         all_targets.extend(target.cpu().numpy())
    
    # 计算评估指标
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # 
    # metrics = {
    #     'accuracy': accuracy_score(all_targets, all_predictions),
    #     'precision': precision_score(all_targets, all_predictions, average='weighted'),
    #     'recall': recall_score(all_targets, all_predictions, average='weighted'),
    #     'f1_score': f1_score(all_targets, all_predictions, average='weighted')
    # }
    
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    
    print("\n评估结果:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics


def evaluate(model_path, config_path):
    """
    完整的评估流程
    
    :param model_path: 模型权重文件路径
    :param config_path: 配置文件路径
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 加载模型
    model = load_model(model_path, config['model'])
    
    # 准备测试数据
    # processed_data_path = os.path.join(project_root, 'data', 'processed', 'test_data.csv')
    # test_loader = create_dataloaders(processed_data_path, batch_size=config['training']['batch_size'])
    
    # 评估
    # metrics = evaluate_model(model, test_loader)
    
    print("\n评估完成！")


if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    model_path = os.path.join(project_root, 'saved_models', 'best_model.pth')
    config_path = os.path.join(project_root, 'configs', 'base_config.yaml')
    
    if os.path.exists(model_path) and os.path.exists(config_path):
        evaluate(model_path, config_path)
    else:
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
        print("请先训练模型并创建配置文件")
