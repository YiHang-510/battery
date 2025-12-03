# src/your_dl_project/train.py

"""
模型训练脚本
"""

import os
import yaml
from datetime import datetime
from .model import create_model
from .dataset import create_dataloaders


def load_config(config_path):
    """
    加载配置文件
    
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"配置已加载: {config_path}")
    return config


def train_epoch(model, train_loader, optimizer, criterion, epoch):
    """
    训练一个 epoch
    
    :param model: 模型
    :param train_loader: 训练数据加载器
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param epoch: 当前 epoch 数
    :return: 平均损失
    """
    # model.train()  # 设置为训练模式（PyTorch）
    
    total_loss = 0.0
    num_batches = 0
    
    # 这里实现训练循环
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    #     
    #     total_loss += loss.item()
    #     num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"Epoch {epoch}: 训练损失 = {avg_loss:.4f}")
    
    return avg_loss


def validate(model, val_loader, criterion):
    """
    在验证集上评估模型
    
    :param model: 模型
    :param val_loader: 验证数据加载器
    :param criterion: 损失函数
    :return: 验证损失和准确率
    """
    # model.eval()  # 设置为评估模式（PyTorch）
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 这里实现验证循环
    # with torch.no_grad():  # PyTorch
    #     for data, target in val_loader:
    #         output = model(data)
    #         loss = criterion(output, target)
    #         total_loss += loss.item()
    #         
    #         pred = output.argmax(dim=1)
    #         correct += (pred == target).sum().item()
    #         total += target.size(0)
    
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    
    print(f"验证: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.4f}")
    
    return avg_loss, accuracy


def train(config_path):
    """
    完整的训练流程
    
    :param config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(config['model'])
    
    # 创建数据加载器
    print("\n准备数据...")
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    # train_loader, val_loader = create_dataloaders(
    #     processed_data_path,
    #     batch_size=config['training']['batch_size']
    # )
    
    # 定义损失函数和优化器
    # criterion = nn.CrossEntropyLoss()  # PyTorch
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 训练循环
    print("\n开始训练...")
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # 训练
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        
        # 验证
        # val_loss, val_acc = validate(model, val_loader, criterion)
        
        # 保存最佳模型
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_path = os.path.join(project_root, 'saved_models', 'best_model.pth')
        #     torch.save(model.state_dict(), save_path)
        #     print(f"最佳模型已保存: {save_path}")
    
    print("\n训练完成！")


if __name__ == "__main__":
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_root, 'configs', 'base_config.yaml')
    
    if os.path.exists(config_path):
        train(config_path)
    else:
        print(f"配置文件不存在: {config_path}")
        print("请先创建配置文件")
