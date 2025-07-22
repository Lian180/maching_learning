import torch
import numpy as np
import random
import os
import logging

# 获取日志记录器实例
logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    设置所有随机种子，以确保结果的可复现性。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"已设置随机种子为: {seed}")

def save_model(model, path):
    """
    保存 PyTorch 模型的状态字典。
    """
    torch.save(model.state_dict(), path)
    logger.info(f"模型已保存到: {path}")

def load_model(model, path, device='cpu'):
    """
    加载 PyTorch 模型的状态字典。
    """
    model.load_state_dict(torch.load(path, map_location=device))
    logger.info(f"模型已从 '{path}' 加载。")

