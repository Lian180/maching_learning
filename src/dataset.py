import torch
from torch.utils.data import Dataset

class HousingDataset(Dataset):
    """
    自定义 PyTorch Dataset 类，用于加载房屋数据。
    """
    def __init__(self, features, targets):
        """
        初始化数据集。
        Args:
            features (torch.Tensor): 特征数据张量。
            targets (torch.Tensor): 目标变量（房价）张量。
        """
        self.features = features
        self.targets = targets

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本。
        Args:
            idx (int): 样本索引。
        Returns:
            tuple: (特征张量, 目标张量)。
        """
        return self.features[idx], self.targets[idx]

