import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    残差块，包含两个线性层、Batch Normalization和ReLU激活函数，并带有跳跃连接。
    """
    def __init__(self, in_features, out_features, dropout_rate=0.2): # 添加 dropout_rate 参数
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features) # 批量归一化
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate) # 添加 Dropout 层

        # 如果输入和输出维度不同，需要一个额外的线性层来匹配维度
        self.shortcut = nn.Identity() # 默认是恒等映射
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = x # 保存输入，用于残差连接

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout(out) # 在残差块内部应用 Dropout

        # 将跳跃连接的输出与主路径的输出相加
        out += self.shortcut(identity)
        out = self.relu(out) # 再次激活

        return out

class MLPRegressor(nn.Module):
    """
    多层感知机 (MLP) 回归模型，包含批量归一化、Dropout和残差连接。
    """
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate=0.2): # 添加 dropout_rate 参数
        """
        初始化 MLP 回归模型。
        Args:
            input_dim (int): 输入特征的维度。
            hidden_layers (list): 包含每个隐藏层神经元数量的列表。
            output_dim (int): 输出的维度（对于回归通常是1）。
            dropout_rate (float): Dropout 的比率。
        """
        super(MLPRegressor, self).__init__()

        layers = []
        current_dim = input_dim

        # 构建隐藏层
        for i, h_dim in enumerate(hidden_layers):
            if i == 0: # 第一个隐藏层
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate)) # 使用传入的 dropout_rate
            else: # 后续隐藏层使用残差块
                layers.append(ResidualBlock(current_dim, h_dim, dropout_rate=dropout_rate)) # 传递 dropout_rate 给残差块
            current_dim = h_dim

        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入数据张量。
        Returns:
            torch.Tensor: 模型的预测输出。
        """
        return self.network(x)

