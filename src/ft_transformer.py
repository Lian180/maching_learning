import torch
import torch.nn as nn
import math
from einops import rearrange  # 用于张量形状操作


class PositionalEncoding(nn.Module):
    """
    为特征嵌入添加位置编码。
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # 将位置编码添加到输入嵌入中
        x = x + self.pe[:, :x.size(1)]
        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer 模型，用于表格数据回归。
    FT-Transformer (Feature Tokenizer Transformer) 将每个特征视为一个token，
    通过Transformer编码器学习特征间的复杂交互。
    """

    def __init__(self,
                 num_features,  # 数值特征的数量 (这里指总特征数，包括分类和连续)
                 categories=None,  # 包含每个分类特征唯一值数量的列表 (本例中没有分类特征，但保留接口)
                 num_continuous_features=None,  # 数值特征的数量
                 dim=64,  # 嵌入维度 / Transformer 模型的维度
                 depth=6,  # Transformer 编码器层的数量
                 heads=8,  # 多头注意力机制的头数
                 attn_dropout=0.1,  # 注意力机制的Dropout比率
                 ff_dropout=0.1,  # 前馈网络的Dropout比率
                 mlp_hidden_mults=(4, 2),  # 最终MLP的隐藏层倍数
                 output_dim=1,  # 输出维度 (回归任务为1)
                 ):
        super().__init__()
        assert (num_features > 0 or categories is not None), '必须至少有一个数值特征或分类特征'

        if num_continuous_features is None:
            num_continuous_features = num_features  # 默认所有特征都是连续的

        self.num_categories = 0
        if categories is not None:
            self.num_categories = len(categories)
            self.category_embeds = nn.ModuleList([])
            for num_unique_categories in categories:
                self.category_embeds.append(nn.Embedding(num_unique_categories, dim))

        # 为每个数值特征创建一个独立的线性嵌入层
        # 每个数值特征 (维度为1) 映射到 dim 维度
        self.numerical_embeds = nn.ModuleList([nn.Linear(1, dim) for _ in range(num_continuous_features)])

        # CLS token，用于聚合所有特征的信息，类似于BERT
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 位置编码
        total_tokens = self.num_categories + num_continuous_features + 1  # +1 for CLS token
        self.pos_embedding = PositionalEncoding(dim, max_len=total_tokens)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * mlp_hidden_mults[0],
                                                   dropout=attn_dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 最终的MLP回归头
        mlp_layers = []
        current_dim = dim
        for mult in mlp_hidden_mults:
            mlp_layers.append(nn.Linear(current_dim, dim * mult))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(ff_dropout))
            current_dim = dim * mult
        mlp_layers.append(nn.Linear(current_dim, output_dim))
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x_numerical, x_categorical=None):
        """
        Args:
            x_numerical (torch.Tensor): 数值特征，shape [batch_size, num_continuous_features]
            x_categorical (torch.Tensor, optional): 分类特征，shape [batch_size, num_categories]
        """
        # 嵌入数值特征
        numerical_tokens = []
        for i, embed_layer in enumerate(self.numerical_embeds):
            # x_numerical[:, i:i+1] 确保每个特征是一个 [batch_size, 1] 的张量
            # embed_layer(x_numerical[:, i:i+1]) 的输出形状是 [batch_size, dim]
            # 我们需要将其 unsqueeze(1) 变成 [batch_size, 1, dim] 才能正确拼接
            numerical_tokens.append(embed_layer(x_numerical[:, i:i + 1]).unsqueeze(1))

        # 将所有数值特征的嵌入拼接起来
        x_numerical_embed = torch.cat(numerical_tokens, dim=1)  # [batch_size, num_continuous_features, dim]

        # 嵌入分类特征 (如果存在)
        categorical_tokens = []
        if x_categorical is not None and self.num_categories > 0:
            for i, embed_layer in enumerate(self.category_embeds):
                categorical_tokens.append(embed_layer(x_categorical[:, i]))
            x_categorical_embed = torch.stack(categorical_tokens, dim=1)  # [batch_size, num_categories, dim]
            # 合并分类和数值特征嵌入
            x = torch.cat((x_categorical_embed, x_numerical_embed), dim=1)
        else:
            x = x_numerical_embed  # 如果没有分类特征，只使用数值特征

        # 添加 CLS token
        # b, n, d = x.shape # 这一行现在应该是正确的了，因为 x 的形状是 [batch_size, num_features, dim]
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)  # 使用 x.shape[0] 获取 batch_size
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, 1 + num_features, dim]

        # 添加位置编码
        x = self.pos_embedding(x)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)

        # 取 CLS token 的输出作为最终表示
        cls_output = x[:, 0]  # [batch_size, dim]

        # 通过 MLP 头进行回归预测
        return self.mlp_head(cls_output)

