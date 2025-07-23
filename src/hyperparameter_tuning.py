import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import optuna
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import logging

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# 从其他模块导入必要的函数和类
from src.model import MLPRegressor
from src.ft_transformer import FTTransformer
from src.train import train_model, evaluate_model_on_dataloader
from src.utils import set_seed


def objective(trial, train_loader, val_loader, input_dim, output_dim, device, num_epochs=200):
    """
    Optuna 优化的目标函数。
    这个函数会根据 Optuna 建议的超参数来训练和评估一个模型。
    """
    # 1. 建议超参数
    lr = trial.suggest_loguniform('lr', 5e-6, 5e-3) # 调整学习率搜索范围，更集中在可能的最优值附近
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-4) # 添加权重衰减

    # 选择模型类型：MLP 或 FT-Transformer
    model_type = trial.suggest_categorical('model_type', ['MLP', 'FTTransformer'])

    if model_type == 'MLP':
        n_layers = trial.suggest_int('mlp_n_layers', 3, 6) # 增加 MLP 层数上限
        hidden_layers = []
        for i in range(n_layers):
            # 增加隐藏单元数上限，并调整步长
            hidden_layers.append(trial.suggest_int(f'mlp_n_units_l{i}', 64, 768, step=32))
        dropout_rate = trial.suggest_uniform('mlp_dropout_rate', 0.1, 0.4) # 调整 Dropout 范围
        model = MLPRegressor(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim,
                             dropout_rate=dropout_rate).to(device)
        logger.info(f"Trial {trial.number}: 选择了 MLP 模型")
    else:  # model_type == 'FTTransformer'
        # FT-Transformer 的超参数
        dim = trial.suggest_categorical('ftt_dim', [128, 256, 512]) # 增加维度选项
        depth = trial.suggest_int('ftt_depth', 4, 8) # 增加深度选项
        heads = trial.suggest_categorical('ftt_heads', [8, 16]) # 增加头数选项
        attn_dropout = trial.suggest_uniform('ftt_attn_dropout', 0.05, 0.25) # 调整 Dropout 范围
        ff_dropout = trial.suggest_uniform('ftt_ff_dropout', 0.05, 0.25) # 调整 Dropout 范围

        model = FTTransformer(
            num_features=input_dim,
            num_continuous_features=input_dim,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            output_dim=output_dim
        ).to(device)
        logger.info(f"Trial {trial.number}: 选择了 FT-Transformer 模型")

    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # 添加 weight_decay
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay) # 添加 weight_decay

    # 4. 定义学习率调度器
    # 调整 ReduceLROnPlateau 的 patience 和 factor
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=7)

    # 5. 训练模型 (使用早停)
    patience_es = 15 # 增加早停耐心值
    min_delta_es = 0.000005 # 减小最小改善量，更灵敏地捕捉提升

    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs,
                scheduler=scheduler, patience=patience_es, min_delta=min_delta_es, log_dir='logs')

    # 6. 评估模型并返回指标
    # 注意：这里返回的 val_loss 是在标准化空间计算的，因为 Optuna 优化的是这个值
    val_loss, _, _, _, _ = evaluate_model_on_dataloader(model, val_loader, criterion, device, is_training_phase=True)

    return val_loss


def run_hyperparameter_optimization(train_loader, val_loader, input_dim, output_dim, device, n_trials=10): #
    """
    运行 Optuna 超参数优化。
    """
    logger.info(f"\n--- 开始 Optuna 超参数优化 (共 {n_trials} 次试验) ---")

    study = optuna.create_study(direction='minimize')

    study.optimize(lambda trial: objective(trial, train_loader, val_loader, input_dim, output_dim, device),
                   n_trials=n_trials,
                   show_progress_bar=True)

    logger.info("\n--- Optuna 超参数优化完成 ---")
    logger.info(f"最佳 Trial 的值 (验证损失): {study.best_value:.4f}")
    logger.info(f"最佳超参数: {study.best_params}")

    return study

