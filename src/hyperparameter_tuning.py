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


def objective(trial, train_loader, val_loader, input_dim, output_dim, device, num_epochs=50):  # 将 num_epochs 减少到 50
    """
    Optuna 优化的目标函数。
    这个函数会根据 Optuna 建议的超参数来训练和评估一个模型。
    """
    # 1. 建议超参数
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])

    # 选择模型类型：MLP 或 FT-Transformer
    model_type = trial.suggest_categorical('model_type', ['MLP', 'FTTransformer'])

    if model_type == 'MLP':
        n_layers = trial.suggest_int('mlp_n_layers', 2, 5)
        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(trial.suggest_int(f'mlp_n_units_l{i}', 32, 512, step=32))
        dropout_rate = trial.suggest_uniform('mlp_dropout_rate', 0.1, 0.5)
        model = MLPRegressor(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim,
                             dropout_rate=dropout_rate).to(device)
        logger.info(f"Trial {trial.number}: 选择了 MLP 模型")
    else:  # model_type == 'FTTransformer'
        # FT-Transformer 的超参数
        dim = trial.suggest_categorical('ftt_dim', [64, 128, 256])
        depth = trial.suggest_int('ftt_depth', 2, 6)
        heads = trial.suggest_categorical('ftt_heads', [4, 8])
        attn_dropout = trial.suggest_uniform('ftt_attn_dropout', 0.05, 0.3)
        ff_dropout = trial.suggest_uniform('ftt_ff_dropout', 0.05, 0.3)

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
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # 4. 定义学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. 训练模型 (使用早停)
    patience_es = 10
    min_delta_es = 0.00001

    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs,
                scheduler=scheduler, patience=patience_es, min_delta=min_delta_es, log_dir='logs')

    # 6. 评估模型并返回指标
    # 注意：这里返回的 val_loss 是在标准化空间计算的，因为 Optuna 优化的是这个值
    val_loss, _, _, _, _ = evaluate_model_on_dataloader(model, val_loader, criterion, device, is_training_phase=True)

    return val_loss


def run_hyperparameter_optimization(train_loader, val_loader, input_dim, output_dim, device, n_trials=50):
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

