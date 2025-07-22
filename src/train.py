import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import os
import logging

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# 设置Matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs,
                scheduler=None, patience=10, min_delta=0.0001, log_dir='logs'):  # 添加 log_dir 参数
    """
    训练神经网络模型，加入学习率调度器和早停机制，并保存训练图表。
    """
    logger.info("\n--- 模型训练开始 ---")
    model.train()

    train_losses = []
    val_losses = []
    val_r2_scores = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 在验证集上评估，这里不需要反标准化，因为损失和R2是在标准化空间计算的
        val_loss, val_rmse, val_r2, _, _ = evaluate_model_on_dataloader(model, val_loader, criterion, device, is_training_phase=True)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"训练损失: {epoch_train_loss:.4f}, "
                    f"验证损失: {val_loss:.4f}, "
                    f"验证 R2 (标准化): {val_r2:.4f}") # 明确R2是在标准化空间

        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"早停！验证损失在 {patience} 个epoch内没有显著改善。")
                break

    model.load_state_dict(best_model_wts)
    logger.info("--- 模型训练完成 ---")

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练与验证损失曲线 (标准化目标)', fontsize=14) # 更新标题
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失 (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, '训练与验证损失曲线.png')
    plt.savefig(plot_path)  # 保存图表
    logger.info(f"图表已保存到: {plot_path}")
    plt.show()  # 显示图表

    # 绘制验证集 R2 曲线
    plt.figure(figsize=(12, 5))  # 新建一个 figure 以免覆盖
    plt.plot(val_r2_scores, label='验证 R2', color='orange')
    plt.title('验证集 R2 分数曲线 (标准化目标)', fontsize=14) # 更新标题
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R-squared (R²)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(log_dir, '验证集R2分数曲线.png')
    plt.savefig(plot_path)  # 保存图表
    logger.info(f"图表已保存到: {plot_path}")
    plt.show()  # 显示图表


def evaluate_model_on_dataloader(model, data_loader, criterion, device, is_training_phase=False, target_scaler=None):
    """
    在给定数据加载器上评估模型。
    Args:
        model (nn.Module): 要评估的神经网络模型。
        data_loader (DataLoader): 数据加载器 (例如测试集或验证集)。
        criterion (nn.Module): 损失函数。
        device (torch.device): 评估设备 (CPU 或 GPU)。
        is_training_phase (bool): 是否在训练阶段调用 (用于控制打印信息)。
        target_scaler: 用于目标变量反标准化的 scaler 对象。
    Returns:
        tuple: (平均损失, RMSE, R2, 原始尺度的真实值, 原始尺度的预测值)。
    """
    model.eval()
    total_loss = 0.0
    all_targets_scaled = [] # 收集缩放后的真实值
    all_predictions_scaled = [] # 收集缩放后的预测值

    with torch.no_grad():
        for inputs, targets_scaled in data_loader: # targets_scaled 是缩放后的
            inputs, targets_scaled = inputs.to(device), targets_scaled.to(device)
            outputs_scaled = model(inputs) # outputs_scaled 是缩放后的预测

            loss = criterion(outputs_scaled, targets_scaled) # 损失在缩放空间计算

            total_loss += loss.item() * inputs.size(0)
            all_targets_scaled.append(targets_scaled.cpu())
            all_predictions_scaled.append(outputs_scaled.cpu())

    avg_loss = total_loss / len(data_loader.dataset)
    y_true_scaled = torch.cat(all_targets_scaled, dim=0)
    y_pred_scaled = torch.cat(all_predictions_scaled, dim=0)

    # 如果提供了 target_scaler，则将真实值和预测值反标准化到原始尺度
    if target_scaler:
        y_true_original = torch.tensor(target_scaler.inverse_transform(y_true_scaled.numpy()), dtype=torch.float32)
        y_pred_original = torch.tensor(target_scaler.inverse_transform(y_pred_scaled.numpy()), dtype=torch.float32)
    else:
        y_true_original = y_true_scaled
        y_pred_original = y_pred_scaled

    # 使用原始尺度的值计算 RMSE 和 R2
    rmse = np.sqrt(mean_squared_error(y_true_original.numpy(), y_pred_original.numpy()))
    r2 = r2_score(y_true_original.numpy(), y_pred_original.numpy())

    if not is_training_phase:
        logger.info(f"评估结果 (原始房价尺度) - 损失: {avg_loss:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return avg_loss, rmse, r2, y_true_original, y_pred_original # 返回原始尺度的值

