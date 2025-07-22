import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import logging

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# 设置Matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_predictions_vs_actual(y_true, y_pred, title="预测值 vs 真实值", save_path=None):
    """
    绘制预测值与真实值的散点图，并可选择保存。
    Args:
        y_true (torch.Tensor): 真实值 (已在原始尺度)。
        y_pred (torch.Tensor): 预测值 (已在原始尺度)。
        title (str): 图表标题。
        save_path (str, optional): 图表保存路径。
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true.cpu().numpy().flatten(), y=y_pred.cpu().numpy().flatten(), alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("真实值", fontsize=12)
    plt.ylabel("预测值", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        logger.info(f"图表已保存到: {save_path}")
    plt.show()

def plot_residuals(y_true, y_pred, title="残差分布", save_path=None):
    """
    绘制残差的直方图和散点图，并可选择保存。
    Args:
        y_true (torch.Tensor): 真实值 (已在原始尺度)。
        y_pred (torch.Tensor): 预测值 (已在原始尺度)。
        title (str): 图表标题。
        save_path (str, optional): 图表保存路径。
    """
    residuals = y_true.cpu().numpy().flatten() - y_pred.cpu().numpy().flatten()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, color='orange')
    plt.title(f'{title} (直方图)', fontsize=14)
    plt.xlabel('残差 (真实值 - 预测值)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.axvline(0, color='red', linestyle='--', label='残差为0')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred.cpu().numpy().flatten(), y=residuals, alpha=0.6, color='blue')
    plt.axhline(0, color='red', linestyle='--', label='残差为0')
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('残差', fontsize=12)
    plt.title(f'{title} (vs 预测值)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"图表已保存到: {save_path}")
    plt.show()

def calculate_metrics(y_true, y_pred):
    """
    计算并打印评估指标。
    Args:
        y_true (torch.Tensor): 真实值 (已在原始尺度)。
        y_pred (torch.Tensor): 预测值 (已在原始尺度)。
    Returns:
        tuple: (MAE, MSE, RMSE, R2)。
    """
    y_true_np = y_true.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()

    mae = mean_absolute_error(y_true_np, y_pred_np)
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_np, y_pred_np)

    logger.info(f"平均绝对误差 (MAE): {mae:.4f}")
    logger.info(f"均方误差 (MSE): {mse:.4f}")
    logger.info(f"均方根误差 (RMSE): {rmse:.4f}")
    logger.info(f"R-squared (R²): {r2:.4f}")

    return mae, mse, rmse, r2

