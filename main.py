import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import joblib
import warnings
import time
import optuna
import logging
import os

# 从 src 模块导入函数
from src.data_preprocessing import load_data, initial_data_exploration, visualize_initial_data, \
    handle_missing_values, handle_outliers, feature_engineering, \
    split_and_scale_data
from src.dataset import HousingDataset
from src.model import MLPRegressor
from src.ft_transformer import FTTransformer  # 导入 FTTransformer
from src.train import train_model, evaluate_model_on_dataloader
from src.evaluate import plot_predictions_vs_actual, plot_residuals  # 导入绘图函数
from src.utils import set_seed, save_model, load_model
from src.hyperparameter_tuning import run_hyperparameter_optimization, objective # 导入 objective

# 忽略警告，使输出更整洁
warnings.filterwarnings('ignore')

# 设置Matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")  # 这行保留print，因为日志系统还没完全初始化

# --- 配置日志记录 ---
log_dir = 'logs'  # 统一使用 logs 文件夹
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"project_run_ftt_optuna_{time.strftime('%Y%m%d_%H%M%S')}.log")  # 修改日志文件名

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 项目主入口 ---
if __name__ == "__main__":
    logger.info("--- 机器学习房屋价格预测项目 (深度学习版) - 集成FT-Transformer和Optuna启动 ---")

    # 定义数据文件路径和模型保存路径前缀
    data_file_path = './data/california_housing.csv'
    model_save_path_prefix = './models/housing_model_ftt_optuna'  # 修改模型保存路径前缀
    scaler_save_path = './models/scaler_ftt_optuna.pkl'  # 修改scaler保存路径
    target_scaler_save_path = './models/target_scaler_ftt_optuna.pkl'  # 新增目标scaler保存路径

    # 设置随机种子，确保结果可复现
    set_seed(42)

    # --- 1. 数据层面：数据获取、探索与预处理 ---
    logger.info("\n--- 阶段1: 数据获取、探索与预处理 ---")
    df = load_data(data_file_path)
    initial_data_exploration(df.copy())
    visualize_initial_data(df.copy(), save_dir=log_dir)

    # 1.1 缺失值处理
    df_processed = handle_missing_values(df)

    # 1.2 异常值处理 (使用IQR方法)
    df_processed = handle_outliers(df_processed, method='iqr')

    # 1.3 特征工程
    df_processed = feature_engineering(df_processed)

    # 1.4 数据划分与标准化
    # 将数据划分为训练+验证集 和 测试集
    # 80% 训练+验证, 20% 测试
    X_temp_df, X_test_df, y_temp_df, y_test_df, feature_scaler, target_scaler = split_and_scale_data(
        df_processed, target_column='target', test_size=0.1, random_state=42
    )

    # 从训练+验证集中再划分出训练集和验证集 (例如 80% 训练，20% 验证)
    # 注意：X_temp_df 和 y_temp_df 已经是缩放过的，直接使用 train_test_split 即可
    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        X_temp_df, y_temp_df, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )

    # 保存 feature_scaler 和 target_scaler，以便新数据预测时使用
    joblib.dump(feature_scaler, scaler_save_path)
    joblib.dump(target_scaler, target_scaler_save_path)
    logger.info(f"特征标准化器已保存到: {scaler_save_path}")
    logger.info(f"目标标准化器已保存到: {target_scaler_save_path}")

    # 将处理后的DataFrame转换为PyTorch Tensor
    X_train_tensor = torch.tensor(X_train_df.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_df.values, dtype=torch.float32).view(-1, 1)  # 转换为列向量
    X_val_tensor = torch.tensor(X_val_df.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_df.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_df.values, dtype=torch.float32).view(-1, 1)

    batch_size = 64 # 恢复 batch_size 为 64
    train_dataset = HousingDataset(X_train_tensor, y_train_tensor)
    val_dataset = HousingDataset(X_val_tensor, y_val_tensor)
    test_dataset = HousingDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"\n训练集特征形状 (Tensor): {X_train_tensor.shape}")
    logger.info(f"验证集特征形状 (Tensor): {X_val_tensor.shape}")
    logger.info(f"测试集特征形状 (Tensor): {X_test_tensor.shape}")
    logger.info(f"训练集目标形状 (Tensor): {y_train_tensor.shape}")
    logger.info(f"验证集目标形状 (Tensor): {y_val_tensor.shape}")
    logger.info(f"测试集目标形状 (Tensor): {y_test_tensor.shape}")
    logger.info(f"训练数据加载器包含 {len(train_loader)} 个批次。")
    logger.info(f"验证数据加载器包含 {len(val_loader)} 个批次。")
    logger.info(f"测试数据加载器包含 {len(test_loader)} 个批次。")

    # --- 2. 方法层面：超参数优化 (Optuna) ---
    logger.info("\n--- 阶段2: 超参数优化 (Optuna) ---")
    input_dim = X_train_tensor.shape[1]
    output_dim = 1

    # 直接调用 run_hyperparameter_optimization，它会调用 objective
    study = run_hyperparameter_optimization(train_loader, val_loader, input_dim, output_dim, device, n_trials=10) #

    best_params = study.best_params
    logger.info(f"\nOptuna 找到的最佳超参数: {best_params}")

    # 根据 Optuna 找到的最佳模型类型和超参数构建最终模型
    best_model_type = best_params['model_type']
    if best_model_type == 'MLP':
        best_model = MLPRegressor(
            input_dim=input_dim,
            hidden_layers=[best_params[f'mlp_n_units_l{i}'] for i in range(best_params['mlp_n_layers'])],
            output_dim=output_dim,
            dropout_rate=best_params['mlp_dropout_rate']
        ).to(device)
    elif best_model_type == 'FTTransformer':
        best_model = FTTransformer(
            num_features=input_dim,
            num_continuous_features=input_dim,
            dim=best_params['ftt_dim'],
            depth=best_params['ftt_depth'],
            heads=best_params['ftt_heads'],
            attn_dropout=best_params['ftt_attn_dropout'],
            ff_dropout=best_params['ftt_ff_dropout'],
            output_dim=output_dim
        ).to(device)
    else:
        raise ValueError(f"未知模型类型: {best_model_type}")

    logger.info("\n使用最佳超参数构建的最终模型结构:")
    logger.info(best_model)

    # 根据最佳超参数选择优化器
    if best_params['optimizer'] == 'Adam':
        final_optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']) # 添加 weight_decay
    elif best_params['optimizer'] == 'RMSprop':
        final_optimizer = optim.RMSprop(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']) # 添加 weight_decay
    else:
        # 默认使用 Adam，并确保 weight_decay 参数被传递
        final_optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    final_criterion = nn.MSELoss()
    final_scheduler = lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.3, patience=15) # 调整最终训练的调度器参数

    logger.info(f"\n最终模型使用的损失函数: {final_criterion.__class__.__name__}")
    logger.info(f"最终模型使用的优化器: {final_optimizer.__class__.__name__}, 学习率: {final_optimizer.defaults['lr']:.6f}, 权重衰减: {final_optimizer.defaults['weight_decay']:.6f}") # 打印权重衰减
    logger.info(f"最终模型使用的学习率调度器: {final_scheduler.__class__.__name__}")

    # --- 3. 方法层面：模型训练与调优 (使用最佳超参数) ---
    logger.info("\n--- 阶段3: 模型训练与调优 (使用最佳超参数) ---")
    num_epochs_final = 200  # 恢复为100个Epoch
    patience_final = 30  # 最终训练早停耐心值可以更大
    min_delta_final = 0.000001  # 最终训练早停最小改善量，更小以捕捉微弱提升

    train_model(best_model, train_loader, val_loader, final_criterion, final_optimizer, device, num_epochs_final,
                scheduler=final_scheduler, patience=patience_final, min_delta=min_delta_final, log_dir=log_dir)

    # --- 4. 分析层面：模型评估与可视化 ---
    logger.info("\n--- 阶段4: 模型评估与可视化 ---")
    # 在测试集上评估模型
    # 传递 target_scaler 给 evaluate_model_on_dataloader
    test_loss, test_rmse, test_r2, y_true_original_scale, y_pred_original_scale = evaluate_model_on_dataloader(
        best_model, test_loader, final_criterion, device, target_scaler=target_scaler)

    logger.info(f"\n--- 最佳模型在测试集上的最终评估结果 (原始房价尺度) ---")
    logger.info(f"测试集损失 (MSE): {test_loss:.4f}")
    logger.info(f"测试集 RMSE: {test_rmse:.4f}")
    logger.info(f"测试集 R-squared (R²): {test_r2:.4f}")

    # 绘图时使用原始尺度的 y_true 和 y_pred
    plot_predictions_vs_actual(y_true_original_scale, y_pred_original_scale,
                               title=f"{best_model_type}模型: 真实房价 vs 预测房价",
                               save_path=os.path.join(log_dir, f"{best_model_type}_predictions_vs_actual.png"))
    plot_residuals(y_true_original_scale, y_pred_original_scale, title=f"{best_model_type}模型: 预测残差分布",
                   save_path=os.path.join(log_dir, f"{best_model_type}_residuals_distribution.png"))

    # --- 5. 分析层面：模型保存与应用 ---
    logger.info("\n--- 阶段5: 模型保存与应用 ---")
    # 保存最佳模型
    save_model(best_model, f"{model_save_path_prefix}_final.pth")

    # 加载模型进行预测示例
    loaded_model_path = f"{model_save_path_prefix}_final.pth"
    if best_model_type == 'MLP':
        loaded_model = MLPRegressor(
            input_dim=input_dim,
            hidden_layers=[best_params[f'mlp_n_units_l{i}'] for i in range(best_params['mlp_n_layers'])],
            output_dim=output_dim,
            dropout_rate=best_params['mlp_dropout_rate']
        ).to(device)
    elif best_model_type == 'FTTransformer':
        loaded_model = FTTransformer(
            num_features=input_dim,
            num_continuous_features=input_dim,
            dim=best_params['ftt_dim'],
            depth=best_params['ftt_depth'],
            heads=best_params['ftt_heads'],
            attn_dropout=best_params['ftt_attn_dropout'],
            ff_dropout=best_params['ftt_ff_dropout'],
            output_dim=output_dim
        ).to(device)
    else:
        raise ValueError(f"未知模型类型: {best_model_type}")

    load_model(loaded_model, loaded_model_path, device=device)
    loaded_model.eval()

    # 注意：这里的新数据需要与特征工程后的列名和顺序保持一致
    # 因此，我们需要对新数据进行与训练数据相同的预处理和特征工程
    new_data_raw = pd.DataFrame({
        'MedInc': [3.5, 4.0, 2.8, 5.2, 3.1],
        'HouseAge': [25.0, 15.0, 40.0, 10.0, 30.0],
        'AveRooms': [5.5, 6.0, 4.8, 7.0, 5.0],
        'AveBedrms': [1.1, 1.0, 1.2, 0.9, 1.1],
        'Population': [1200.0, 1500.0, 800.0, 2000.0, 1000.0],
        'AveOccup': [2.8, 3.0, 2.5, 3.2, 2.7],
        'Latitude': [34.0, 34.5, 33.8, 34.2, 33.9],
        'Longitude': [-118.0, -118.5, -117.5, -118.2, -117.8]
    })

    logger.info("\n待预测的新数据 (原始):")
    logger.info(new_data_raw)

    # 复制原始数据帧，以便进行特征工程，确保列名一致
    # 这一步是为了获取所有特征工程后的列名，确保新数据在转换时保持一致
    temp_df_for_cols = load_data(data_file_path) # 重新加载原始数据以获取完整的列名和结构
    temp_df_for_cols = handle_missing_values(temp_df_for_cols)
    temp_df_for_cols = handle_outliers(temp_df_for_cols, method='iqr')
    temp_df_for_cols = feature_engineering(temp_df_for_cols)
    feature_columns_for_prediction = temp_df_for_cols.drop(columns=['target']).columns

    # 对新数据应用相同的特征工程
    new_data_processed = feature_engineering(new_data_raw.copy())
    new_data_aligned = new_data_processed.reindex(columns=feature_columns_for_prediction, fill_value=0)


    # 使用之前保存的 feature_scaler 和 target_scaler 进行标准化和反标准化
    new_data_scaled = feature_scaler.transform(new_data_aligned)
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        if best_model_type == 'MLP':
            predictions_scaled = loaded_model(new_data_tensor).cpu().numpy()
        elif best_model_type == 'FTTransformer':
            # FTTransformer 的 forward 方法需要 x_numerical 和 x_categorical
            # 由于当前数据集没有分类特征，x_categorical 传入 None
            predictions_scaled = loaded_model(new_data_tensor, x_categorical=None).cpu().numpy()
        else:
            raise ValueError(f"未知模型类型: {best_model_type}")

    # 反标准化预测结果
    predictions_original_scale = target_scaler.inverse_transform(predictions_scaled).flatten()

    new_data_raw['Predicted_House_Value'] = predictions_original_scale
    logger.info("\n新数据预测结果:")
    logger.info(new_data_raw)

    logger.info("\n--- 机器学习房屋价格预测项目 (深度学习版) 完成 ---")

