import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import numpy as np
import os
import logging
import io

# 获取日志记录器实例
logger = logging.getLogger(__name__)

# 设置Matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    """
    加载加利福尼亚房屋价格数据集。
    如果文件不存在，则从scikit-learn加载并保存为CSV。
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"数据从 '{file_path}' 加载成功！")
    except FileNotFoundError:
        logger.info(f"文件 '{file_path}' 未找到，正在从scikit-learn下载并保存...")
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df['target'] = housing.target  # 目标变量（房价）通常在 .target 属性中
        df.to_csv(file_path, index=False)
        logger.info(f"数据已下载并保存到 '{file_path}'。")
    return df


def initial_data_exploration(df):
    """
    对加载的数据进行初步探索。
    包括查看基本信息、描述性统计和缺失值检查。
    """
    logger.info("\n--- 数据概览 ---")
    logger.info("数据前5行:")
    logger.info(df.head())

    logger.info("\n数据基本信息:")
    # 使用 StringIO 捕获 df.info() 的输出，然后通过 logger.info 打印
    buffer = io.StringIO()
    df.info(buf=buffer)
    logger.info(buffer.getvalue())

    logger.info("\n描述性统计:")
    logger.info(df.describe())

    logger.info("\n缺失值检查:")
    logger.info(df.isnull().sum())

    logger.info("\n数据初步探索完成。")


def visualize_initial_data(df, save_dir='logs'):
    """
    绘制数据分布和特征与目标变量的散点图，并保存。
    """
    logger.info("\n--- 数据初步可视化 ---")
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在

    # 绘制目标变量（房价）的分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['target'], kde=True, bins=30, color='skyblue')
    plt.title('房屋价格分布', fontsize=16)
    plt.xlabel('房价 (十万美元)', fontsize=12)
    plt.ylabel('频率 / 密度', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plot_path = os.path.join(save_dir, '房屋价格分布.png')
    plt.savefig(plot_path)  # 保存图表
    logger.info(f"图表已保存到: {plot_path}")
    plt.show()  # 显示图表

    # 绘制特征与目标变量（房价）之间的散点图
    features = df.drop(columns=['target']).columns
    n_cols_scatter = 3
    n_rows_scatter = (len(features) + n_cols_scatter - 1) // n_cols_scatter
    plt.figure(figsize=(n_cols_scatter * 6, n_rows_scatter * 5))

    for i, col in enumerate(features):
        plt.subplot(n_rows_scatter, n_cols_scatter, i + 1)
        sns.scatterplot(x=df[col], y=df['target'], alpha=0.6, color='darkblue')
        plt.title(f'{col} vs. 房价', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('房价', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, '特征与房价散点图.png')
    plt.savefig(plot_path)  # 保存图表
    logger.info(f"图表已保存到: {plot_path}")
    plt.show()  # 显示图表

    logger.info("\n数据初步可视化完成。")


def handle_missing_values(df):
    """
    处理缺失值。对于加利福尼亚房屋数据集，通常没有缺失值，
    但这里提供一个使用 KNNImputer 的通用方法。
    """
    logger.info("\n--- 缺失值处理 ---")
    if df.isnull().sum().sum() > 0:
        logger.info("发现缺失值，正在使用 KNNImputer 填充...")
        numeric_cols = df.select_dtypes(include=np.number).columns
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        logger.info("缺失值填充完成。")
    else:
        logger.info("数据集中没有发现缺失值。")
    return df


def handle_outliers(df, method='iqr'):
    """
    处理异常值。
    method: 'iqr' (四分位距法)
    """
    logger.info("\n--- 异常值处理 ---")
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

    if 'target' in numeric_cols:
        numeric_cols.remove('target')  # 目标变量通常不进行异常值处理

    outlier_count = 0
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 识别异常值
        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        outlier_count += len(outliers)

        # 替换异常值（这里选择替换为边界值，也可以选择删除行或使用中位数等）
        df_cleaned[col] = np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col])
        df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound, df_cleaned[col])

    if outlier_count > 0:
        logger.info(f"已使用 IQR 方法处理 {outlier_count} 个异常值 (替换为边界值)。")
    else:
        logger.info("未发现明显异常值或已处理。")
    return df_cleaned


def feature_engineering(df):
    """
    进行特征工程，创建新的特征。
    创新点：可以根据对房屋数据的理解，创建更复杂的特征。
    """
    logger.info("\n--- 特征工程 ---")
    df_fe = df.copy()

    # 示例1：每户平均卧室数比例 (如果 AveRooms 和 AveBedrms 存在)
    if 'AveRooms' in df_fe.columns and 'AveBedrms' in df_fe.columns:
        df_fe['Bedrms_Per_Room'] = df_fe['AveBedrms'] / df_fe['AveRooms']
        logger.info("已创建新特征 'Bedrms_Per_Room'。")

    # 示例2：人口密度 (如果 Population, Latitude, Longitude 存在)
    # 这是一个简化的人口密度，实际需要更复杂的地理信息
    if 'Population' in df_fe.columns and 'Latitude' in df_fe.columns and 'Longitude' in df_fe.columns:
        # 假设一个简单的区域面积代理，例如基于经纬度范围
        # 实际应用中，这需要更精确的地理面积计算
        # 这里仅作示例，表示可以结合多个特征创建新特征
        df_fe['Population_Density'] = df_fe['Population'] / (df_fe['Latitude'] * df_fe['Longitude']).abs()
        # 避免除以0或极小值
        df_fe['Population_Density'] = df_fe['Population_Density'].replace([np.inf, -np.inf], np.nan).fillna(0)
        logger.info("已创建新特征 'Population_Density'。")

    # 示例3：组合地理位置特征 (经纬度组合)
    if 'Latitude' in df_fe.columns and 'Longitude' in df_fe.columns:
        df_fe['Lat_x_Lon'] = df_fe['Latitude'] * df_fe['Longitude']
        logger.info("已创建新特征 'Lat_x_Lon'。")

    logger.info("特征工程完成。")
    return df_fe


def split_and_scale_data(df, target_column='target', test_size=0.2, random_state=42, scaler_type='standard',
                         feature_scaler=None):
    """
    将数据划分为训练集和测试集，并进行特征缩放。
    同时对目标变量进行标准化。
    scaler_type: 'standard' (StandardScaler), 'minmax' (MinMaxScaler) 或 'none' (不缩放)。
    feature_scaler: 如果 scaler_type='none' 且需要使用已有的feature_scaler进行转换，则传入该feature_scaler。
    """
    logger.info("\n--- 数据划分与特征缩放 ---")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 特征缩放
    if scaler_type == 'standard':
        if feature_scaler is None:  # 第一次缩放时创建新的scaler
            feature_scaler = StandardScaler()
            logger.info("选择 StandardScaler 进行特征缩放。")
        else:  # 如果传入了feature_scaler，则直接使用
            logger.info("使用传入的 StandardScaler 进行特征缩放。")
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
    elif scaler_type == 'minmax':
        if feature_scaler is None:  # 第一次缩放时创建新的scaler
            feature_scaler = MinMaxScaler()
            logger.info("选择 MinMaxScaler 进行特征缩放。")
        else:  # 如果传入了feature_scaler，则直接使用
            logger.info("使用传入的 MinMaxScaler 进行特征缩放。")
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
    elif scaler_type == 'none':
        # 不进行缩放，直接返回原始数据
        logger.info("不进行特征缩放。")
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
        feature_scaler = None  # 在不缩放时，不返回feature_scaler
    else:
        raise ValueError("scaler_type 必须是 'standard', 'minmax' 或 'none'")

    # 目标变量缩放 (始终使用 StandardScaler 进行目标变量缩放)
    # 注意：y_train.values.reshape(-1, 1) 是因为 StandardScaler 期望二维输入
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    logger.info("目标变量已使用 StandardScaler 进行缩放。")

    # 将缩放后的数据转换回DataFrame，保持列名，方便后续处理
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    # 目标变量也转换回 Series，保持索引
    y_train_scaled_series = pd.Series(y_train_scaled, index=y_train.index)
    y_test_scaled_series = pd.Series(y_test_scaled, index=y_test.index)

    logger.info(f"训练集特征形状: {X_train_scaled_df.shape}")
    logger.info(f"测试集特征形状: {X_test_scaled_df.shape}")
    logger.info(f"训练集目标形状: {y_train_scaled_series.shape}")
    logger.info(f"测试集目标形状: {y_test_scaled_series.shape}")
    logger.info("数据划分与特征缩放完成。")

    return X_train_scaled_df, X_test_scaled_df, y_train_scaled_series, y_test_scaled_series, feature_scaler, target_scaler

