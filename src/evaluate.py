#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Model Evaluation

This module handles evaluation and optimization of model predictions.
本模块处理模型预测的评估和优化。

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from typing import Dict, Optional, Tuple, Union, List, Any


def apply_brier_optimal_strategy(predictions_df: pd.DataFrame, 
                               lower_bound: float = 0.3, 
                               upper_bound: float = 0.36,
                               adjustment_factor: float = 0.5) -> pd.DataFrame:
    """
    Apply optimal risk strategy under Brier score.
    
    根据Brier分数应用最优风险策略。
    
    According to theoretical analysis, modifying predictions within certain
    probability ranges can optimize the expected Brier score by applying
    strategic risk adjustments.
    
    根据理论分析，对特定概率范围内的预测进行修改可以通过应用战略风险调整来优化预期的Brier分数。
    
    Parameters:
    -----------
    predictions_df : pandas.DataFrame
        DataFrame containing prediction probabilities
        包含预测概率的数据框
    lower_bound : float, optional (default=0.3)
        Lower threshold for applying risk strategy
        应用风险策略的下限阈值
    upper_bound : float, optional (default=0.36)
        Upper threshold for applying risk strategy
        应用风险策略的上限阈值
    adjustment_factor : float, optional (default=0.5)
        Factor used to adjust predictions in the risk range
        用于调整风险范围内预测的因子
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with adjusted prediction probabilities
        调整后预测概率的数据框
    """
    # Gold medal solution proves: theoretically, taking a risk strategy for games with 33.3% win probability is optimal
    # Max value of f(p) = p(1-p)^2 occurs at p=1/3
    # 金牌解决方案证明：理论上，对于胜率为33.3%的比赛采取风险策略是最优的
    # 函数f(p) = p(1-p)^2的最大值出现在p=1/3处
    
    # Create a view with only the predictions column to avoid copying the entire dataframe
    # 仅创建预测列的视图以避免复制整个数据框
    pred_series = predictions_df['Pred']
    
    # Calculate mask for risky predictions
    # 计算风险预测的掩码
    risky_mask = (pred_series >= lower_bound) & (pred_series <= upper_bound)
    risky_count = np.sum(risky_mask)
    
    if risky_count > 0:
        # Only copy the dataframe if we actually need to modify it
        # 仅在需要修改时才复制数据框
        predictions_copy = predictions_df.copy()
        
        # Apply adjustment using vectorized operations
        # 使用向量化操作应用调整
        risky_indices = risky_mask[risky_mask].index
        current_preds = predictions_copy.loc[risky_indices, 'Pred']
        predictions_copy.loc[risky_indices, 'Pred'] = 0.5 + (current_preds - lower_bound) * adjustment_factor
        
        # Generate strategy application report
        # 生成策略应用报告
        total_predictions = len(predictions_copy)
        
        print(f"Applying optimal risk strategy:")
        print(f"  - Total predictions: {total_predictions}")
        print(f"  - Predictions with risk strategy applied: {risky_count} ({risky_count/total_predictions*100:.2f}%)")
        print(f"  - Adjustment range: [{lower_bound}, {upper_bound}] with factor {adjustment_factor}")
        
        return predictions_copy
    else:
        # If no risky predictions, return the original dataframe without copying
        # 如果没有风险预测，则返回原始数据框而不进行复制
        print("No predictions in the risk range - no adjustments applied")
        return predictions_df


def evaluate_predictions(y_true: Union[List, np.ndarray], 
                        y_pred: Union[List, np.ndarray],
                        confidence_thresholds: Tuple[float, float] = (0.3, 0.7),
                        gender: str = None) -> Dict[str, Any]:
    """
    Evaluate prediction performance using multiple metrics.
    
    使用多种指标评估预测性能。
    
    This function computes various evaluation metrics to assess prediction quality,
    including calibration metrics (Brier score, log loss) and classification metrics
    (accuracy), with special attention to high-confidence predictions.
    
    此函数计算各种评估指标以评估预测质量，包括校准指标（Brier分数，对数损失）和
    分类指标（准确率），并特别关注高置信度预测。
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
        真实的二元标签（0或1）
    y_pred : array-like
        Predicted probabilities in range [0, 1]
        预测的概率值，范围在[0, 1]之间
    confidence_thresholds : tuple of float, optional (default=(0.3, 0.7))
        Lower and upper thresholds for high confidence predictions
        高置信度预测的下限和上限阈值
    gender : str, optional
        Gender of the predictions
        预测的性别
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
        包含评估指标的字典
    """
    # Convert inputs to numpy arrays for consistent processing
    # 将输入转换为numpy数组以进行一致处理
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    
    # Input validation
    # 输入验证
    if not (0 <= np.min(y_pred_np) and np.max(y_pred_np) <= 1):
        raise ValueError("Predicted probabilities must be in the range [0, 1]")
    
    if not np.all(np.isin(y_true_np, [0, 1])):
        raise ValueError("True labels must be binary (0 or 1)")
    
    metrics = {}
    
    # Calculate Brier score (lower is better)
    # 计算Brier分数（越低越好）
    metrics['brier_score'] = brier_score_loss(y_true_np, y_pred_np)
    
    # Calculate log loss (lower is better)
    # 计算对数损失（越低越好）
    # Add small epsilon to avoid log(0) which leads to infinity
    # 添加小的epsilon值以避免log(0)导致的无穷大
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred_np, epsilon, 1 - epsilon)
    metrics['log_loss'] = log_loss(y_true_np, y_pred_clipped)
    
    # Convert predictions to binary decisions using threshold of 0.5
    # 使用0.5的阈值将预测转换为二元决策
    y_pred_binary = (y_pred_np > 0.5).astype(int)
    
    # Calculate accuracy using vectorized operations
    # 使用向量化操作计算准确率
    metrics['accuracy'] = np.mean(y_pred_binary == y_true_np)
    
    # Calculate accuracy for predictions with high confidence
    # 计算高置信度预测的准确率
    low_thresh, high_thresh = confidence_thresholds
    high_conf_mask = (y_pred_np <= low_thresh) | (y_pred_np >= high_thresh)
    high_conf_count = np.sum(high_conf_mask)
    
    # Only calculate high confidence metrics if we have high confidence predictions
    # 仅在有高置信度预测时才计算高置信度指标
    if high_conf_count > 0:
        metrics['high_confidence_accuracy'] = np.mean(
            (y_pred_binary[high_conf_mask] == y_true_np[high_conf_mask])
        )
        metrics['high_confidence_count'] = int(high_conf_count)
        metrics['high_confidence_percentage'] = (high_conf_count / len(y_true_np)) * 100
        
        # Add more detailed metrics for high confidence predictions
        # 为高置信度预测添加更详细的指标
        high_conf_true = y_true_np[high_conf_mask]
        high_conf_pred_binary = y_pred_binary[high_conf_mask]
        
        # Calculate true positives, false positives, etc.
        # 计算真阳性、假阳性等
        true_pos = np.sum((high_conf_true == 1) & (high_conf_pred_binary == 1))
        false_pos = np.sum((high_conf_true == 0) & (high_conf_pred_binary == 1))
        true_neg = np.sum((high_conf_true == 0) & (high_conf_pred_binary == 0))
        false_neg = np.sum((high_conf_true == 1) & (high_conf_pred_binary == 0))
        
        # Calculate precision and recall if possible
        # 计算精确率和召回率（如果可能）
        if (true_pos + false_pos) > 0:
            metrics['high_confidence_precision'] = true_pos / (true_pos + false_pos)
        else:
            metrics['high_confidence_precision'] = None
            
        if (true_pos + false_neg) > 0:
            metrics['high_confidence_recall'] = true_pos / (true_pos + false_neg)
        else:
            metrics['high_confidence_recall'] = None
    else:
        metrics['high_confidence_accuracy'] = None
        metrics['high_confidence_count'] = 0
        metrics['high_confidence_percentage'] = 0
        metrics['high_confidence_precision'] = None
        metrics['high_confidence_recall'] = None
    
    # 添加性别标识到输出信息
    gender_str = f" ({gender})" if gender else ""
    
    # 打印指标报告
    # 打印指标报告 / Print metrics report
    print(f"\n预测评估指标{gender_str}:")
    print(f"  - Brier分数: {metrics['brier_score']:.6f} (越低越好)")
    print(f"  - 对数损失: {metrics['log_loss']:.6f} (越低越好)")
    print(f"  - 准确率: {metrics['accuracy']:.4f}")
    
    if metrics['high_confidence_accuracy'] is not None:
        print(f"  - High Confidence Accuracy: {metrics['high_confidence_accuracy']:.4f} "
              f"({metrics['high_confidence_count']} predictions, "
              f"{metrics['high_confidence_percentage']:.1f}% of total)")
        
        if metrics['high_confidence_precision'] is not None:
            print(f"  - High Confidence Precision: {metrics['high_confidence_precision']:.4f}")
        if metrics['high_confidence_recall'] is not None:
            print(f"  - High Confidence Recall: {metrics['high_confidence_recall']:.4f}")
    
    # 添加ROC AUC指标 - 确保处理可能的异常情况
    try:
        # 计算ROC AUC，但确保有足够的样本和至少两个类别
        if len(np.unique(y_true_np)) >= 2 and len(y_true_np) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true_np, y_pred_np)
        else:
            # 当数据不满足条件时，提供默认值
            metrics['roc_auc'] = 0.5  # 随机分类器的AUC
            print("警告: 无法计算ROC AUC，样本或类别不足")
    except Exception as e:
        # 捕获任何计算ROC AUC时的异常
        metrics['roc_auc'] = 0.5  # 提供默认值
        print(f"计算ROC AUC时出错: {str(e)}")
    
    return metrics


def visualization_prediction_distribution(y_pred: Union[List, np.ndarray], 
                                         y_true: Optional[Union[List, np.ndarray]] = None, 
                                         save_path: Optional[str] = None,
                                         show_plot: bool = True,
                                         fig_size: Tuple[int, int] = (12, 6),
                                         key_points: List[float] = [0.3, 0.5, 0.7]) -> plt.Figure:
    """
    Visualize the distribution of predictions.
    
    可视化预测分布。
    
    Creates visualization of prediction probability distributions, optionally comparing
    with true outcomes. This can help identify calibration issues or biases in the
    prediction model.
    
    创建预测概率分布的可视化，可选择与真实结果进行比较。这有助于识别预测模型中的校准问题或偏差。
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted probabilities in range [0, 1]
        预测的概率值，范围在[0, 1]之间
    y_true : array-like, optional
        True binary labels (0 or 1)
        真实的二元标签（0或1）
    save_path : str, optional
        Path to save the visualization
        保存可视化结果的路径
    show_plot : bool, optional (default=True)
        Whether to display the plot (set to False for batch processing)
        是否显示图形（批处理时设为False）
    fig_size : tuple of int, optional (default=(12, 6))
        Figure size (width, height) in inches
        图形大小（宽度，高度），单位为英寸
    key_points : list of float, optional (default=[0.3, 0.5, 0.7])
        Key probability points to highlight with vertical lines
        用垂直线突出显示的关键概率点
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
        生成的图形对象
    """
    # Convert inputs to numpy arrays
    # 将输入转换为numpy数组
    y_pred_np = np.asarray(y_pred)
    
    # Input validation
    # 输入验证
    if not (0 <= np.min(y_pred_np) and np.max(y_pred_np) <= 1):
        raise ValueError("Predicted probabilities must be in the range [0, 1]")
    
    if y_true is not None:
        y_true_np = np.asarray(y_true)
        if not np.all(np.isin(y_true_np, [0, 1])):
            raise ValueError("True labels must be binary (0 or 1)")
    
    # Create figure with specified size
    # 创建指定大小的图形
    fig = plt.figure(figsize=fig_size)
    
    # Plot prediction distribution
    # 绘制预测分布
    ax1 = plt.subplot(1, 2, 1)
    sns.histplot(y_pred_np, bins=20, kde=True, ax=ax1)
    ax1.set_title('Prediction Distribution')
    ax1.set_xlabel('Predicted Win Probability')
    ax1.set_ylabel('Frequency')
    
    # Add vertical lines at key probability points
    # 在关键概率点添加垂直线
    colors = ['r', 'g', 'r']  # Default colors for 0.3, 0.5, 0.7
    if len(key_points) != len(colors):
        # Generate colors dynamically if key_points length doesn't match default colors
        # 如果key_points长度与默认颜色不匹配，则动态生成颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(key_points)))
    
    for i, point in enumerate(key_points):
        ax1.axvline(point, color=colors[i % len(colors)], linestyle='--', 
                   alpha=0.5, label=f'{point}')
    ax1.legend()
    
    # Calculate summary statistics for the prediction distribution
    # 计算预测分布的摘要统计信息
    mean_pred = np.mean(y_pred_np)
    median_pred = np.median(y_pred_np)
    std_pred = np.std(y_pred_np)
    
    # Add text with summary statistics
    # 添加包含摘要统计信息的文本
    stats_text = f"Mean: {mean_pred:.3f}\nMedian: {median_pred:.3f}\nStd Dev: {std_pred:.3f}"
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # If true values are provided, plot predictions by outcome
    # 如果提供了真实值，则按结果绘制预测
    if y_true is not None:
        ax2 = plt.subplot(1, 2, 2)
        
        # Create a more efficient data structure for plotting
        # 创建更高效的数据结构用于绘图
        # Using structured arrays directly instead of DataFrame for better performance
        # 直接使用结构化数组而不是DataFrame以获得更好的性能
        df = pd.DataFrame({'y_true': y_true_np, 'y_pred': y_pred_np})
        
        # Create boxplot of predictions by actual outcome
        # 按实际结果创建预测的箱线图
        sns.boxplot(x='y_true', y='y_pred', data=df, ax=ax2)
        
        # Add individual points on top of boxplot with reduced opacity for better visibility
        # 在箱线图上添加单个点，降低不透明度以提高可见性
        sns.stripplot(x='y_true', y='y_pred', data=df, 
                     size=3, color='black', alpha=0.2, jitter=True, ax=ax2)
        
        ax2.set_title('Predictions by Actual Outcome')
        ax2.set_xlabel('Actual Outcome (0=Loss, 1=Win)')
        ax2.set_ylabel('Predicted Win Probability')
        
        # Calculate summary statistics by outcome
        # 按结果计算摘要统计信息
        for outcome in [0, 1]:
            outcome_preds = y_pred_np[y_true_np == outcome]
            if len(outcome_preds) > 0:
                mean_val = np.mean(outcome_preds)
                ax2.text(outcome, 0.02, f'Mean: {mean_val:.3f}', 
                        ha='center', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    # 如果提供了路径，则保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Only show the plot if requested
    # 仅在请求时显示图形
    if show_plot:
        plt.show()
    
    return fig


def calibration_curve(y_true: Union[List, np.ndarray], 
                     y_pred: Union[List, np.ndarray], 
                     n_bins: int = 10,
                     save_path: Optional[str] = None,
                     show_plot: bool = True) -> Dict[str, np.ndarray]:
    """
    Generate and visualize prediction calibration curve.
    
    生成并可视化预测校准曲线。
    
    A calibration curve plots predicted probabilities against actual outcome rates.
    A well-calibrated model should produce points that lie close to the diagonal.
    
    校准曲线绘制预测概率与实际结果率之间的关系。校准良好的模型应该产生接近对角线的点。
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
        真实的二元标签（0或1）
    y_pred : array-like
        Predicted probabilities in range [0, 1]
        预测的概率值，范围在[0, 1]之间
    n_bins : int, optional (default=10)
        Number of bins to use for the calibration curve
        用于校准曲线的分箱数量
    save_path : str, optional
        Path to save the visualization
        保存可视化结果的路径
    show_plot : bool, optional (default=True)
        Whether to display the plot (set to False for batch processing)
        是否显示图形（批处理时设为False）
        
    Returns:
    --------
    dict
        Dictionary containing calibration data:
        包含校准数据的字典：
        'bin_centers': Centers of the probability bins
                       概率分箱的中心
        'bin_actual': Actual outcome rate for each bin
                     每个分箱的实际结果率
        'bin_counts': Number of predictions in each bin
                     每个分箱中的预测数量
        'bin_errors': Standard error for each bin's estimate
                     每个分箱估计的标准误差
    """
    # Convert inputs to numpy arrays
    # 将输入转换为numpy数组
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Input validation
    # 输入验证
    if not (0 <= np.min(y_pred_np) and np.max(y_pred_np) <= 1):
        raise ValueError("Predicted probabilities must be in the range [0, 1]")
    
    if not np.all(np.isin(y_true_np, [0, 1])):
        raise ValueError("True labels must be binary (0 or 1)")
    
    if n_bins < 2:
        raise ValueError("Number of bins must be at least 2")
    
    # Create bin edges and calculate bin centers
    # 创建分箱边界并计算分箱中心
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Use vectorized operations to calculate bin indices for each prediction
    # 使用向量化操作计算每个预测的分箱索引
    bin_indices = np.digitize(y_pred_np, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure indices don't exceed bins
    
    # Initialize arrays for bin statistics
    # 初始化分箱统计数组
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_errors = np.zeros(n_bins)
    
    # Calculate bin sums and counts using vectorized operations
    # 使用向量化操作计算分箱总和和计数
    for i in range(n_bins):
        mask = (bin_indices == i)
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_sums[i] = np.sum(y_true_np[mask])
    
    # Calculate actual outcome rates and standard errors
    # 计算实际结果率和标准误差
    bin_actual = np.zeros(n_bins)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_actual[i] = bin_sums[i] / bin_counts[i]
            # Calculate standard error using binomial formula √(p*(1-p)/n)
            # 使用二项式公式√(p*(1-p)/n)计算标准误差
            if bin_counts[i] > 1:  # Avoid division by zero
                bin_errors[i] = np.sqrt(bin_actual[i] * (1 - bin_actual[i]) / bin_counts[i])
    
    fig = plt.figure(figsize=(10, 6))
    
    # Plot calibration curve with error bars
    # 绘制带误差线的校准曲线
    plt.errorbar(bin_centers, bin_actual, yerr=bin_errors, fmt='o-', 
                label='Calibration Curve', ecolor='lightgray', capsize=3)
    
    # Add perfect calibration line
    # 添加完美校准线
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    
    # Add bin counts as text
    # 添加分箱计数为文本
    for i in range(n_bins):
        if bin_counts[i] > 0:  # Only add text for bins with data
            plt.text(bin_centers[i], bin_actual[i] + max(0.03, bin_errors[i] + 0.01), 
                    f'n={int(bin_counts[i])}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Calculate overall calibration error metrics
    # 计算总体校准误差指标
    mce = np.sum(np.abs(bin_actual - bin_centers) * (bin_counts / np.sum(bin_counts)))
    rmsce = np.sqrt(np.sum(np.square(bin_actual - bin_centers) * (bin_counts / np.sum(bin_counts))))
    
    # Add calibration error metrics to the plot
    # 将校准误差指标添加到图中
    plt.text(0.05, 0.95, f'MCE: {mce:.4f}\nRMSCE: {rmsce:.4f}', 
            transform=plt.gca().transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Outcome Rate')
    plt.title('Prediction Calibration Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure if path is provided
    # 如果提供了路径，则保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Only show the plot if requested
    # 仅在请求时显示图形
    if show_plot:
        plt.show()
    
    # Return calibration data
    # 返回校准数据
    return {
        'bin_centers': bin_centers,
        'bin_actual': bin_actual,
        'bin_counts': bin_counts,
        'bin_errors': bin_errors,
        'metrics': {
            'mce': mce,  # Mean Calibration Error
            'rmsce': rmsce  # Root Mean Squared Calibration Error
        }
    }