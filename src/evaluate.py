#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Model Evaluation

This module handles evaluation and optimization of model predictions.
本模块处理模型预测的评估和优化。

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
Version: 2.1 ### 增加了预测所有可能的球队对阵的假设结果
Version: 3.0 ### 增加了cudf的支持
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
from typing import Dict, Optional, Tuple, Union, List, Any
import cupy as cp
from contextlib import contextmanager
from utils import gpu_context, to_gpu, to_cpu


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
    
    # 创建掩码而不是Series视图
    risky_mask = (predictions_df['Pred'] >= lower_bound) & (predictions_df['Pred'] <= upper_bound)
    risky_count = risky_mask.sum()
    
    if risky_count > 0:
        # 只复制需要修改的部分
        predictions_copy = predictions_df.copy()
        
        # 直接使用布尔索引修改
        risky_indices = predictions_copy.index[risky_mask]
        current_preds = predictions_copy.loc[risky_indices, 'Pred']
        predictions_copy.loc[risky_indices, 'Pred'] = 0.5 + (current_preds - lower_bound) * adjustment_factor
        
        # 生成报告
        total_predictions = len(predictions_copy)
        print(f"Applying optimal risk strategy:")
        print(f"  - Total predictions: {total_predictions}")
        print(f"  - Predictions with risk strategy applied: {risky_count} ({risky_count/total_predictions*100:.2f}%)")
        print(f"  - Adjustment range: [{lower_bound}, {upper_bound}] with factor {adjustment_factor}")
        
        return predictions_copy
    else:
        # 没有修改时返回原始数据
        print("No predictions in the risk range - no adjustments applied")
        return predictions_df


def evaluate_predictions(y_true, y_pred, confidence_thresholds=(0.3, 0.7),
                       gender=None, use_gpu=True):
    """评估预测结果，支持GPU加速"""
    with gpu_context(use_gpu) as gpu_available:
        if gpu_available:
            try:
                # 将数据转移到GPU
                y_true_gpu = to_gpu(y_true)
                y_pred_gpu = to_gpu(y_pred)
                
                # 计算评估指标
                metrics = {}
                
                # Brier分数
                metrics['brier_score'] = float(cp.mean((y_pred_gpu - y_true_gpu) ** 2))
                
                # 对数损失
                epsilon = 1e-15
                y_pred_clipped = cp.clip(y_pred_gpu, epsilon, 1 - epsilon)
                metrics['log_loss'] = float(-cp.mean(
                    y_true_gpu * cp.log(y_pred_clipped) + 
                    (1 - y_true_gpu) * cp.log(1 - y_pred_clipped)
                ))
                
                try:
                    # 尝试使用更稳定的GPU ROC AUC实现
                    # 排序预测值
                    sort_indices = cp.argsort(y_pred_gpu)[::-1]
                    sorted_y_true = y_true_gpu[sort_indices]
                    sorted_y_pred = y_pred_gpu[sort_indices]
                    
                    # 去除重复的预测值，以避免计算错误
                    unique_pred_values, unique_indices = cp.unique(sorted_y_pred, return_index=True)
                    
                    if len(unique_pred_values) > 1:
                        # 计算TPR和FPR
                        n_pos = cp.sum(y_true_gpu)
                        n_neg = len(y_true_gpu) - n_pos
                        
                        if n_pos > 0 and n_neg > 0:
                            # 计算累积TPR和FPR
                            tpr = cp.zeros(len(unique_indices) + 1)
                            fpr = cp.zeros(len(unique_indices) + 1)
                            
                            for i, idx in enumerate(unique_indices):
                                tpr[i+1] = cp.sum(sorted_y_true[:idx+1]) / n_pos
                                fpr[i+1] = cp.sum(1 - sorted_y_true[:idx+1]) / n_neg
                            
                            # 计算AUC
                            metrics['roc_auc'] = float(cp.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2)
                        else:
                            # 正例或负例为0，AUC无意义
                            metrics['roc_auc'] = 0.5
                    else:
                        # 所有预测值相同，AUC无意义
                        metrics['roc_auc'] = 0.5
                except Exception as e:
                    print(f"GPU ROC AUC计算失败: {e}，使用sklearn计算")
                    # 退回到CPU计算
                    y_true_cpu = to_cpu(y_true_gpu)
                    y_pred_cpu = to_cpu(y_pred_gpu)
                    from sklearn.metrics import roc_auc_score
                    metrics['roc_auc'] = roc_auc_score(y_true_cpu, y_pred_cpu)
                
                return metrics
            except Exception as e:
                print(f"GPU评估失败: {e}")
                use_gpu = False
            finally:
                # 确保释放GPU内存
                cp.get_default_memory_pool().free_all_blocks()
    
    # 如果GPU不可用或失败，使用CPU评估
    # ... 原有的CPU评估代码 ...
    
    # 需要添加完整的CPU实现:
    metrics = {}
    
    # Brier分数
    from sklearn.metrics import brier_score_loss
    metrics['brier_score'] = brier_score_loss(y_true, y_pred)
    
    # 对数损失
    from sklearn.metrics import log_loss
    metrics['log_loss'] = log_loss(y_true, y_pred)
    
    # ROC AUC
    from sklearn.metrics import roc_auc_score
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
    
    return metrics


def visualization_prediction_distribution(y_pred: Union[List, np.ndarray], 
                                         y_true: Optional[Union[List, np.ndarray]] = None, 
                                         save_path: Optional[str] = None,
                                         show_plot: bool = True,
                                         fig_size: Tuple[int, int] = (12, 6),
                                         key_points: List[float] = [0.3, 0.5, 0.7],
                                         title: Optional[str] = None) -> plt.Figure:
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
    title : str, optional (default=None)
        Title for the plot
        图表标题
        
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
            # 如果y_true不是二元值，将其转换为None而不是抛出错误
            # If y_true is not binary, convert it to None instead of raising an error
            print("警告：真实标签包含非二元值(0/1)，将忽略真实标签进行可视化。")
            print("Warning: True labels contain non-binary values (0/1), ignoring true labels for visualization.")
            y_true = None
    
    # Create figure with specified size
    # 创建指定大小的图形
    fig = plt.figure(figsize=fig_size)
    
    # Plot prediction distribution
    # 绘制预测分布
    ax1 = plt.subplot(1, 2 if y_true is not None else 1, 1)
    sns.histplot(y_pred_np, bins=20, kde=True, ax=ax1)
    ax1.set_title('Prediction Distribution')
    ax1.set_xlabel('Predicted Win Probability')
    ax1.set_ylabel('Frequency')
    
    # Add main title if provided
    # 如果提供了主标题，则添加
    if title:
        plt.suptitle(title, fontsize=16, y=1.05)
    
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
        
        # 在上面的代码中，我们可能已经将y_true设置为None，所以需要重新获取y_true_np
        # We might have set y_true to None above, so we need to get y_true_np again
        y_true_np = np.asarray(y_true)
        
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
                     show_plot: bool = True,
                     title: Optional[str] = None) -> Dict[str, np.ndarray]:
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
    title : str, optional (default=None)
        Title for the plot
        图表标题
        
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
    plt.title(title or 'Prediction Calibration Curve')
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