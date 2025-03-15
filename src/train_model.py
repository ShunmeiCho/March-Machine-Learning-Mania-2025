#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - 模型训练

This module handles training and optimizing the prediction model.
本模块负责训练和优化预测模型。

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Union, Any
from joblib import Memory
from functools import partial
import concurrent.futures  # 为并行处理添加导入

# 添加中文字体支持
# Add Chinese font support
try:
    import matplotlib.font_manager as fm
    # 检查系统是否有中文字体
    chinese_fonts = [f for f in fm.findSystemFonts() if 'chinese' in f.lower() or 'cjk' in f.lower() or 'noto' in f.lower()]
    if chinese_fonts:
        # 使用找到的第一个中文字体
        plt.rcParams['font.family'] = fm.FontProperties(fname=chinese_fonts[0]).get_name()
    else:
        # 如果没有找到中文字体，使用一些通用的无衬线字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("已配置中文字体支持 / Chinese font support configured")
except Exception as e:
    print(f"配置中文字体支持时出错: {e} / Error configuring Chinese font support: {e}")

# 设置缓存目录
# Set up cache directory
CACHE_DIR = './cache'
memory = Memory(CACHE_DIR, verbose=0)

# 使用缓存装饰函数 / Decorate functions with cache
@memory.cache
def cached_merge_features(tourney_train, team_stats, seed_features, matchup_history, progression_probs=None, gender='men'):
    """
    合并特征的缓存版本
    Cached version of feature merging
    
    当数据集不变时避免重复计算
    Avoids recalculating when datasets are unchanged
    
    Args:
        gender: 'men' 或 'women'，指定处理哪种性别的数据
               'men' or 'women', specifies which gender's data to process
    """
    print(f"使用缓存版本的特征合并... ({gender}) / Using cached version of feature merging... ({gender})")
    return merge_features(tourney_train, team_stats, seed_features, matchup_history, progression_probs, gender=gender)


def check_feature_correlations(X_train, threshold=0.95, enable_check=True):
    """
    检查特征之间的相关性并移除高度相关的特征
    Check correlations between features and remove highly correlated ones
    
    参数:
        X_train: 特征矩阵，可以是 pandas DataFrame 或 numpy array
        threshold: 相关性阈值
        enable_check: 是否启用检查
        
    返回:
        X_train_reduced: 移除高度相关特征后的矩阵
        selected_columns: 保留的列名列表
    """
    if not enable_check:
        print("跳过特征相关性检查 / Skipping feature correlation check")
        # 根据输入类型返回适当的列名
        if isinstance(X_train, pd.DataFrame):
            return X_train, list(X_train.columns)
        else:
            return X_train, [f'feature_{i}' for i in range(X_train.shape[1])]
    
    print("检查特征相关性... / Checking feature correlations...")
    
    # 记录原始输入类型
    is_numpy = isinstance(X_train, np.ndarray)
    
    # 如果是 numpy 数组，转换为 DataFrame
    if is_numpy:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
    else:
        X_train_df = X_train
        feature_names = X_train.columns.tolist()
    
    # 计算相关性矩阵
    corr_matrix = X_train_df.corr().abs()
    
    # 获取上三角矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找出高度相关的特征
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if len(to_drop) > 0:
        print(f"发现 {len(to_drop)} 个高度相关特征: / Found {len(to_drop)} highly correlated features:")
        for col in to_drop:
            correlated_with = upper.index[upper[col] > threshold].tolist()
            for other_col in correlated_with:
                print(f"  - {col} 和 {other_col} 相关性: {corr_matrix.loc[col, other_col]:.4f} / correlation: {corr_matrix.loc[col, other_col]:.4f}")
        
        print(f"从训练数据中删除 {len(to_drop)} 个高度相关特征 / Removing {len(to_drop)} highly correlated features from training data")
        
        # 从DataFrame中删除列
        X_train_reduced_df = X_train_df.drop(columns=to_drop)
        
        # 获取保留的列名
        selected_columns = X_train_reduced_df.columns.tolist()
        
        # 根据原始输入类型返回相应格式
        if is_numpy:
            return X_train_reduced_df.values, selected_columns
        else:
            return X_train_reduced_df, selected_columns
    else:
        print("未发现高度相关特征 / No highly correlated features found")
        if is_numpy:
            return X_train, feature_names
        else:
            return X_train, feature_names


def add_favorite_longshot_features(X_df):
    """
    添加热门-冷门偏差纠正特征
    Add favorite-longshot bias correction features
    
    Args:
        X_df: 输入特征数据框
        
    Returns:
        X_df: 增加了偏差纠正特征的数据框
    """
    # 如果没有种子特征，返回原始数据 / If no seed features, return original data
    if 'team1_seed' not in X_df.columns or 'team2_seed' not in X_df.columns:
        return X_df
    
    # 使用向量化操作替代循环以提高性能 / Use vectorized operations for better performance
    
    # 1. 种子强度校正 / Seed strength correction
    X_df.loc[:, 'team1_seed_strength'] = 1 / (X_df['team1_seed'] + 0.5)
    X_df.loc[:, 'team2_seed_strength'] = 1 / (X_df['team2_seed'] + 0.5)
    
    # 2. 添加校正因子 / Add correction factors
    X_df.loc[:, 'seed_strength_diff'] = X_df['team1_seed_strength'] - X_df['team2_seed_strength']
    
    # 3. 考虑常规赛中被低估的强队 / Account for stronger teams being underestimated
    # 热门-冷门偏差校正：强队（种子数低）往往被低估
    # Favorite-longshot bias correction: strong teams (low seed numbers) tend to be underestimated
    X_df.loc[:, 'team1_bias_correction'] = 0.05 * np.log(X_df['team1_seed'] + 1)
    X_df.loc[:, 'team2_bias_correction'] = 0.05 * np.log(X_df['team2_seed'] + 1)
    X_df.loc[:, 'bias_correction_diff'] = X_df['team2_bias_correction'] - X_df['team1_bias_correction']
    
    # 4. 校正胜率和得分差异 / Correct win rate and point difference
    if 'win_rate_diff' in X_df.columns:
        X_df.loc[:, 'adjusted_win_rate_diff'] = X_df['win_rate_diff'] + X_df['bias_correction_diff'] * 0.2
    
    if 'point_diff_diff' in X_df.columns:
        X_df.loc[:, 'adjusted_point_diff_diff'] = X_df['point_diff_diff'] + X_df['bias_correction_diff'] * 3.0
    
    # 5. 添加种子交互特征 / Add seed interaction features
    X_df.loc[:, 'seed_product'] = X_df['team1_seed'] * X_df['team2_seed']
    X_df.loc[:, 'seed_ratio'] = X_df['team1_seed'] / X_df['team2_seed'].replace(0, 0.1)
    
    # 6. 锦标赛压力特征 / Tournament pressure features
    X_df.loc[:, 'team1_pressure'] = 1 / (X_df['team1_seed'] ** 0.8)
    X_df.loc[:, 'team2_pressure'] = 1 / (X_df['team2_seed'] ** 0.8)
    X_df.loc[:, 'pressure_diff'] = X_df['team1_pressure'] - X_df['team2_pressure']
    
    return X_df


def process_matchup(matchup, team_stats, seed_features, matchup_history, progression_probs, gender='men'):
    """
    处理单个比赛的对阵数据，提取相关特征
    Process a single matchup, extracting relevant features
    
    Args:
        matchup: 比赛对阵数据
        team_stats: 团队统计数据
        seed_features: 种子特征
        matchup_history: 对战历史
        progression_probs: 晋级概率
        gender: 'men' 或 'women'，指定处理哪种性别的数据
               'men' or 'women', specifies which gender's data to process
        
    Returns:
        tuple: (特征字典, 目标变量)
    """
    season = matchup['Season']
    team1 = matchup['Team1']
    team2 = matchup['Team2']
    target = matchup['Team1_Win']
    round_num = matchup.get('Round', 0)
    day_num = matchup.get('DayNum', 0)
    
    # 如果缺少任一队伍的统计信息，跳过 / Skip if we don't have statistics for either team
    if (season not in team_stats or 
        team1 not in team_stats[season] or 
        team2 not in team_stats[season] or
        season not in seed_features or
        team1 not in seed_features[season] or
        team2 not in seed_features[season]):
        return None, None
    
    # 获取团队统计信息 / Get team statistics
    team1_stats = team_stats[season][team1]
    team2_stats = team_stats[season][team2]
    
    # 获取种子信息 / Get seed information
    team1_seed = seed_features[season][team1]
    team2_seed = seed_features[season][team2]
    
    # 初始化特征字典 / Initialize feature dictionary
    features = {}
    
    # 添加gender特征 / Add gender feature
    features['gender'] = 1 if gender == 'men' else 0
    
    # 添加team1统计信息（带前缀） / Add team1 statistics (with prefix)
    for stat, value in team1_stats.items():
        features[f'team1_{stat}'] = value
    
    # 添加team2统计信息（带前缀） / Add team2 statistics (with prefix)
    for stat, value in team2_stats.items():
        features[f'team2_{stat}'] = value
    
    # 添加种子特征 / Add seed features
    features['team1_seed'] = team1_seed['seed_num']
    features['team2_seed'] = team2_seed['seed_num']
    features['seed_diff'] = team1_seed['seed_num'] - team2_seed['seed_num']
    
    # 添加对战历史（如果可用） / Add matchup history (if available)
    matchup_key = tuple(sorted([team1, team2]))
    
    # 校正：只考虑比赛日期前的对战历史 / Correction: Only consider matchup history before the game date
    if season in matchup_history:
        if matchup_key in matchup_history[season]:
            matchup_data = matchup_history[season][matchup_key]
            features['matchup_games'] = matchup_data['games']
            features['matchup_winrate_team1'] = matchup_data['wins_team1'] / matchup_data['games'] if matchup_data['games'] > 0 else 0.5
            features['matchup_avg_point_diff'] = matchup_data['avg_point_diff']
        else:
            features['matchup_games'] = 0
            features['matchup_winrate_team1'] = 0.5  # 如无历史默认为50% / Default to 50% if no history
            features['matchup_avg_point_diff'] = 0
    else:
        features['matchup_games'] = 0
        features['matchup_winrate_team1'] = 0.5
        features['matchup_avg_point_diff'] = 0
    
    # 添加金牌解决方案见解：晋级概率差异 / Add gold medal solution insight: progression probability differences
    if progression_probs is not None and season in progression_probs:
        if team1 in progression_probs[season] and team2 in progression_probs[season]:
            team1_prog = progression_probs[season][team1]
            team2_prog = progression_probs[season][team2]
            
            # 添加原始晋级概率 / Add raw progression probabilities
            for rd in range(1, 7):
                rd_key = f'rd{rd}_win'
                if rd_key in team1_prog and rd_key in team2_prog:
                    features[f'team1_{rd_key}'] = team1_prog[rd_key]
                    features[f'team2_{rd_key}'] = team2_prog[rd_key]
                    features[f'{rd_key}_diff'] = team1_prog[rd_key] - team2_prog[rd_key]
            
            # 从晋级概率转换而来的比赛概率 / Add matchup probability converted from progression probabilities
            if round_num > 0:
                from feature_engineering import convert_progression_to_matchup
                matchup_prob = convert_progression_to_matchup(team1_prog, team2_prog, round_num)
                features['progression_matchup_prob'] = matchup_prob
    
    # 添加计算特征 / Add calculated features
    features['win_rate_diff'] = team1_stats['win_rate'] - team2_stats['win_rate']
    features['point_diff_diff'] = team1_stats['point_diff'] - team2_stats['point_diff']
    
    if 'fg_pct' in team1_stats and 'fg_pct' in team2_stats:
        features['fg_pct_diff'] = team1_stats['fg_pct'] - team2_stats['fg_pct']
        features['fg3_pct_diff'] = team1_stats['fg3_pct'] - team2_stats['fg3_pct']
        features['ft_pct_diff'] = team1_stats['ft_pct'] - team2_stats['ft_pct']
        features['rebounding_diff'] = (team1_stats['avg_or'] + team1_stats['avg_dr']) - (team2_stats['avg_or'] + team2_stats['avg_dr'])
        features['assist_diff'] = team1_stats['avg_ast'] - team2_stats['avg_ast']
        features['turnover_diff'] = team1_stats['avg_to'] - team2_stats['avg_to']
        features['steal_diff'] = team1_stats['avg_stl'] - team2_stats['avg_stl']
        features['block_diff'] = team1_stats['avg_blk'] - team2_stats['avg_blk']
    
    # 添加势头特征差异 / Add momentum feature differences
    if 'momentum' in team1_stats and 'momentum' in team2_stats:
        features['momentum_diff'] = team1_stats['momentum'] - team2_stats['momentum']
        features['recent_win_rate_diff'] = team1_stats['recent_win_rate'] - team2_stats['recent_win_rate']
    
    return features, target


def merge_features(tourney_train, team_stats, seed_features, matchup_history, progression_probs=None, use_parallel=True, n_jobs=-1, gender='men'):
    """
    合并所有特征用于模型训练
    Merge all features for model training
    
    Args:
        tourney_train: 锦标赛训练数据
        team_stats: 团队统计数据
        seed_features: 种子特征
        matchup_history: 对战历史
        progression_probs: 晋级概率
        use_parallel: 是否使用并行处理
        n_jobs: 并行作业数
        gender: 'men' 或 'women'，指定处理哪种性别的数据
               'men' or 'women', specifies which gender's data to process
        
    Returns:
        X_df: 特征数据框
        y_series: 目标变量
    """
    # 初始化数据列表 / Initialize data lists
    X_data = []
    y_data = []
    
    # 使用并行处理来加速 / Use parallel processing to speed up
    if use_parallel and len(tourney_train) > 100:
        print(f"使用并行处理 {n_jobs} 个作业... ({gender}) / Using parallel processing with {n_jobs} jobs... ({gender})")
        process_func = partial(process_matchup, 
                              team_stats=team_stats, 
                              seed_features=seed_features, 
                              matchup_history=matchup_history, 
                              progression_probs=progression_probs,
                              gender=gender)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            results = list(executor.map(process_func, [row for _, row in tourney_train.iterrows()]))
        
        for features, target in results:
            if features is not None:
                X_data.append(features)
                y_data.append(target)
    else:
        # 处理每个锦标赛对阵 / Process each tournament matchup
        for _, matchup in tourney_train.iterrows():
            features, target = process_matchup(matchup, team_stats, seed_features, matchup_history, progression_probs, gender=gender)
            if features is not None:
                X_data.append(features)
                y_data.append(target)
    
    # 转换为DataFrame / Convert to DataFrame
    X_df = pd.DataFrame(X_data)
    
    # 应用热门-冷门偏差校正特征 / Apply favorite-longshot bias correction features
    X_df = add_favorite_longshot_features(X_df)
    
    y_series = pd.Series(y_data)
    
    print(f"生成的特征数据形状 ({gender}): {X_df.shape} / Generated feature data shape ({gender}): {X_df.shape}")
    
    return X_df, y_series


def package_features(team_stats, seed_features, matchup_history, progression_probs=None):
    """Package all features into a single dictionary for easier handling"""
    return {
        'team_stats': team_stats,
        'seed_features': seed_features,
        'matchup_history': matchup_history,
        'progression_probs': progression_probs
    }


def build_xgboost_model(X_train, y_train, X_val, y_val, random_seed=42, 
                       use_early_stopping=True, param_tuning=False,
                       visualize=True, save_model_path=None, gender='men'):
    """
    建立XGBoost模型
    Build XGBoost model
    
    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        random_seed: 随机种子
        use_early_stopping: 是否使用早停
        param_tuning: 是否调参
        visualize: 是否可视化
        save_model_path: 模型保存路径
        gender: 性别
        
    返回:
        model: 训练好的模型
        selected_columns: 所选择的特征列名
    """
    print(f"训练XGBoost模型... ({gender}) / Training XGBoost model... ({gender})")
    
    # 记录原始输入类型
    is_numpy_train = isinstance(X_train, np.ndarray)
    is_numpy_val = isinstance(X_val, np.ndarray)
    
    # 检查特征相关性，并返回筛选后的特征和列名
    X_train, selected_columns = check_feature_correlations(X_train)
    
    # 确保验证集也使用相同的特征集
    if is_numpy_val:
        # 如果X_val是NumPy数组，需要将列名转换为索引
        # 创建一个DataFrame获取所有特征名
        all_features = [f'feature_{i}' for i in range(X_val.shape[1])]
        # 找到selected_columns中每个特征在all_features中的索引
        selected_indices = [all_features.index(col) for col in selected_columns if col in all_features]
        # 使用索引选择列
        X_val = X_val[:, selected_indices]
    else:
        # 如果X_val是DataFrame，可以直接用列名索引
        X_val = X_val[selected_columns]
    
    # 创建XGBoost分类器
    # Create XGBoost classifier
    if param_tuning:
        print(f"执行参数调优... ({gender}) / Performing parameter tuning... ({gender})")
        # 参数网格搜索 (Parameter grid search)
        # 配置要搜索的参数网格
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,  # 避免警告信息
            random_state=random_seed
        )
        # 定义网格搜索
        # Define grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='neg_log_loss',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        # 执行网格搜索
        # Execute grid search
        grid_search.fit(X_train, y_train)
        # 获取最佳参数
        # Get best parameters
        best_params = grid_search.best_params_
        print(f"最佳参数 ({gender}): {best_params}")
        # 使用最佳参数创建模型
        # Create model with best parameters
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_seed,
            **best_params
        )
    else:
        # 使用默认参数创建模型
        # Create model with default parameters
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=random_seed,
            learning_rate=0.05,
            n_estimators=300,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1
        )
    
    # 设置基础模型 - 增强正则化以减少过拟合 / Set up base model - enhanced regularization to reduce overfitting
    if use_early_stopping:
        # 早停参数 / Early stopping parameters
        early_stopping_rounds = 10
        print(f"启用早停，rounds={early_stopping_rounds} / Enabling early stopping with rounds={early_stopping_rounds}")
        
        # 不要在参数中设置n_estimators，因为它会被eval_set覆盖 / Don't set n_estimators in params as it will be overridden by eval_set
        xgb_params = {k: v for k, v in xgb_model.get_params().items() if k != 'n_estimators'}
        xgb_params.update({
            'early_stopping_rounds': early_stopping_rounds  # 在这里添加早停参数
        })
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        # 训练模型（带早停） / Train model (with early stopping)
        print(f"训练模型（带早停）... ({gender}) / Training model with early stopping... ({gender})")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        
        n_used = xgb_model.best_iteration + 1
        print(f"早停在 {n_used} 轮后 ({gender}) / Early stopping occurred after {n_used} rounds ({gender})")
    
    # 预测并评估 / Predict and evaluate
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_brier = brier_score_loss(y_val, xgb_pred)
    xgb_log_loss_val = log_loss(y_val, xgb_pred)
    
    print(f"XGBoost模型评估 ({gender}): / XGBoost model evaluation ({gender}):")
    print(f"  - Brier分数: {xgb_brier:.6f} / Brier score: {xgb_brier:.6f}")
    print(f"  - 对数损失: {xgb_log_loss_val:.6f} / Log Loss: {xgb_log_loss_val:.6f}")
    
    # 输出特征重要性 / Output feature importance
    # 确保使用正确的特征名称
    if isinstance(X_train, np.ndarray):
        # 如果是numpy数组，使用索引作为特征名称
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    else:
        # 否则使用DataFrame的列名
        feature_names = X_train.columns.tolist()
        
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nXGBoost特征重要性（前15）({gender}): / XGBoost feature importance (top 15) ({gender}):")
    print(feature_importance.head(15))
    
    # 可视化特征重要性 / Visualize feature importance
    if visualize:
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'XGBoost特征重要性 - 前20 ({gender}) / XGBoost Feature Importance - Top 20 ({gender})')
        plt.tight_layout()
        plt.show()
        
        # 可视化预测分布 / Visualize prediction distribution
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(xgb_pred, bins=20, kde=True)
        plt.title(f'预测概率分布 ({gender}) / Prediction Probability Distribution ({gender})')
        plt.xlabel('预测胜率 / Predicted Win Probability')
        plt.ylabel('频率 / Frequency')
        
        # 可视化预测vs实际结果 / Visualize prediction vs actual results
        plt.subplot(1, 2, 2)
        sns.boxplot(x=y_val, y=xgb_pred)
        plt.title(f'预测vs实际结果 ({gender}) / Predictions vs Actual Results ({gender})')
        plt.xlabel('实际结果 (0=输, 1=赢) / Actual Result (0=Loss, 1=Win)')
        plt.ylabel('预测胜率 / Predicted Win Probability')
        plt.tight_layout()
        plt.show()
    
    # 保存模型（如果指定路径） / Save model if path specified
    if save_model_path:
        model_data = {
            'model': xgb_model,
            'columns': selected_columns
        }
        
        from joblib import dump
        dump(model_data, save_model_path)
        print(f"模型已保存到 {save_model_path} / Model saved to {save_model_path}")
    
    return xgb_model, selected_columns


# 添加一个新函数来处理男女比赛数据的共同训练 / Add a new function to handle combined training for men's and women's data
def train_gender_specific_models(m_features, m_targets, w_features, w_targets, 
                               m_tourney_train, w_tourney_train, 
                               random_seed=42, use_early_stopping=True,
                               save_models_dir=None):
    """
    训练性别特定的模型（男子和女子锦标赛分别训练）
    Train gender-specific models (separate models for men's and women's tournaments)
    
    Args:
        m_features: 男子比赛特征
        m_targets: 男子比赛目标变量
        w_features: 女子比赛特征
        w_targets: 女子比赛目标变量
        m_tourney_train: 男子锦标赛训练数据
        w_tourney_train: 女子锦标赛训练数据
        random_seed: 随机种子
        use_early_stopping: 是否使用早停
        save_models_dir: 模型保存目录
        
    Returns:
        dict: 包含训练好的模型和特征列的字典
              Dictionary containing trained models and feature columns
    """
    from data_preprocessing import prepare_train_val_data_time_aware
    
    print("训练性别特定模型... / Training gender-specific models...")
    
    models = {}
    
    # 为男子比赛训练模型 / Train model for men's games
    print("\n处理男子比赛数据... / Processing men's games data...")
    X_train_m, X_val_m, y_train_m, y_val_m = prepare_train_val_data_time_aware(
        m_features, m_targets, m_tourney_train
    )
    
    men_model_path = os.path.join(save_models_dir, 'men_model.pkl') if save_models_dir else None
    men_model, men_features = build_xgboost_model(
        X_train_m, y_train_m, X_val_m, y_val_m, 
        random_seed=random_seed,
        use_early_stopping=use_early_stopping,
        save_model_path=men_model_path,
        gender='men'
    )
    
    # 为女子比赛训练模型 / Train model for women's games
    print("\n处理女子比赛数据... / Processing women's games data...")
    X_train_w, X_val_w, y_train_w, y_val_w = prepare_train_val_data_time_aware(
        w_features, w_targets, w_tourney_train
    )
    
    women_model_path = os.path.join(save_models_dir, 'women_model.pkl') if save_models_dir else None
    women_model, women_features = build_xgboost_model(
        X_train_w, y_train_w, X_val_w, y_val_w, 
        random_seed=random_seed,
        use_early_stopping=use_early_stopping,
        save_model_path=women_model_path,
        gender='women'
    )
    
    # 存储模型和特征 / Store models and features
    models['men'] = {
        'model': men_model,
        'features': men_features
    }
    
    models['women'] = {
        'model': women_model,
        'features': women_features
    }
    
    return models