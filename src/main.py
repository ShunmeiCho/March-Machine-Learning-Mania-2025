#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Main Script
NCAA篮球锦标赛预测模型 - 主脚本

This is the main script that orchestrates the entire workflow of the prediction model.
这是编排整个预测模型工作流程的主脚本。

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
Version: 2.1 ### 增加了预测所有可能的球队对阵的假设结果
Version: 3.0 ### 增加了cudf的支持
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from datetime import datetime
from joblib import Parallel, delayed, Memory
from functools import partial
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# 定义全局变量
MENS_TEAMS_IDS = set()
WOMENS_TEAMS_IDS = set()

# Import modules
from data_preprocessing import load_data, explore_data, create_tourney_train_data, prepare_train_val_data_time_aware
from feature_engineering import process_seeds, calculate_team_stats, create_matchup_history, calculate_progression_probabilities
from train_model import merge_features, package_features, build_xgboost_model, train_gender_specific_models
from submission import prepare_all_predictions, create_submission, validate_submission
from evaluate import apply_brier_optimal_strategy, evaluate_predictions, visualization_prediction_distribution, calibration_curve
from utils import set_random_seed, save_model, load_model, save_features, load_features, print_section, timer, ensure_directory

# 设置缓存目录，用于存储中间计算结果 
# Set up cache directory for storing intermediate computation results
CACHE_DIR = './cache'
memory = Memory(CACHE_DIR, verbose=0)


def parse_arguments():
    """
    解析命令行参数
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='NCAA Basketball Tournament Prediction Model')
    
    # 基本参数 (Basic parameters)
    parser.add_argument('--data_path', type=str, default='../input',
                        help='数据目录的路径 (Path to data directory)')
    parser.add_argument('--output_path', type=str, default='../output',
                        help='输出目录的路径 (Path to output directory)')
    parser.add_argument('--explore', action='store_true',
                        help='是否进行数据探索和可视化 (Whether to explore data and visualize)')
    parser.add_argument('--train_start_year', type=int, default=2016,
                        help='训练数据的起始年份 (Start year for training data)')
    parser.add_argument('--train_end_year', type=int, default=2024,
                        help='训练数据的结束年份 (End year for training data)')
    parser.add_argument('--target_year', type=int, default=2025,
                        help='预测目标年份 (Target year for predictions)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子，用于结果的可重复性 (Random seed for reproducibility)')
    parser.add_argument('--n_cores', type=int, default=-1,
                        help='用于并行处理的CPU核心数，-1表示使用所有可用核心减1 (Number of CPU cores for parallel processing)')
    parser.add_argument('--use_cache', action='store_true',
                        help='是否使用缓存数据 (Whether to use cached data)')
    
    # 模型参数 (Model parameters)
    parser.add_argument('--xgb_trees', type=int, default=500,
                        help='XGBoost模型的树数量 (Number of trees for XGBoost model)')
    parser.add_argument('--xgb_depth', type=int, default=6,
                        help='XGBoost模型的最大树深度 (Maximum tree depth for XGBoost model)')
    parser.add_argument('--xgb_lr', type=float, default=0.05,
                        help='XGBoost模型的学习率 (Learning rate for XGBoost model)')
    
    # 预测和提交参数 (Prediction and submission parameters)
    parser.add_argument('--generate_predictions', action='store_true',
                        help='为所有可能的对阵生成预测 (Generate predictions for all possible matchups)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='预测输出文件 (Output file for predictions)')
    parser.add_argument('--men_model', type=str, default=None,
                        help='男子模型文件路径 (Path to men\'s model file)')
    parser.add_argument('--women_model', type=str, default=None,
                        help='女子模型文件路径 (Path to women\'s model file)')
    parser.add_argument('--men_features', type=str, default=None,
                        help='男子特征文件路径 (Path to men\'s features file)')
    parser.add_argument('--women_features', type=str, default=None,
                        help='女子特征文件路径 (Path to women\'s features file)')
    parser.add_argument('--load_models', action='store_true',
                        help='加载预训练模型而不是训练新模型 (Load pre-trained models instead of training new ones)')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment with GPU support"""
    # 设置随机种子
    set_random_seed(args.random_seed)
    
    # GPU配置
    try:
        import cupy as cp
        num_gpus = cp.cuda.runtime.getDeviceCount()
        if num_gpus > 0:
            print(f"检测到 {num_gpus} 个GPU设备")
            cp.cuda.Device(0).use()
            args.use_gpu = True
        else:
            print("未检测到GPU设备，将使用CPU")
            args.use_gpu = False
    except Exception as e:
        print(f"GPU初始化失败: {e}")
        args.use_gpu = False
    
    # 创建输出目录
    if args.output_path:
        ensure_directory(args.output_path)
        ensure_directory(os.path.join(args.output_path, 'models'))
        ensure_directory(os.path.join(args.output_path, 'predictions'))
    
    # 设置CPU核心数
    if args.n_cores == -1:
        n_cores = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_cores = args.n_cores
    
    return n_cores


@memory.cache
def process_features_parallel(data_dict, train_start_year, train_end_year, n_cores):
    """
    并行处理特征工程步骤
    Process feature engineering steps in parallel
    
    Args:
        data_dict: 数据字典 (Data dictionary)
        train_start_year: 训练开始年份 (Training start year)
        train_end_year: 训练结束年份 (Training end year)
        n_cores: 使用的CPU核心数 (Number of CPU cores to use)
        
    Returns:
        dict: 处理后的特征字典 (Processed features dictionary)
    """
    print_section("并行处理特征数据 (Processing Features in Parallel)", width=60)
    
    # 男子比赛数据处理 (Men's data processing)
    print("处理男子比赛数据... (Processing men's data...)")
    m_seed_features = process_seeds(
        data_dict['m_tourney_seeds'],
        train_start_year, 
        train_end_year
    )
    
    m_team_stats = calculate_team_stats(
        data_dict['m_regular_season'], 
        data_dict['m_regular_detail'],
        train_start_year, 
        train_end_year
    )
    
    # 并行创建男子对战历史 (Create men's matchup history in parallel)
    print("创建男子对战历史... (Creating men's matchup history...)")
    seasons = list(range(train_start_year, train_end_year + 1))
    
    process_season = partial(
        create_matchup_history_for_season,
        regular_season=data_dict['m_regular_season'],
        tourney_results=data_dict['m_tourney_results']
    )
    
    results = Parallel(n_jobs=n_cores)(
        delayed(process_season)(season) for season in seasons
    )
    
    m_matchup_history = pd.concat(results)
    
    m_progression_probs = calculate_progression_probabilities(
        m_seed_features,
        m_team_stats
    )
    
    # 打包男子特征 (Package men's features)
    m_features_dict = package_features(
        m_team_stats, 
        m_seed_features, 
        m_matchup_history, 
        m_progression_probs
    )
    
    # 新增：女子比赛数据处理 (Women's data processing)
    print("\n处理女子比赛数据... (Processing women's data...)")
    w_seed_features = process_seeds(
        data_dict['w_tourney_seeds'],
        train_start_year, 
        train_end_year
    )
    
    w_team_stats = calculate_team_stats(
        data_dict['w_regular_season'], 
        data_dict['w_regular_detail'],
        train_start_year, 
        train_end_year
    )
    
    # 并行创建女子对战历史 (Create women's matchup history in parallel)
    print("创建女子对战历史... (Creating women's matchup history...)")
    process_w_season = partial(
        create_matchup_history_for_season,
        regular_season=data_dict['w_regular_season'],
        tourney_results=data_dict['w_tourney_results']
    )
    
    w_results = Parallel(n_jobs=n_cores)(
        delayed(process_w_season)(season) for season in seasons
    )
    
    w_matchup_history = pd.concat(w_results)
    
    w_progression_probs = calculate_progression_probabilities(
        w_seed_features,
        w_team_stats
    )
    
    # 打包女子特征 (Package women's features)
    w_features_dict = package_features(
        w_team_stats, 
        w_seed_features, 
        w_matchup_history, 
        w_progression_probs
    )
    
    # 返回男女特征字典 (Return men's and women's feature dictionaries)
    return {
        'men': m_features_dict,
        'women': w_features_dict
    }


def create_matchup_history_for_season(season, regular_season, tourney_results):
    """
    为单个赛季创建对战历史
    Create matchup history for a single season
    
    Args:
        season: 赛季年份 (Season year)
        regular_season: 常规赛数据 (Regular season data)
        tourney_results: 锦标赛结果数据 (Tournament results data)
        
    Returns:
        DataFrame: 该赛季的对战历史 (Matchup history for the season)
    """
    # 筛选出当前赛季的数据
    # Filter data for current season
    season_regular = regular_season[regular_season['Season'] == season]
    season_tourney = tourney_results[tourney_results['Season'] == season]
    
    # 合并常规赛和锦标赛结果
    # Merge regular season and tournament results
    all_games = pd.concat([season_regular, season_tourney])
    
    # 为每个对战创建唯一键
    # Create unique key for each matchup
    all_games['Team1'] = all_games[['WTeamID', 'LTeamID']].min(axis=1)
    all_games['Team2'] = all_games[['WTeamID', 'LTeamID']].max(axis=1)
    all_games['Team1Won'] = (all_games['WTeamID'] == all_games['Team1']).astype(int)
    
    # 使用groupby聚合计算对战历史
    # Use groupby aggregation to calculate matchup history
    result = {}
    
    if not all_games.empty:
        grouped = all_games.groupby(['Team1', 'Team2'])
        
        for (team1, team2), group in grouped:
            # 计算对战统计
            # Calculate matchup statistics
            games_count = len(group)
            team1_wins = group['Team1Won'].sum()
            
            # 计算得分
            # Calculate points
            team1_points = group.apply(
                lambda x: x['WScore'] if x['WTeamID'] == team1 else x['LScore'], 
                axis=1
            ).sum()
            
            team2_points = group.apply(
                lambda x: x['WScore'] if x['WTeamID'] == team2 else x['LScore'], 
                axis=1
            ).sum()
            
            # 存储为字典
            # Store as dictionary
            matchup_key = (team1, team2)
            result[matchup_key] = {
                'games': games_count,
                'wins_team1': team1_wins,
                'points_team1': team1_points,
                'points_team2': team2_points,
                'avg_point_diff': (team1_points - team2_points) / games_count
            }
    
    # 转换为所需的输出格式
    # Convert to required output format
    return pd.DataFrame({
        'Season': season,
        'matchups': [result]
    })


def optimize_memory_usage(df):
    """
    优化DataFrame的内存使用
    Optimize memory usage of a DataFrame
    
    Args:
        df: 需要优化的DataFrame (DataFrame to optimize)
        
    Returns:
        DataFrame: 优化后的DataFrame (Optimized DataFrame)
    """
    # 对每列应用最优数据类型
    # Apply optimal data types to each column
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df[col]) < 0.5:  # 如果基数较低，使用类别类型
                df[col] = df[col].astype('category')
    
    return df


@timer
def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置环境（随机种子等）
    n_cores = setup_environment(args)
    
    # 保存n_cores到args以便后续使用
    args.n_cores = n_cores
    
    # 打印标题
    print_section("NCAA Basketball Tournament Prediction Model")
    
    # 输出配置信息
    print("配置信息 (Configuration):")
    print(f"  - 数据路径 (Data path): {args.data_path}")
    print(f"  - 输出路径 (Output path): {args.output_path}")
    print(f"  - 训练年份 (Training years): {args.train_start_year}-{args.train_end_year}")
    print(f"  - 目标年份 (Target year): {args.target_year}")
    print(f"  - 随机种子 (Random seed): {args.random_seed}")
    print(f"  - 使用CPU核心数 (CPU cores used): {args.n_cores}")
    
    # 加载数据
    data_dict = load_data(args.data_path, use_cache=args.use_cache)
    
    # 设置全局变量供其他模块使用
    global MENS_TEAMS_IDS, WOMENS_TEAMS_IDS
    MENS_TEAMS_IDS = set(data_dict['m_teams']['TeamID'].values)
    WOMENS_TEAMS_IDS = set(data_dict['w_teams']['TeamID'].values)
    
    # 打印数据基本信息
    m_teams_count = data_dict['m_teams'].shape[0]
    w_teams_count = data_dict['w_teams'].shape[0]
    id_overlap = len(set(data_dict['m_teams']['TeamID'].values) & set(data_dict['w_teams']['TeamID'].values))
    print(f"男子队伍数量: {m_teams_count}")
    print(f"女子队伍数量: {w_teams_count}")
    print(f"ID重叠: {id_overlap}")
    
    # 如果启用数据探索
    if args.explore:
        print_section("数据探索与可视化")
        print("执行数据探索和可视化...")
        seed_numbers = explore_data(data_dict, show_plots=True)
        print(f"探索完成，共分析了 {len(seed_numbers)} 个种子数值")
        
        # 也可以添加一些额外的探索性分析
        if 'm_regular_season' in data_dict:
            print("\n常规赛结果分析:")
            m_regular = data_dict['m_regular_season']
            print(f"男子常规赛比赛总数: {len(m_regular)}")
            print(f"赛季范围: {m_regular['Season'].min()} - {m_regular['Season'].max()}")
            print(f"平均得分: {m_regular['WScore'].mean():.2f} (获胜方), {m_regular['LScore'].mean():.2f} (失败方)")
        
        # 在探索模式下可以提前退出，不执行完整的训练和预测流程
        if not args.generate_predictions:
            print("探索模式完成，退出程序。如需生成预测，请添加 --generate_predictions 参数")
            return
    
    # 设置训练年份范围
    train_start_year = args.train_start_year
    train_end_year = args.train_end_year
    print(f"使用{train_start_year}-{train_end_year}年数据进行训练")
    
    # 处理特征（并行）
    features_dict = process_features_parallel(data_dict, train_start_year, train_end_year, args.n_cores)
    
    # 保存特征，以便将来复用
    ensure_directory('features')
    save_features(features_dict['men'], 'features/men_features.pkl')
    save_features(features_dict['women'], 'features/women_features.pkl')
    
    # 创建可视化目录结构
    vis_base_dir = os.path.join(args.output_path, 'visualizations')
    ensure_directory(vis_base_dir)
    ensure_directory(os.path.join(vis_base_dir, 'men'))
    ensure_directory(os.path.join(vis_base_dir, 'women'))
    ensure_directory(os.path.join(vis_base_dir, 'predictions'))
    
    # 加载已训练的模型或训练新模型
    if args.load_models:
        print("加载已训练的模型...")
        m_model_path = args.men_model or 'models/men_model.pkl'
        w_model_path = args.women_model or 'models/women_model.pkl'
        
        # 加载男子模型
        print(f"加载男子模型: {m_model_path}")
        m_model, m_model_columns = load_model(m_model_path)
        
        # 加载女子模型
        print(f"加载女子模型: {w_model_path}")
        w_model, w_model_columns = load_model(w_model_path)
        
        # 加载特征
        m_features_path = args.men_features or 'features/men_features.pkl'
        w_features_path = args.women_features or 'features/women_features.pkl'
        
        print(f"加载男子特征: {m_features_path}")
        m_features_dict = load_features(m_features_path)
        
        print(f"加载女子特征: {w_features_path}")
        w_features_dict = load_features(w_features_path)
        
        # 加载验证数据，用于评估和可视化已加载的模型
        print("准备验证数据用于模型评估...")
        # 男子验证数据
        m_train_data = create_tourney_train_data(
            data_dict['m_tourney_results'], 
            train_start_year, 
            train_end_year
        )
        
        # 确保从特征字典中提取正确的特征矩阵和目标变量
        m_features_df, m_targets = merge_features(
            m_train_data, 
            features_dict['men']['team_stats'],
            features_dict['men']['seed_features'],
            features_dict['men']['matchup_history'],
            progression_probs=features_dict['men'].get('progression_probs'),
            gender='men'
        )
        m_X = m_features_df.values  # 提取特征矩阵
        m_y = m_targets.values  # 提取目标变量
        
        # 现在使用正确的参数类型调用函数
        m_X_train, m_X_val, m_y_train, m_y_val = prepare_train_val_data_time_aware(
            m_X, m_y, m_train_data, test_size=0.2, random_state=args.random_seed
        )
        
        # 女子验证数据
        w_train_data = create_tourney_train_data(
            data_dict['w_tourney_results'], 
            train_start_year, 
            train_end_year
        )
        
        # 首先需要从特征字典构建特征矩阵
        w_features_df, w_targets = merge_features(
            w_train_data, 
            features_dict['women']['team_stats'],
            features_dict['women']['seed_features'],
            features_dict['women']['matchup_history'],
            progression_probs=features_dict['women'].get('progression_probs'),
            gender='women'
        )
        w_X = w_features_df.values  # 提取特征矩阵
        w_y = w_targets.values  # 提取目标变量
        
        # 现在使用正确的参数类型调用函数
        w_X_train, w_X_val, w_y_train, w_y_val = prepare_train_val_data_time_aware(
            w_X, w_y, w_train_data, test_size=0.2, random_state=args.random_seed
        )
        
        # 评估加载的模型
        print_section("评估加载的模型性能")
        # 确保使用正确的特征列进行评估
        if m_model_columns:
            # 检查m_X_val的类型并相应地选择列
            if isinstance(m_X_val, np.ndarray):
                all_features = [f'feature_{i}' for i in range(m_X_val.shape[1])]
                selected_indices = []
                for col in m_model_columns:
                    try:
                        if col in all_features:
                            selected_indices.append(all_features.index(col))
                    except ValueError:
                        print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
                
                if selected_indices:
                    m_X_val = m_X_val[:, selected_indices]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
            else:
                # 确保只使用存在于数据中的列
                existing_cols = [col for col in m_model_columns if col in m_X_val.columns]
                if existing_cols:
                    m_X_val = m_X_val[existing_cols]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
        
        if w_model_columns:
            # 检查w_X_val的类型并相应地选择列
            if isinstance(w_X_val, np.ndarray):
                all_features = [f'feature_{i}' for i in range(w_X_val.shape[1])]
                selected_indices = []
                for col in w_model_columns:
                    try:
                        if col in all_features:
                            selected_indices.append(all_features.index(col))
                    except ValueError:
                        print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
                
                if selected_indices:
                    w_X_val = w_X_val[:, selected_indices]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
            else:
                # 确保只使用存在于数据中的列
                existing_cols = [col for col in w_model_columns if col in w_X_val.columns]
                if existing_cols:
                    w_X_val = w_X_val[existing_cols]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
        
        m_val_preds = m_model.predict_proba(m_X_val)[:, 1]
        w_val_preds = w_model.predict_proba(w_X_val)[:, 1]
        
        # 评估男子模型
        m_eval_results = evaluate_predictions(m_y_val, m_val_preds)
        print(f"男子模型验证集Brier分数: {m_eval_results['brier_score']:.4f}")
        print(f"男子模型验证集ROC AUC: {m_eval_results['roc_auc']:.4f}")
        print(f"男子模型验证集对数损失: {m_eval_results['log_loss']:.4f}")
        
        # 应用Brier最优策略，查看效果
        # 将NumPy数组转换为DataFrame，并添加'Pred'列
        m_val_preds_df = pd.DataFrame({'Pred': m_val_preds})
        m_brier_preds = apply_brier_optimal_strategy(m_val_preds_df)
        m_brier_results = evaluate_predictions(m_y_val, m_brier_preds['Pred'].values)
        print(f"应用Brier策略后，男子模型Brier分数: {m_brier_results['brier_score']:.4f}")
        
        # 生成男子模型的可视化结果
        print("生成男子模型可视化结果...")
        vis_dir = os.path.join(vis_base_dir, 'men')
        
        # 确保使用正确的特征列进行预测
        if m_model_columns and isinstance(m_X_train, np.ndarray):
            # 创建特征名列表
            all_features = [f'feature_{i}' for i in range(m_X_train.shape[1])]
            # 找到m_model_columns中特征在all_features中的索引
            selected_indices = []
            for col in m_model_columns:
                try:
                    if col in all_features:
                        selected_indices.append(all_features.index(col))
                except ValueError:
                    print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
            
            if selected_indices:
                m_X_train_filtered = m_X_train[:, selected_indices]
            else:
                print("警告: 没有匹配的特征列，使用全部特征")
            
            visualization_prediction_distribution(
                m_y_train, 
                m_model.predict_proba(m_X_train_filtered)[:, 1], 
                title='Men\'s Model Training Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'train_pred_distribution.png')
            )
            visualization_prediction_distribution(
                m_y_val, 
                m_val_preds, 
                title='Men\'s Model Validation Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'val_pred_distribution.png')
            )
            calibration_curve(
                m_y_val, 
                m_val_preds,
                title='Men\'s Model Calibration Curve',
                save_path=os.path.join(vis_dir, 'calibration_curve.png')
            )
        
        # 评估女子模型
        w_eval_results = evaluate_predictions(w_y_val, w_val_preds)
        print(f"女子模型验证集Brier分数: {w_eval_results['brier_score']:.4f}")
        print(f"女子模型验证集ROC AUC: {w_eval_results['roc_auc']:.4f}")
        print(f"女子模型验证集对数损失: {w_eval_results['log_loss']:.4f}")
        
        # 应用Brier最优策略，查看效果
        # 将NumPy数组转换为DataFrame，并添加'Pred'列
        w_val_preds_df = pd.DataFrame({'Pred': w_val_preds})
        w_brier_preds = apply_brier_optimal_strategy(w_val_preds_df)
        w_brier_results = evaluate_predictions(w_y_val, w_brier_preds['Pred'].values)
        print(f"应用Brier策略后，女子模型Brier分数: {w_brier_results['brier_score']:.4f}")
        
        # 生成女子模型的可视化结果
        print("生成女子模型可视化结果...")
        vis_dir = os.path.join(vis_base_dir, 'women')
        
        # 确保使用正确的特征列进行预测
        if w_model_columns and isinstance(w_X_train, np.ndarray):
            # 创建特征名列表
            all_features = [f'feature_{i}' for i in range(w_X_train.shape[1])]
            # 找到w_model_columns中特征在all_features中的索引
            selected_indices = []
            for col in w_model_columns:
                try:
                    if col in all_features:
                        selected_indices.append(all_features.index(col))
                except ValueError:
                    print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
            
            if selected_indices:
                w_X_train_filtered = w_X_train[:, selected_indices]
            else:
                print("警告: 没有匹配的特征列，使用全部特征")
            
            visualization_prediction_distribution(
                w_y_train, 
                w_model.predict_proba(w_X_train_filtered)[:, 1], 
                title='Women\'s Model Training Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'train_pred_distribution.png')
            )
            visualization_prediction_distribution(
                w_y_val, 
                w_val_preds, 
                title='Women\'s Model Validation Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'val_pred_distribution.png')
            )
            calibration_curve(
                w_y_val, 
                w_val_preds,
                title='Women\'s Model Calibration Curve',
                save_path=os.path.join(vis_dir, 'calibration_curve.png')
            )
        
    else:
        print("训练新模型...")
        # 准备训练数据
        print("准备男子训练数据...")
        m_train_data = create_tourney_train_data(
            data_dict['m_tourney_results'], 
            train_start_year, 
            train_end_year
        )
        
        # 确保从特征字典中提取正确的特征矩阵和目标变量
        m_features_df, m_targets = merge_features(
            m_train_data, 
            features_dict['men']['team_stats'],
            features_dict['men']['seed_features'],
            features_dict['men']['matchup_history'],
            progression_probs=features_dict['men'].get('progression_probs'),
            gender='men'
        )
        m_X = m_features_df.values  # 提取特征矩阵
        m_y = m_targets.values  # 提取目标变量
        
        # 现在使用正确的参数类型调用函数
        m_X_train, m_X_val, m_y_train, m_y_val = prepare_train_val_data_time_aware(
            m_X, m_y, m_train_data, test_size=0.2, random_state=args.random_seed
        )
        
        # 训练男子模型
        print("训练男子模型...")
        model_dir = os.path.join(args.output_path, "models")
        ensure_directory(model_dir)
        m_model, m_model_columns = build_xgboost_model(
            m_X_train, m_y_train, m_X_val, m_y_val,
            random_seed=args.random_seed,
            use_early_stopping=True,
            save_model_path=os.path.join(model_dir, "men_model_men.pkl"),
            gender='men'
        )
        
        # 准备女子训练数据
        print("准备女子训练数据...")
        w_train_data = create_tourney_train_data(
            data_dict['w_tourney_results'], 
            train_start_year, 
            train_end_year
        )
        
        # 首先需要从特征字典构建特征矩阵
        w_features_df, w_targets = merge_features(
            w_train_data, 
            features_dict['women']['team_stats'],
            features_dict['women']['seed_features'],
            features_dict['women']['matchup_history'],
            progression_probs=features_dict['women'].get('progression_probs'),
            gender='women'
        )
        w_X = w_features_df.values  # 提取特征矩阵
        w_y = w_targets.values  # 提取目标变量
        
        # 现在使用正确的参数类型调用函数
        w_X_train, w_X_val, w_y_train, w_y_val = prepare_train_val_data_time_aware(
            w_X, w_y, w_train_data, test_size=0.2, random_state=args.random_seed
        )
        
        # 训练女子模型
        print("训练女子模型...")
        w_model, w_model_columns = build_xgboost_model(
            w_X_train, w_y_train, w_X_val, w_y_val,
            random_seed=args.random_seed,
            use_early_stopping=True,
            save_model_path=os.path.join(model_dir, "women_model_women.pkl"),
            gender='women'
        )
        
        # 使用训练好的模型进行进一步处理
        m_features_dict = features_dict['men']
        w_features_dict = features_dict['women']
        
        # 评估模型性能并生成可视化结果
        print_section("评估模型性能")
        # 确保使用正确的特征列进行评估
        if m_model_columns:
            # 检查m_X_val的类型并相应地选择列
            if isinstance(m_X_val, np.ndarray):
                all_features = [f'feature_{i}' for i in range(m_X_val.shape[1])]
                selected_indices = []
                for col in m_model_columns:
                    try:
                        if col in all_features:
                            selected_indices.append(all_features.index(col))
                    except ValueError:
                        print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
                
                if selected_indices:
                    m_X_val = m_X_val[:, selected_indices]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
            else:
                # 确保只使用存在于数据中的列
                existing_cols = [col for col in m_model_columns if col in m_X_val.columns]
                if existing_cols:
                    m_X_val = m_X_val[existing_cols]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
        
        if w_model_columns:
            # 检查w_X_val的类型并相应地选择列
            if isinstance(w_X_val, np.ndarray):
                all_features = [f'feature_{i}' for i in range(w_X_val.shape[1])]
                selected_indices = []
                for col in w_model_columns:
                    try:
                        if col in all_features:
                            selected_indices.append(all_features.index(col))
                    except ValueError:
                        print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
                
                if selected_indices:
                    w_X_val = w_X_val[:, selected_indices]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
            else:
                # 确保只使用存在于数据中的列
                existing_cols = [col for col in w_model_columns if col in w_X_val.columns]
                if existing_cols:
                    w_X_val = w_X_val[existing_cols]
                else:
                    print("警告: 没有匹配的特征列，使用全部特征")
        
        m_val_preds = m_model.predict_proba(m_X_val)[:, 1]
        w_val_preds = w_model.predict_proba(w_X_val)[:, 1]
        
        # 评估男子模型
        m_eval_results = evaluate_predictions(m_y_val, m_val_preds)
        print(f"男子模型验证集Brier分数: {m_eval_results['brier_score']:.4f}")
        print(f"男子模型验证集ROC AUC: {m_eval_results['roc_auc']:.4f}")
        print(f"男子模型验证集对数损失: {m_eval_results['log_loss']:.4f}")
        
        # 应用Brier最优策略，查看效果
        # 将NumPy数组转换为DataFrame，并添加'Pred'列
        m_val_preds_df = pd.DataFrame({'Pred': m_val_preds})
        m_brier_preds = apply_brier_optimal_strategy(m_val_preds_df)
        m_brier_results = evaluate_predictions(m_y_val, m_brier_preds['Pred'].values)
        print(f"应用Brier策略后，男子模型Brier分数: {m_brier_results['brier_score']:.4f}")
        
        # 生成男子模型的可视化结果
        print("生成男子模型可视化结果...")
        vis_dir = os.path.join(vis_base_dir, 'men')
        
        # 确保使用正确的特征列进行预测
        if m_model_columns and isinstance(m_X_train, np.ndarray):
            # 创建特征名列表
            all_features = [f'feature_{i}' for i in range(m_X_train.shape[1])]
            # 找到m_model_columns中特征在all_features中的索引
            selected_indices = []
            for col in m_model_columns:
                try:
                    if col in all_features:
                        selected_indices.append(all_features.index(col))
                except ValueError:
                    print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
            
            if selected_indices:
                m_X_train_filtered = m_X_train[:, selected_indices]
            else:
                print("警告: 没有匹配的特征列，使用全部特征")
            
            visualization_prediction_distribution(
                m_y_train, 
                m_model.predict_proba(m_X_train_filtered)[:, 1], 
                title='Men\'s Model Training Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'train_pred_distribution.png')
            )
            visualization_prediction_distribution(
                m_y_val, 
                m_val_preds, 
                title='Men\'s Model Validation Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'val_pred_distribution.png')
            )
            calibration_curve(
                m_y_val, 
                m_val_preds,
                title='Men\'s Model Calibration Curve',
                save_path=os.path.join(vis_dir, 'calibration_curve.png')
            )
        
        # 评估女子模型
        w_eval_results = evaluate_predictions(w_y_val, w_val_preds)
        print(f"女子模型验证集Brier分数: {w_eval_results['brier_score']:.4f}")
        print(f"女子模型验证集ROC AUC: {w_eval_results['roc_auc']:.4f}")
        print(f"女子模型验证集对数损失: {w_eval_results['log_loss']:.4f}")
        
        # 应用Brier最优策略，查看效果
        # 将NumPy数组转换为DataFrame，并添加'Pred'列
        w_val_preds_df = pd.DataFrame({'Pred': w_val_preds})
        w_brier_preds = apply_brier_optimal_strategy(w_val_preds_df)
        w_brier_results = evaluate_predictions(w_y_val, w_brier_preds['Pred'].values)
        print(f"应用Brier策略后，女子模型Brier分数: {w_brier_results['brier_score']:.4f}")
        
        # 生成女子模型的可视化结果
        print("生成女子模型可视化结果...")
        vis_dir = os.path.join(vis_base_dir, 'women')
        
        # 确保使用正确的特征列进行预测
        if w_model_columns and isinstance(w_X_train, np.ndarray):
            # 创建特征名列表
            all_features = [f'feature_{i}' for i in range(w_X_train.shape[1])]
            # 找到w_model_columns中特征在all_features中的索引
            selected_indices = []
            for col in w_model_columns:
                try:
                    if col in all_features:
                        selected_indices.append(all_features.index(col))
                except ValueError:
                    print(f"警告: 特征 '{col}' 在训练数据中未找到，将被忽略")
            
            if selected_indices:
                w_X_train_filtered = w_X_train[:, selected_indices]
            else:
                print("警告: 没有匹配的特征列，使用全部特征")
            
            visualization_prediction_distribution(
                w_y_train, 
                w_model.predict_proba(w_X_train_filtered)[:, 1], 
                title='Women\'s Model Training Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'train_pred_distribution.png')
            )
            visualization_prediction_distribution(
                w_y_val, 
                w_val_preds, 
                title='Women\'s Model Validation Set Prediction Distribution',
                save_path=os.path.join(vis_dir, 'val_pred_distribution.png')
            )
            calibration_curve(
                w_y_val, 
                w_val_preds,
                title='Women\'s Model Calibration Curve',
                save_path=os.path.join(vis_dir, 'calibration_curve.png')
            )
    
    # === 生成所有可能对阵的预测 ===
    if args.generate_predictions:
        print("生成2025年所有可能对阵的预测...")
        
        # 确保正确传递模型列名
        men_predictions = prepare_all_predictions(
            m_model, features_dict, data_dict, 
            model_columns=m_model_columns,
            year=args.target_year, 
            n_jobs=args.n_cores,
            gender='men'
        )
        
        women_predictions = prepare_all_predictions(
            w_model, features_dict, data_dict, 
            model_columns=w_model_columns,
            year=args.target_year, 
            n_jobs=args.n_cores,
            gender='women'
        )
        
        # 合并预测
        all_predictions = pd.concat([men_predictions, women_predictions])
        
        # 添加预测分布可视化
        print("生成预测分布可视化...")
        vis_dir = os.path.join(vis_base_dir, 'predictions')
        
        # 男子预测分布可视化
        m_pred_probs = men_predictions['Pred'].values
        visualization_prediction_distribution(
            None, 
            m_pred_probs, 
            title='Men\'s Predictions Distribution',
            save_path=os.path.join(vis_dir, 'men_predictions_dist.png')
        )
        
        # 女子预测分布可视化
        w_pred_probs = women_predictions['Pred'].values
        visualization_prediction_distribution(
            None, 
            w_pred_probs, 
            title='Women\'s Predictions Distribution',
            save_path=os.path.join(vis_dir, 'women_predictions_dist.png')
        )
        
        # 所有预测的整体分布
        all_pred_probs = all_predictions['Pred'].values
        visualization_prediction_distribution(
            None, 
            all_pred_probs, 
            title='All Predictions Distribution',
            save_path=os.path.join(vis_dir, 'all_predictions_dist.png')
        )
        
        # 分析预测偏差，查看男队和女队预测是否有系统性差异
        print_section("分析预测偏差")
        print(f"男子队伍预测平均值: {m_pred_probs.mean():.4f}")
        print(f"女子队伍预测平均值: {w_pred_probs.mean():.4f}")
        print(f"男子队伍预测标准差: {m_pred_probs.std():.4f}")
        print(f"女子队伍预测标准差: {w_pred_probs.std():.4f}")
        
        # 生成比较图表
        plt.figure(figsize=(10, 6))
        sns.kdeplot(m_pred_probs, label='Men\'s Teams', fill=True, alpha=0.5)
        sns.kdeplot(w_pred_probs, label='Women\'s Teams', fill=True, alpha=0.5)
        plt.title('Men\'s vs Women\'s Teams Prediction Probability Distribution')
        plt.xlabel('Predicted Win Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(vis_dir, 'men_vs_women_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建提交文件
        submission_file = args.output_file or f"submission_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        print(f"创建提交文件: {submission_file}")
        submission = create_submission(
            all_predictions, data_dict['sample_sub'], 
            filename=os.path.join(args.output_path, submission_file)
        )
        
        # 验证提交文件
        print("验证提交文件...")
        is_valid = validate_submission(submission, data_dict)
        
        if is_valid:
            print("✓ 提交文件验证成功！符合2025比赛要求")
        else:
            print("⚠ 提交文件验证失败！请检查")
    
    print("处理完成！所有可视化结果已保存至 {}".format(vis_base_dir))


if __name__ == "__main__":
    main()