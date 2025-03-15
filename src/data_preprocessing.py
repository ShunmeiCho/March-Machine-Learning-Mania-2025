#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Data Preprocessing

This module handles loading and preprocessing data for the NCAA basketball tournament prediction model.

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tempfile
import hashlib
from joblib import Parallel, delayed
from tqdm import tqdm


def load_data(data_path, use_cache=True, cache_dir=None):
    """
    加载NCAA篮球比赛数据集中的必要文件，并支持缓存功能
    Load necessary data files from the NCAA basketball dataset with caching support
    
    参数 Parameters:
        data_path (str): 数据文件所在的目录路径
                         Directory path containing the data files
        use_cache (bool): 是否使用缓存，默认为True
                          Whether to use cache, defaults to True
        cache_dir (str): 缓存目录，默认为None时使用系统临时目录
                         Cache directory, uses system temp directory when None
    
    返回 Returns:
        dict: 包含所有加载数据集的字典，键为数据集名称，值为对应的DataFrame
              Dictionary with all loaded datasets, keys are dataset names, values are corresponding DataFrames
    """
    # 如果未指定缓存目录，使用系统临时目录
    # Use system temp directory if cache_dir is not specified
    if cache_dir is None:
        cache_dir = tempfile.gettempdir()
    
    # 创建缓存文件路径
    # Create cache file path
    cache_hash = hashlib.md5(data_path.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"ncaa_data_cache_{cache_hash}.pkl")
    
    # 尝试从缓存加载
    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        try:
            print("Loading data from cache...")
            with open(cache_file, 'rb') as f:
                data_dict = pickle.load(f)
            print("Data loaded from cache successfully.")
            for key, df in data_dict.items():
                print(f"{key}: shape {df.shape}")
            return data_dict
        except Exception as e:
            print(f"Error loading cache: {e}. Will load from original files.")
    
    print("Loading data from original files...")
    
    # 定义要加载的文件列表，使用字典映射数据集名称到文件名
    # Define files to be loaded using a dictionary mapping dataset names to filenames
    files_to_load = {
        'm_teams': 'MTeams.csv',               # 男子队伍信息 (Men's teams information)
        'w_teams': 'WTeams.csv',               # 女子队伍信息 (Women's teams information)
        'm_regular_season': 'MRegularSeasonCompactResults.csv',  # 男子常规赛结果 (Men's regular season results)
        'w_regular_season': 'WRegularSeasonCompactResults.csv',  # 女子常规赛结果 (Women's regular season results)
        'm_tourney_results': 'MNCAATourneyCompactResults.csv',   # 男子锦标赛结果 (Men's tournament results)
        'w_tourney_results': 'WNCAATourneyCompactResults.csv',   # 女子锦标赛结果 (Women's tournament results)
        'm_regular_detail': 'MRegularSeasonDetailedResults.csv', # 男子常规赛详细统计 (Men's regular season detailed stats)
        'w_regular_detail': 'WRegularSeasonDetailedResults.csv', # 女子常规赛详细统计 (Women's regular season detailed stats)
        'm_tourney_seeds': 'MNCAATourneySeeds.csv',              # 男子锦标赛种子信息 (Men's tournament seeds)
        'w_tourney_seeds': 'WNCAATourneySeeds.csv',              # 女子锦标赛种子信息 (Women's tournament seeds)
        'sample_sub': 'SampleSubmissionStage1.csv'               # 样本提交文件 (Sample submission file)
    }
    
    # 验证文件存在性，避免后续错误
    # Verify file existence to avoid subsequent errors
    for filename in files_to_load.values():
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
    
    # 使用字典推导式高效地加载所有文件，避免重复代码
    # Use dictionary comprehension to efficiently load all files, avoiding code duplication
    data_dict = {key: pd.read_csv(os.path.join(data_path, filename)) 
                for key, filename in files_to_load.items() 
                if os.path.exists(os.path.join(data_path, filename))}
    
    # 保存到缓存
    # Save to cache
    if use_cache:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Data cached to {cache_file}")
        except Exception as e:
            print(f"Error caching data: {e}")
    
    print("Data loading complete. Dataset overview:")
    for key, df in data_dict.items():
        print(f"{key}: shape {df.shape}")
    
    return data_dict


def explore_data(data_dict, show_plots=True):
    """
    探索数据结构和基本统计特征，生成可视化图表
    Explore data structure and basic statistics, generate visualizations
    
    参数 Parameters:
        data_dict (dict): 包含所有数据集的字典
                          Dictionary containing all datasets
        show_plots (bool): 是否显示图表，默认为True
                           Whether to display plots, defaults to True
    
    返回 Returns:
        pandas.Series: 男子锦标赛种子数值，用于后续分析
                       Men's tournament seed numbers for further analysis
    """
    
    # 打印每个数据集的基本信息
    # Print basic information for each dataset
    print("Dataset overview:")
    for name, df in data_dict.items():
        print(f"\n{name} dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:")
        print(df.head(2))
    
    # 探索男子锦标赛结果按赛季分布
    # Explore men's tournament results distribution by season
    m_tourney = data_dict['m_tourney_results']
    print("\nMen's tournament results by season:")
    print(m_tourney['Season'].value_counts().sort_index())
    
    # 探索种子分布情况
    # Explore seed distribution
    m_seeds = data_dict['m_tourney_seeds']
    print("\nMen's tournament seed distribution:")
    # 提取种子数值 - 使用更高效的方法
    # Extract numeric seed values - using more efficient methods
    # 直接使用数值提取，避免正则表达式的开销
    # Directly extract numbers, avoiding regular expression overhead
    m_seeds['Seed_Number'] = m_seeds['Seed'].str[1:3].astype(int)
    print(m_seeds['Seed_Number'].value_counts().sort_index())
    
    # 只有在需要时才创建可视化图表，避免不必要的计算
    # Only create visualizations when needed, avoiding unnecessary computation
    if show_plots:
        plt.figure(figsize=(12, 5))
        
        # 绘制比分差异分布图 - 使用向量化操作提高性能
        # Plot score difference distribution - use vectorized operations for better performance
        plt.subplot(1, 2, 1)
        m_tourney = data_dict['m_tourney_results']
        score_diff = m_tourney['WScore'] - m_tourney['LScore']
        # 替换无穷值为NaN，使用pandas内置方法
        # Replace infinite values with NaN, using pandas built-in method
        score_diff = score_diff.replace([np.inf, -np.inf], np.nan)
        sns.histplot(score_diff, kde=True)
        plt.title('Distribution of score differences in championships (MEN)')
        plt.xlabel('Score difference')
        
        # 绘制种子与胜率关系图 - 使用更高效的数据处理方法
        # Plot seed vs win rate relationship - use more efficient data processing methods
        plt.subplot(1, 2, 2)
        
        # 使用向量化操作代替循环
        # Use vectorized operations instead of loops
        m_tourney = data_dict['m_tourney_results']
        m_seeds = data_dict['m_tourney_seeds']
        
        # 创建联合键以便进行高效连接
        # Create join keys for efficient joining
        m_seeds['key'] = m_seeds['Season'].astype(str) + '_' + m_seeds['TeamID'].astype(str)
        m_tourney['w_key'] = m_tourney['Season'].astype(str) + '_' + m_tourney['WTeamID'].astype(str)
        m_tourney['l_key'] = m_tourney['Season'].astype(str) + '_' + m_tourney['LTeamID'].astype(str)
        
        # 使用映射函数代替循环，提高性能
        # Use mapping function instead of loops for better performance
        m_seeds_dict = dict(zip(m_seeds['key'], m_seeds['Seed'].str[1:3].astype(int)))
        
        # 映射种子值到比赛结果
        # Map seed values to game results
        m_tourney['w_seed'] = m_tourney['w_key'].map(m_seeds_dict)
        m_tourney['l_seed'] = m_tourney['l_key'].map(m_seeds_dict)
        
        # 丢弃种子值缺失的行
        # Drop rows with missing seed values
        seed_games = m_tourney.dropna(subset=['w_seed', 'l_seed'])
        
        # 向量化操作计算最小种子和低种子是否获胜
        # Vectorized operations to calculate min seed and whether lower seed won
        seed_games['min_seed'] = seed_games[['w_seed', 'l_seed']].min(axis=1)
        seed_games['lower_seed_won'] = (seed_games['w_seed'] < seed_games['l_seed']).astype(int)
        
        # 计算按最小种子分组的低种子胜率
        # Calculate lower seed win rate by min seed group
        seed_win_rates = seed_games.groupby('min_seed')['lower_seed_won'].mean()
        
        # 绘制种子与胜率关系的条形图
        # Plot bar chart of seed vs. win rate relationship
        sns.barplot(x=seed_win_rates.index, y=seed_win_rates.values)
        plt.title('The team with the lower seeds has a higher winning percentage')
        plt.xlabel('Minimum seed')
        plt.ylabel('Winning percentage')
        
        plt.tight_layout()
        plt.show()
    
    return m_seeds['Seed_Number']  # 返回种子数值用于后续分析 (Return seed numbers for later use)


def estimate_round_number(day_num):
    """
    根据比赛日期估计锦标赛轮次
    Estimate tournament round based on game date (DayNum)
    
    NCAA锦标赛通常遵循以下时间表:
    The NCAA tournament typically follows this schedule:
    - 第1轮: First Four/Play-in (日期<136)
    - 第2轮: First Round/Round of 64 (日期136-137)
    - 第3轮: Second Round/Round of 32 (日期138-142)
    - 第4轮: Sweet 16 (日期143-144)
    - 第5轮: Elite 8 (日期145-149)
    - 第6轮: Final Four and Championship (日期>=150)
    
    参数 Parameters:
        day_num (int): 比赛日期编号
                       Game date number
    
    返回 Returns:
        int: 估计的锦标赛轮次 (1-6)
             Estimated tournament round (1-6)
    """
    # 使用高效的条件判断估计轮次
    # Use efficient conditional logic to estimate round
    if day_num < 136:
        return 1
    elif day_num < 138:
        return 2
    elif day_num < 143:
        return 3
    elif day_num < 145:
        return 4
    elif day_num < 150:
        return 5
    else:
        return 6


def create_tourney_train_data(tourney_results, start_year, end_year):
    """
    从锦标赛结果创建训练数据集，使用向量化操作提高性能
    Create training dataset from tournament results, using vectorized operations for better performance
    
    参数 Parameters:
        tourney_results (pd.DataFrame): 锦标赛结果数据
                                        Tournament results data
        start_year (int): 起始年份
                          Start year for filtering
        end_year (int): 结束年份
                        End year for filtering
    
    返回 Returns:
        pd.DataFrame: 处理后的训练数据
                      Processed training data
    """
    # 使用query方法高效过滤数据
    # Use query method to filter data efficiently
    filtered_results = tourney_results.query(f"Season >= {start_year} and Season <= {end_year}").copy()
    
    # 创建临时列用于向量化操作
    # Create temporary columns for vectorized operations
    filtered_results.loc[:, 'Team1'] = filtered_results[['WTeamID', 'LTeamID']].min(axis=1)
    filtered_results.loc[:, 'Team2'] = filtered_results[['WTeamID', 'LTeamID']].max(axis=1)
    filtered_results.loc[:, 'Team1_Win'] = (filtered_results['WTeamID'] == filtered_results['Team1']).astype(int)
    
    # 向量化估计比赛轮次
    # Vectorized estimation of game rounds
    filtered_results.loc[:, 'Round'] = filtered_results['DayNum'].apply(estimate_round_number)
    
    # 选择必要的列返回
    # Select necessary columns to return
    tourney_train_data = filtered_results[['Season', 'Team1', 'Team2', 'Team1_Win', 'Round', 'DayNum']]
    
    return tourney_train_data


def prepare_train_val_data_time_aware(X, y, tourney_train, test_size=0.2, random_state=42):
    """
    考虑时间顺序拆分训练和验证数据集，提供更多参数控制和健壮性
    Split data into training and validation sets considering time order, with better parameter control and robustness
    
    参数 Parameters:
        X (np.ndarray): 特征矩阵
                        Feature matrix
        y (np.ndarray): 目标变量
                        Target variable
        tourney_train (pd.DataFrame): 包含赛季信息的训练数据
                                      Training data containing season information
        test_size (float): 验证集比例，默认0.2
                           Validation set proportion, default 0.2
        random_state (int): 随机种子，用于回退到随机分割时
                           Random seed, used when falling back to random splitting
    
    返回 Returns:
        tuple: (X_train, X_val, y_train, y_val) 训练集和验证集的特征和目标变量
               Training and validation sets' features and targets
    """
    # 输入验证
    # Input validation
    if X.shape[0] != y.shape[0] or X.shape[0] != tourney_train.shape[0]:
        raise ValueError("数据维度不匹配：X、y和tourney_train的行数必须相同。"
                         "Dimension mismatch: X, y and tourney_train must have same number of rows.")
    
    # 使用pandas的Series类型进行高效操作
    # Use pandas Series for efficient operations
    seasons = pd.Series(tourney_train['Season'].values)
    
    # 计算时间切分点
    # Calculate time splitting point
    unique_seasons = seasons.unique()
    unique_seasons.sort()  # 确保按时间顺序排列 (ensure time order)
    
    # 计算验证集开始的季节索引
    # Calculate index for validation set starting season
    cutoff_idx = max(1, int(len(unique_seasons) * (1 - test_size)))
    
    # 分割训练和验证赛季
    # Split training and validation seasons
    train_seasons = unique_seasons[:cutoff_idx]
    test_seasons = unique_seasons[cutoff_idx:]
    
    # 使用isin方法创建高效的布尔掩码
    # Use isin method to create efficient boolean masks
    train_mask = seasons.isin(train_seasons).values
    test_mask = seasons.isin(test_seasons).values
    
    # 确保分割后的数据集非空
    # Ensure non-empty datasets after splitting
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("警告：时间分割导致训练集或验证集为空，回退到随机分割。"
              "Warning: Time split resulted in empty training or validation set, falling back to random split.")
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # 使用布尔索引进行高效分割
    # Use boolean indexing for efficient splitting
    X_train, X_val = X[train_mask], X[test_mask]
    y_train, y_val = y[train_mask], y[test_mask]
    
    print(f"时间分割：训练赛季 {train_seasons}，验证赛季 {test_seasons}"
          f"\nTime split: Training seasons {train_seasons}, Validation seasons {test_seasons}")
    print(f"训练集大小：{X_train.shape[0]}，验证集大小：{X_val.shape[0]}"
          f"\nTraining set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
    
    return X_train, X_val, y_train, y_val


def parallel_feature_extraction(data_dict, n_jobs=-1, verbose=1):
    """
    使用并行处理从数据中提取特征，适用于大型数据集
    Extract features from data using parallel processing, suitable for large datasets
    
    参数 Parameters:
        data_dict (dict): 包含所有数据集的字典
                          Dictionary containing all datasets
        n_jobs (int): 并行使用的CPU数量，-1表示所有CPU
                      Number of CPUs to use, -1 means all CPUs
        verbose (int): 详细程度，0=静默，1=进度条，>1=更多信息
                      Verbosity level, 0=silent, 1=progress bar, >1=more info
    
    返回 Returns:
        pd.DataFrame: 提取的特征
                      Extracted features
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    # 定义要并行处理的数据
    # Define data to be processed in parallel
    regular_season = data_dict['m_regular_season']
    teams = data_dict['m_teams']
    
    # 按团队分组以便并行处理
    # Group by team for parallel processing
    team_ids = teams['TeamID'].unique()
    
    def process_team(team_id):
        """处理单个团队的数据 (Process data for a single team)"""
        # 获取团队比赛 (Get team games)
        team_games = regular_season[(regular_season['WTeamID'] == team_id) | 
                                   (regular_season['LTeamID'] == team_id)]
        
        # 计算获胜率 (Calculate win rate)
        wins = team_games[team_games['WTeamID'] == team_id].shape[0]
        total_games = team_games.shape[0]
        win_rate = wins / total_games if total_games > 0 else 0
        
        # 计算平均得分 (Calculate average score)
        w_scores = team_games[team_games['WTeamID'] == team_id]['WScore'].sum()
        l_scores = team_games[team_games['LTeamID'] == team_id]['LScore'].sum()
        total_points = w_scores + l_scores
        avg_points = total_points / total_games if total_games > 0 else 0
        
        return {
            'TeamID': team_id,
            'WinRate': win_rate,
            'AvgPoints': avg_points,
            'TotalGames': total_games
        }
    
    # 使用并行处理提取特征
    # Extract features using parallel processing
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_team)(team_id) for team_id in tqdm(team_ids, desc="处理团队 (Processing teams)")
    )
    
    # 将结果转换为DataFrame
    # Convert results to DataFrame
    team_features = pd.DataFrame(results)
    
    return team_features