#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Feature Engineering

This module handles creating and transforming features for the NCAA basketball tournament prediction model.

Author: Junming Zhao
Date: 2025-03-13
Version: 2.0 ### 增加了针对女队的预测
"""
import pandas as pd
import numpy as np
from scipy.special import expit
import pandas as pd
import numpy as np
from data_preprocessing import estimate_round_number


def process_seeds(tourney_seeds, start_year, end_year):
    """
    Process seed information into numerical features
    将种子信息转换为数值特征
    """
    # 预先过滤数据，避免在循环中重复筛选
    # Pre-filter data to avoid repeated filtering in the loop
    filtered_seeds = tourney_seeds[(tourney_seeds['Season'] >= start_year) & 
                                  (tourney_seeds['Season'] <= end_year)]
    
    # 使用字典推导式替代循环创建字典
    # Use dictionary comprehension instead of loop to create dictionary
    seed_features = {
        season: {
            row['TeamID']: {
                'seed_num': int(row['Seed'][1:3]),
                'region': row['Seed'][0],
                'seed_str': row['Seed']
            }
            for _, row in filtered_seeds[filtered_seeds['Season'] == season].iterrows()
        }
        for season in range(start_year, end_year + 1)
    }
    
    return seed_features


def calculate_team_stats(regular_season, regular_detail, start_year, end_year):
    """
    Calculate performance statistics for each team in each season
    计算每个赛季中每支队伍的表现统计数据
    """
    team_stats = {}
    
    # 预先过滤赛季数据
    regular_season_filtered = regular_season.query(f"Season >= {start_year} and Season <= {end_year}")
    
    # 处理详细数据
    if regular_detail is not None:
        regular_detail_filtered = regular_detail.query(f"Season >= {start_year} and Season <= {end_year}")
    else:
        regular_detail_filtered = None
    
    # 使用每个赛季的唯一队伍ID进行预处理
    all_season_team_ids = set()
    for season in range(start_year, end_year + 1):
        season_data = regular_season_filtered[regular_season_filtered['Season'] == season]
        season_team_ids = set(season_data['WTeamID'].unique()) | set(season_data['LTeamID'].unique())
        all_season_team_ids.update(season_team_ids)
    
    # 预先分配内存
    columns = ['num_wins', 'total_games', 'points_scored', 'points_allowed', 
              'num_losses', 'win_rate', 'avg_points_scored', 'avg_points_allowed', 
              'point_diff', 'home_wins', 'home_games', 'home_win_rate',
              'away_wins', 'away_games', 'away_win_rate', 'recent_win_rate', 
              'momentum', 'season_rank', 'normalized_rank']
    
    for season in range(start_year, end_year + 1):
        season_data = regular_season_filtered[regular_season_filtered['Season'] == season]
        
        # 使用向量化操作创建胜负数据框
        # Use vectorized operations to create win/loss dataframes
        wins_df = season_data[['WTeamID', 'WScore', 'LScore', 'WLoc', 'DayNum']].rename(
            columns={'WTeamID': 'TeamID', 'WScore': 'PointsScored', 'LScore': 'PointsAllowed', 'WLoc': 'Loc'}
        ).assign(Win=1)
        
        losses_df = season_data[['LTeamID', 'LScore', 'WScore', 'WLoc', 'DayNum']].rename(
            columns={'LTeamID': 'TeamID', 'LScore': 'PointsScored', 'WScore': 'PointsAllowed'}
        ).assign(Win=0)
        
        # 转换主场/客场标记
        # Transform home/away indicators
        losses_df['Loc'] = losses_df['WLoc'].map({'H': 'A', 'A': 'H', 'N': 'N'})
        
        # 合并所有比赛数据
        # Merge all games data
        all_games = pd.concat([wins_df, losses_df]).reset_index(drop=True)
        
        # 使用groupby进行聚合计算
        # Use groupby for aggregation calculations
        basic_stats = all_games.groupby('TeamID').agg(
            num_wins=('Win', 'sum'),
            total_games=('Win', 'count'),
            points_scored=('PointsScored', 'sum'),
            points_allowed=('PointsAllowed', 'sum')
        )
        
        # 计算派生统计数据
        # Calculate derived statistics
        basic_stats['num_losses'] = basic_stats['total_games'] - basic_stats['num_wins']
        basic_stats['win_rate'] = basic_stats['num_wins'] / basic_stats['total_games']
        basic_stats['avg_points_scored'] = basic_stats['points_scored'] / basic_stats['total_games']
        basic_stats['avg_points_allowed'] = basic_stats['points_allowed'] / basic_stats['total_games']
        basic_stats['point_diff'] = basic_stats['avg_points_scored'] - basic_stats['avg_points_allowed']
        
        # 主场/客场统计数据
        # Home/away statistics
        home_stats = all_games[all_games['Loc'] == 'H'].groupby('TeamID').agg(
            home_wins=('Win', 'sum'),
            home_games=('Win', 'count')
        )
        home_stats['home_win_rate'] = home_stats['home_wins'] / home_stats['home_games']
        
        away_stats = all_games[all_games['Loc'] == 'A'].groupby('TeamID').agg(
            away_wins=('Win', 'sum'),
            away_games=('Win', 'count')
        )
        away_stats['away_win_rate'] = away_stats['away_wins'] / away_stats['away_games']
        
        # 合并基础统计数据
        # Merge basic statistics
        team_season_stats = pd.merge(basic_stats, home_stats, on='TeamID', how='left')
        team_season_stats = pd.merge(team_season_stats, away_stats, on='TeamID', how='left')
        
        # 处理NaN值（对于没有主场或客场比赛的队伍）
        # Handle NaN values (for teams without home or away games)
        team_season_stats = team_season_stats.fillna({
            'home_win_rate': 0, 
            'away_win_rate': 0,
            'home_wins': 0,
            'home_games': 0,
            'away_wins': 0,
            'away_games': 0
        })
        
        # 转换为字典前减少内存使用
        # 使用浮点精度降低
        float_cols = ['win_rate', 'avg_points_scored', 'avg_points_allowed', 
                     'point_diff', 'home_win_rate', 'away_win_rate', 
                     'recent_win_rate', 'momentum', 'normalized_rank']
        
        for col in float_cols:
            if col in team_season_stats:
                team_season_stats[col] = team_season_stats[col].astype('float32')
        
        # 使用最优的整数类型
        int_cols = ['num_wins', 'total_games', 'home_wins', 'home_games', 
                   'away_wins', 'away_games', 'season_rank']
        
        for col in int_cols:
            if col in team_season_stats:
                team_season_stats[col] = team_season_stats[col].astype('int32')
                
        # 转换为字典
        teams_season_stats = team_season_stats.to_dict('index')
        team_stats[season] = teams_season_stats
    
    return team_stats


def create_matchup_history(regular_season, tourney_results, start_year, end_year):
    """
    Create features based on historical matchups
    基于历史对战创建特征
    """
    # 合并常规赛和锦标赛结果
    # Merge regular season and tournament results
    all_games = pd.concat([regular_season, tourney_results])
    all_games = all_games[(all_games['Season'] >= start_year) & 
                          (all_games['Season'] <= end_year)]
    
    # 初始化结果字典
    # Initialize result dictionary
    matchup_history = {season: {} for season in range(start_year, end_year + 1)}
    
    # 为每个对战创建唯一键
    # Create unique key for each matchup
    all_games['Team1'] = all_games[['WTeamID', 'LTeamID']].min(axis=1)
    all_games['Team2'] = all_games[['WTeamID', 'LTeamID']].max(axis=1)
    all_games['Team1Won'] = (all_games['WTeamID'] == all_games['Team1']).astype(int)
    
    # 使用groupby聚合计算对战历史
    # Use groupby aggregation to calculate matchup history
    grouped = all_games.groupby(['Season', 'Team1', 'Team2'])
    
    for (season, team1, team2), group in grouped:
        # 计算对战统计
        # Calculate matchup statistics
        games_count = len(group)
        team1_wins = group['Team1Won'].sum()
        
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
        matchup_history[season][matchup_key] = {
            'games': games_count,
            'wins_team1': team1_wins,
            'points_team1': team1_points,
            'points_team2': team2_points,
            'avg_point_diff': (team1_points - team2_points) / games_count
        }
    
    return matchup_history


def calculate_progression_probabilities(seed_features, team_stats):
    """
    Calculate progression probabilities for each team in various rounds
    计算各轮次中每支队伍的晋级概率
    """
    # 基于历史数据的种子晋级率（近似值）
    # Seed progression rates based on historical data (approximate values)
    seed_progression_rates = {
        1: [0.99, 0.93, 0.74, 0.52, 0.36, 0.22],  # 1号种子在每轮中的晋级率
        2: [0.96, 0.82, 0.56, 0.36, 0.23, 0.12],
        3: [0.94, 0.72, 0.44, 0.22, 0.12, 0.06],
        4: [0.92, 0.64, 0.34, 0.18, 0.08, 0.04],
        5: [0.85, 0.47, 0.24, 0.12, 0.04, 0.02],
        6: [0.76, 0.36, 0.18, 0.08, 0.03, 0.01],
        7: [0.72, 0.32, 0.15, 0.06, 0.02, 0.01],
        8: [0.56, 0.22, 0.10, 0.04, 0.01, 0.005],
        9: [0.44, 0.18, 0.08, 0.03, 0.01, 0.004],
        10: [0.68, 0.28, 0.12, 0.05, 0.02, 0.007],
        11: [0.59, 0.24, 0.10, 0.04, 0.01, 0.005],
        12: [0.36, 0.15, 0.06, 0.02, 0.007, 0.002],
        13: [0.20, 0.08, 0.03, 0.01, 0.003, 0.001],
        14: [0.16, 0.05, 0.02, 0.005, 0.001, 0.0005],
        15: [0.06, 0.02, 0.005, 0.001, 0.0003, 0.0001],
        16: [0.01, 0.003, 0.0005, 0.0001, 0.00002, 0.000005],
    }
    
    # 创建字典推导式初始化结果
    # Create dictionary comprehension to initialize results
    progression_probs = {}
    
    # 使用向量化操作计算晋级概率
    # Use vectorized operations to calculate progression probabilities
    for season in seed_features:
        season_probs = {}
        
        for team_id, seed_info in seed_features[season].items():
            seed_num = seed_info['seed_num']
            
            # 查找种子号对应的基础晋级率
            # Look up base progression rates for seed number
            base_rates = seed_progression_rates.get(
                seed_num, 
                seed_progression_rates[min(seed_progression_rates.keys(), key=lambda k: abs(k-seed_num))]
            )
            
            # 应用队伍实力调整
            # Apply team strength adjustments
            team_strength = team_stats.get(season, {}).get(team_id, {})
            
            # 计算实力调整因子
            # Calculate strength adjustment factor
            strength_factor = 1.0
            
            if team_strength:
                # 根据胜率调整
                # Adjust based on win rate
                win_rate_adj = (team_strength.get('win_rate', 0.5) - 0.5) * 0.3
                strength_factor += win_rate_adj
                
                # 根据得分差异调整
                # Adjust based on point difference
                point_diff_adj = team_strength.get('point_diff', 0) * 0.02
                strength_factor += point_diff_adj
                
                # 根据排名调整
                # Adjust based on ranking
                rank_adj = (1 - team_strength.get('normalized_rank', 0.5)) * 0.15
                strength_factor += rank_adj
            
            # 应用调整，但保持在合理范围内
            # Apply adjustment, but keep within reasonable range
            adjusted_rates = [min(0.999, max(0.001, rate * strength_factor)) for rate in base_rates]
            
            # 存储晋级概率
            # Store progression probabilities
            team_progression = {f'rd{i+1}_win': rate for i, rate in enumerate(adjusted_rates)}
            season_probs[team_id] = team_progression
        
        progression_probs[season] = season_probs
    
    return progression_probs


def convert_progression_to_matchup(team1_prog, team2_prog, round_num):
    """
    Convert progression probabilities to matchup probabilities
    将晋级概率转换为对战概率（实现goto_conversion概念）
    
    Parameters:
    -----------
    team1_prog : dict
        Team 1 progression probabilities
        队伍1的晋级概率
    team2_prog : dict
        Team 2 progression probabilities
        队伍2的晋级概率
    round_num : int
        Tournament round number
        锦标赛轮次编号
        
    Returns:
    --------
    float
        Probability of team1 winning against team2
        队伍1战胜队伍2的概率
    """
    # 获取当前轮次和前一轮次的键
    # Get current round and previous round keys
    curr_round_key = f'rd{round_num}_win'
    prev_round_key = f'rd{round_num-1}_win' if round_num > 1 else None
    
    # 获取条件晋级概率
    # Get conditional progression probabilities
    if round_num == 1 or prev_round_key is None:
        # 第一轮直接使用基础概率
        # For first round, use base probabilities directly
        team1_win_given_reach = team1_prog.get(curr_round_key, 0.5)
        team2_win_given_reach = team2_prog.get(curr_round_key, 0.5)
    else:
        # 计算条件概率：P(晋级到下一轮|已经到达当前轮)
        # Calculate conditional probability: P(advance to next round | already reached current round)
        prev_t1 = team1_prog.get(prev_round_key, 0.001)
        prev_t2 = team2_prog.get(prev_round_key, 0.001)
        
        curr_t1 = team1_prog.get(curr_round_key, 0.0005)
        curr_t2 = team2_prog.get(curr_round_key, 0.0005)
        
        # 安全地计算条件概率
        # Safely calculate conditional probabilities
        team1_win_given_reach = curr_t1 / prev_t1 if prev_t1 > 0 else 0.5
        team2_win_given_reach = curr_t2 / prev_t2 if prev_t2 > 0 else 0.5
    
    # 确保概率在有效范围内
    # Ensure probabilities are in valid range
    team1_win_given_reach = max(0.001, min(0.999, team1_win_given_reach))
    team2_win_given_reach = max(0.001, min(0.999, team2_win_given_reach))
    
    # 将条件概率转换为对战概率
    # Convert conditional probabilities to matchup probabilities
    raw_sum = team1_win_given_reach + team2_win_given_reach
    
    # 如果概率总和接近1，不需要太多调整
    # If probability sum is close to 1, not much adjustment needed
    if 0.95 <= raw_sum <= 1.05:
        # 简单归一化
        # Simple normalization
        team1_matchup_prob = team1_win_given_reach / raw_sum
    else:
        # 应用偏差校正（热门队伍被低估，冷门队伍被高估）
        # Apply bias correction (favorite teams underestimated, longshot teams overestimated)
        if team1_win_given_reach > team2_win_given_reach:
            # 队伍1是热门，给予额外权重
            # team1 is favorite, give extra weight
            ratio = team1_win_given_reach / team2_win_given_reach
            boosted_ratio = ratio ** 1.1  # 为热门队伍增加权重 (Boost for favorite teams)
            team1_matchup_prob = boosted_ratio / (1 + boosted_ratio)
        else:
            # 队伍2是热门
            # team2 is favorite
            ratio = team2_win_given_reach / team1_win_given_reach
            boosted_ratio = ratio ** 1.1  # 为热门队伍增加权重 (Boost for favorite teams)
            team1_matchup_prob = 1 / (1 + boosted_ratio)
    
    return team1_matchup_prob