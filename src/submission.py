#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NCAA Basketball Tournament Prediction Model - Prediction and Submission

This module handles generating predictions for tournament games and creating submission files.

Author: Junming Zhao
Date: 2025-03-13    
Version: 2.0 ### 增加了针对女队的预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
from evaluate import apply_brier_optimal_strategy
from feature_engineering import convert_progression_to_matchup
from train_model import add_favorite_longshot_features


def check_sample_submission_format(sample_submission):
    """Check the ID format in the sample submission file"""
    if sample_submission is None or 'ID' not in sample_submission.columns:
        print("警告：样本提交文件缺失或格式不正确")
        return False
        
    ids = sample_submission['ID'].values
    print("Sample ID format check:")
    
    # Print first 10 IDs
    print("First 10 ID examples:")
    for i in range(min(10, len(ids))):
        print(f"  {i+1}. {ids[i]}")
    
    # Analyze ID format
    patterns = {}
    id_parts = {}
    
    for id_str in ids[:1000]:  # Analyze first 1000 IDs
        parts = id_str.split('_')
        pattern = f"{len(parts)} parts"
        
        if pattern in patterns:
            patterns[pattern] += 1
        else:
            patterns[pattern] = 1
            
        # Record character type at each position
        for i, part in enumerate(parts):
            pos_key = f"position{i}"
            if pos_key not in id_parts:
                id_parts[pos_key] = {}
            
            if part.isdigit():
                type_key = "numeric"
            elif part.lower() in ['men', 'women', 'm', 'w']:
                type_key = "gender"
            else:
                type_key = "other"
                
            if type_key in id_parts[pos_key]:
                id_parts[pos_key][type_key] += 1
            else:
                id_parts[pos_key][type_key] = 1
    
    # Print pattern analysis
    print("\nID pattern analysis:")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} IDs ({count/len(ids)*100:.1f}%)")
    
    # Print position analysis
    print("\nID position content analysis:")
    for pos, types in id_parts.items():
        print(f"  {pos}:")
        for type_key, count in types.items():
            print(f"    {type_key}: {count} occurrences")
            
    return True


def build_matchup_features(team1, team2, features_dict, target_year=2025, round_num=2, gender='men', data_dict=None):
    """
    构建队伍对阵的特征
    
    参数:
    team1: 第一支队伍ID
    team2: 第二支队伍ID
    features_dict: 特征字典
    target_year: 目标年份
    round_num: 锦标赛轮次
    gender: 性别 ('men'或'women')
    data_dict: 数据字典(可选)
    
    返回:
    dict: 对阵特征
    """
    # 确保team1 < team2，符合比赛要求
    swap_teams = False
    if team1 > team2:
        team1, team2 = team2, team1
        swap_teams = True
    
    # 获取最近可用赛季的特征
    latest_season = max(features_dict['team_stats'].keys()) if 'team_stats' in features_dict and features_dict['team_stats'] else target_year-1
    team_stats = features_dict.get('team_stats', {}).get(latest_season, {})
    seed_features = features_dict.get('seed_features', {}).get(latest_season, {})
    progression_probs = features_dict.get('progression_probs', {}).get(latest_season, {})
    matchup_history = features_dict.get('matchup_history', {})
    
    # 收集两支队伍的特征
    features = {}
    
    # 基本特征 - 队伍ID
    features['team1_id'] = team1
    features['team2_id'] = team2
    
    # 性别特征 - 如果提供了data_dict则使用
    match_gender = gender  # 默认使用传入的性别参数
    if data_dict is not None:
        team1_gender = identify_team_gender(team1, data_dict)
        team2_gender = identify_team_gender(team2, data_dict)
        features['team1_gender'] = 1 if team1_gender == 'men' else 0
        features['team2_gender'] = 1 if team2_gender == 'men' else 0
        
        # 检查队伍性别是否一致，如果不一致使用第一支队伍的性别
        if team1_gender != team2_gender:
            print(f"警告：队伍性别不匹配 - team1({team1}): {team1_gender}, team2({team2}): {team2_gender}，使用{team1_gender}作为匹配性别")
            match_gender = team1_gender
    
    # 队伍统计数据
    for team_id, prefix in [(team1, 'team1'), (team2, 'team2')]:
        if team_id in team_stats:
            stats = team_stats[team_id]
            # 添加所有可用统计数据
            for stat_name, stat_value in stats.items():
                features[f'{prefix}_{stat_name}'] = stat_value
        else:
            # 扩展默认值列表
            default_stats = {
                'win_rate': 0.5,
                'point_diff': 0.0,
                'num_wins': 0,
                'num_losses': 0,
                'total_games': 0,
                'home_win_rate': 0.5,
                'away_win_rate': 0.5,
                'recent_win_rate': 0.5,
                'momentum': 0.0,
                'season_rank': 0,
                'normalized_rank': 0.5
            }
            # 添加所有默认值
            for stat_name, stat_value in default_stats.items():
                features[f'{prefix}_{stat_name}'] = stat_value
    
    # 种子特征
    for team_id, prefix in [(team1, 'team1'), (team2, 'team2')]:
        if team_id in seed_features:
            seed_info = seed_features[team_id]
            features[f'{prefix}_seed'] = seed_info['seed_num']
            features[f'{prefix}_region'] = ord(seed_info['region']) - ord('A')  # 转换为数值
        else:
            # 默认值
            features[f'{prefix}_seed'] = 16  # 假设最低种子
            features[f'{prefix}_region'] = 0
    
    # 计算特征差异
    features['seed_diff'] = features.get('team1_seed', 16) - features.get('team2_seed', 16)
    features['win_rate_diff'] = features.get('team1_win_rate', 0.5) - features.get('team2_win_rate', 0.5)
    features['point_diff_diff'] = features.get('team1_point_diff', 0.0) - features.get('team2_point_diff', 0.0)
    
    # 历史对阵
    matchup_key = (team1, team2)
    for season, matchups in matchup_history.items():
        if matchup_key in matchups:
            history = matchups[matchup_key]
            features['previous_matchups'] = history['games']
            features['team1_win_rate_h2h'] = history['wins_team1'] / history['games'] if history['games'] > 0 else 0.5
            features['avg_point_diff_h2h'] = history['avg_point_diff']
            break
    else:
        # 无历史对阵
        features['previous_matchups'] = 0
        features['team1_win_rate_h2h'] = 0.5
        features['avg_point_diff_h2h'] = 0.0
    
    # 晋级概率
    if progression_probs:
        team1_prog = progression_probs.get(team1, {})
        team2_prog = progression_probs.get(team2, {})
        features['team1_rd_win_prob'] = team1_prog.get(f'rd{round_num}_win', 0.5)
        features['team2_rd_win_prob'] = team2_prog.get(f'rd{round_num}_win', 0.5)
        
        # 计算对阵概率
        features['progression_based_prob'] = convert_progression_to_matchup(
            team1_prog, team2_prog, round_num
        )
    
    # 如果交换了队伍，不需要额外处理
    # 因为我们确保了team1 < team2，这对应于要求的预测格式
    
    return features


# 添加新函数用于识别队伍ID性别
def identify_team_gender(team_id, data_dict=None):
    """
    识别队伍ID是属于男队还是女队
    Identify whether a team ID belongs to men's or women's teams
    
    参数:
        team_id (int): 要检查的队伍ID
        data_dict (dict): 包含队伍数据的字典
        
    返回:
        str: 'men'或'women'
    """
    if data_dict is None:
        # 如果没有提供数据字典，尝试基于ID范围猜测性别
        print("警告：没有提供数据字典，尝试基于ID范围猜测队伍性别")
        # 一些锦标赛中的常见ID范围
        if 1000 <= team_id <= 1999:
            return 'men'
        elif 3000 <= team_id <= 3999:
            return 'women'
        else:
            print(f"无法确定队伍ID {team_id} 的性别，默认为男队")
            return 'men'
    
    # 检查男队字典
    m_teams = data_dict.get('m_teams', None)
    if m_teams is not None and team_id in m_teams['TeamID'].values:
        return 'men'
    
    # 检查女队字典
    w_teams = data_dict.get('w_teams', None)
    if w_teams is not None and team_id in w_teams['TeamID'].values:
        return 'women'
    
    # 如果无法确定，记录警告并返回默认值
    print(f"警告: 无法在提供的数据中确定队伍ID {team_id} 的性别，默认为男队")
    return 'men'


def generate_all_possible_matchups(data_dict, gender='both'):
    """
    Generate all possible matchups for tournament prediction
    生成锦标赛预测的所有可能对战
    
    参数:
        data_dict: 数据字典
        gender: 'men', 'women', or 'both'
        
    返回:
        list: 对战信息列表
    """
    # 处理性别参数
    if gender == 'both':
        gender_list = ['men', 'women']
    else:
        gender_list = [gender]
    
    all_matchups = []
    
    for g in gender_list:
        # 确定对应的队伍数据集
        if g == 'men':
            teams_df = data_dict.get('m_teams', None)
            # 优化：尝试从锦标赛种子中获取团队，减少不必要的对战组合
            seeds_df = data_dict.get('m_tourney_seeds', None)
            prefix = 'M'
        else:
            teams_df = data_dict.get('w_teams', None)
            seeds_df = data_dict.get('w_tourney_seeds', None)
            prefix = 'W'
        
        if teams_df is None:
            print(f"警告: 未找到{g}性别的队伍数据")
            continue
        
        # 优化：如果有种子数据，优先使用种子队伍（更可能进入锦标赛）
        if seeds_df is not None and not seeds_df.empty:
            # 获取最新赛季种子队伍
            latest_season = seeds_df['Season'].max()
            recent_seeds = seeds_df[seeds_df['Season'] >= (latest_season - 2)]
            team_ids = recent_seeds['TeamID'].unique().tolist()
            print(f"使用最近种子数据: 为{g}性别选择了 {len(team_ids)} 支队伍")
        else:
            # 如果无种子数据，使用全部队伍
            team_ids = teams_df['TeamID'].tolist()
        
        if len(team_ids) > 100:
            print(f"警告: {g}性别的队伍数量({len(team_ids)})很多，可能导致生成过多对战")
            print(f"考虑限制队伍数量以提高性能")
            # 优化：限制队伍数量
            if len(team_ids) > 150:  # 如果超过150支队伍，保留最新的150支
                print(f"仅使用前150支队伍以提高性能")
                team_ids = team_ids[:150]
            
        # 优化：使用numpy的组合函数生成对战，比嵌套循环更高效
        from itertools import combinations
        
        # 使用combinations直接生成对战组合
        team_pairs = list(combinations(sorted(team_ids), 2))
        
        # 批量创建字典，而不是一个一个添加
        batch_matchups = [
            {
                'team1': min(t1, t2),
                'team2': max(t1, t2),
                'round': 2,
                'gender': g,
                'prefix': prefix
            }
            for t1, t2 in team_pairs
        ]
        
        all_matchups.extend(batch_matchups)
    
    print(f"总共生成了 {len(all_matchups)} 个可能的对战")
    return all_matchups


def prepare_all_predictions(model, features_dict, data_dict, model_columns=None, year=2025, n_jobs=-1, gender='men'):
    """
    Prepare predictions for all possible matchups in the tournament
    准备锦标赛中所有可能对战的预测结果
    
    参数:
        model: 训练好的模型
        features_dict: 特征字典
        data_dict: 数据字典
        model_columns: 模型使用的列名
        year: 预测年份
        n_jobs: 并行作业数量
        gender: 'men', 'women', or 'both'
        
    返回:
        pandas.DataFrame: 包含所有预测的数据框
    """
    from joblib import Parallel, delayed
    import pandas as pd
    from tqdm import tqdm
    
    # 处理gender参数 - 确保处理'both'选项
    gender_list = []
    if gender == 'both':
        gender_list = ['men', 'women']
    else:
        gender_list = [gender]
    
    all_predictions = []
    
    # 对每种性别单独处理
    for g in gender_list:
        print(f"Generating predictions for {g}'s teams...")
        
        # 生成所有可能的对战
        matchups = generate_all_possible_matchups(data_dict, gender=g)
        
        if len(matchups) == 0:
            print(f"警告: 没有找到{g}性别的对战数据")
            continue
            
        print(f"Processing {len(matchups)} possible matchups...")
        
        def process_matchup(matchup_info):
            try:
                team1, team2 = matchup_info['team1'], matchup_info['team2']
                round_num = matchup_info.get('round', 2)  # 默认为第2轮
                
                # 构建特征
                X = build_matchup_features(team1, team2, features_dict, target_year=year, 
                                           round_num=round_num, gender=g, data_dict=data_dict)
                
                # 如果有指定的列，确保特征与模型期望的匹配
                if model_columns is not None:
                    # 创建一个新的包含所有必需列的特征字典
                    matched_X = {}
                    
                    # 仅保留模型需要的列，按照顺序
                    for col in model_columns:
                        if col in X:
                            matched_X[col] = X[col]
                        else:
                            matched_X[col] = 0  # 使用默认值0填充缺失特征
                    
                    # 使用匹配的特征集而不是原始的
                    X = matched_X
                
                # 预测概率
                if hasattr(model, 'predict_proba'):
                    # 分类模型
                    team1_win_prob = model.predict_proba(pd.DataFrame([X])[model_columns])[0][1]
                else:
                    # 回归模型
                    team1_win_prob = float(model.predict(pd.DataFrame([X])[model_columns])[0])
                    # 确保概率在有效范围内
                    team1_win_prob = max(0.0001, min(0.9999, team1_win_prob))
                
                # 创建提交ID字符串
                if g == 'men':
                    id_prefix = "M"
                else:
                    id_prefix = "W"
                    
                id_str = f"{year}_{id_prefix}_{team1}_{team2}"
                
                return {
                    'ID': id_str,
                    'Pred': team1_win_prob
                }
            except Exception as e:
                print(f"Error processing matchup {matchup_info}: {str(e)}")
                # 返回空占位符结果
                return None
        
        # 并行处理所有对战
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_matchup)(matchup_info) 
            for matchup_info in tqdm(matchups, desc=f"Processing {g}'s matchups")
        )
        
        # 过滤掉空结果并添加到所有预测中
        valid_results = [r for r in results if r is not None]
        all_predictions.extend(valid_results)
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    if len(predictions_df) == 0:
        print("警告: 没有生成任何预测结果!")
    else:
        print(f"完成: 生成了 {len(predictions_df)} 个预测结果")
        
    return predictions_df


def create_submission(predictions_df, sample_submission, filename=None):
    """
    Create submission file from predictions
    从预测结果创建提交文件
    
    参数:
        predictions_df: 包含预测的数据框
        sample_submission: 样本提交文件
        filename: 输出文件名，如果为None则使用时间戳
        
    返回:
        pandas.DataFrame: 提交数据框
    """
    # 验证输入
    if predictions_df is None or len(predictions_df) == 0:
        print("错误: 没有预测结果可用于创建提交文件")
        return None
        
    if sample_submission is None or 'ID' not in sample_submission.columns:
        print("错误: 样本提交文件无效")
        return None
    
    # 创建ID到预测的映射
    pred_dict = dict(zip(predictions_df['ID'], predictions_df['Pred']))
    
    # 使用sample_submission的ID顺序创建新的提交
    submission = sample_submission.copy()
    
    def get_prediction(id_str):
        """从预测字典中获取预测值，并处理不同的ID格式"""
        # 直接查找
        if id_str in pred_dict:
            return pred_dict[id_str]
            
        # 尝试解析ID格式
        parts = id_str.split('_')
        if len(parts) >= 4:  # 至少包含年份、性别和两个队伍ID
            year = parts[0]
            gender_code = parts[1]
            team1 = int(parts[2])
            team2 = int(parts[3])
            
            # 确保team1 < team2
            if team1 > team2:
                team1, team2 = team2, team1
                
            # 重新构建标准格式ID
            std_id = f"{year}_{gender_code}_{team1}_{team2}"
            
            if std_id in pred_dict:
                pred = pred_dict[std_id]
                # 如果我们交换了队伍顺序，需要翻转概率
                if team1 != int(parts[2]):
                    pred = 1 - pred
                return pred
        
        # 如果找不到匹配的预测，返回默认值0.5
        print(f"警告: ID '{id_str}' 没有找到匹配的预测，使用默认值0.5")
        return 0.5
    
    # 使用向量化操作填充预测值
    submission['Pred'] = submission['ID'].apply(get_prediction)
    
    # 应用Brier最优策略校准概率
    try:
        from evaluate import apply_brier_optimal_strategy
        submission = apply_brier_optimal_strategy(submission)
        print("已应用Brier最优策略校准概率")
    except Exception as e:
        print(f"无法应用Brier最优策略: {str(e)}")
    
    # 保存到文件
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"
    
    submission.to_csv(filename, index=False)
    print(f"提交文件已保存到: {filename}")
    
    return submission


def validate_submission(submission_df, data_dict):
    """
    验证提交文件是否符合2025比赛要求
    
    参数:
    submission_df: 提交文件DataFrame
    data_dict: 数据字典
    
    返回:
    bool: 验证结果
    """
    m_teams = data_dict['m_teams']['TeamID'].unique()
    w_teams = data_dict['w_teams']['TeamID'].unique()
    
    expected_m_pairs = len(m_teams) * (len(m_teams) - 1) // 2
    expected_w_pairs = len(w_teams) * (len(w_teams) - 1) // 2
    expected_total = expected_m_pairs + expected_w_pairs
    
    # 检查行数
    if len(submission_df) != len(data_dict['sample_sub']):
        print(f"警告: 提交文件包含 {len(submission_df)} 行，样本提交包含 {len(data_dict['sample_sub'])} 行")
    
    # 解析ID获取队伍编号
    team_pairs = submission_df['ID'].str.split('_').apply(lambda x: [int(i) for i in x])
    
    # 验证ID格式和顺序
    valid_format = all(len(pair) == 2 for pair in team_pairs)
    if not valid_format:
        print("错误: 某些ID格式不正确，应为'team1_team2'")
        return False
    
    valid_order = all(pair[0] < pair[1] for pair in team_pairs)
    if not valid_order:
        print("错误: 某些ID中team1 > team2")
        return False
    
    # 验证预测值范围
    valid_range = ((submission_df['Pred'] >= 0) & (submission_df['Pred'] <= 1)).all()
    if not valid_range:
        print("错误: 预测值必须在[0,1]范围内")
        return False
    
    print("提交文件验证通过!")
    return True