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
import os
from functools import lru_cache

# 全局变量定义
MENS_TEAMS_IDS = set()
WOMENS_TEAMS_IDS = set()

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
    确定队伍属于男子队还是女子队
    
    Parameters:
        team_id (int): 队伍ID
        data_dict (dict): 原始数据字典
        
    Returns:
        str: 'men'、'women'或'unknown'
    """
    if data_dict is None:
        return 'unknown'
    
    m_teams = data_dict.get('m_teams', None)
    w_teams = data_dict.get('w_teams', None)
    
    if m_teams is not None and team_id in m_teams['TeamID'].values:
        return 'men'
    
    if w_teams is not None and team_id in w_teams['TeamID'].values:
        return 'women'
    
    return 'unknown'


def generate_all_possible_matchups(data_dict, gender='both', max_teams=None):
    """
    生成所有可能的队伍对阵组合，确保ID较小的队伍在前
    
    Parameters:
        data_dict (dict): 原始数据字典
        gender (str): 'men', 'women', 或 'both'
        max_teams (int): 每类队伍的最大数量，现在默认为None，使用全部队伍
        
    Returns:
        list or dict: 队伍对阵列表或字典
    """
    result = {}
    
    # 处理男子队伍
    if gender in ['men', 'both']:
        m_teams = data_dict.get('m_teams', None)
        if m_teams is not None:
            # 获取所有队伍ID并排序
            all_m_teams = sorted(m_teams['TeamID'].unique().tolist())
            
            # 生成所有组合，确保team1 < team2
            m_matchups = []
            for i, team1 in enumerate(all_m_teams):
                for team2 in all_m_teams[i+1:]:  # 只取更大的ID
                    m_matchups.append((team1, team2))
            
            print(f"生成了 {len(m_matchups)} 个男子队伍对阵组合")
            result['men'] = m_matchups
    
    # 处理女子队伍
    if gender in ['women', 'both']:
        w_teams = data_dict.get('w_teams', None)
        if w_teams is not None:
            # 获取所有队伍ID并排序
            all_w_teams = sorted(w_teams['TeamID'].unique().tolist())
            
            # 生成所有组合，确保team1 < team2
            w_matchups = []
            for i, team1 in enumerate(all_w_teams):
                for team2 in all_w_teams[i+1:]:  # 只取更大的ID
                    w_matchups.append((team1, team2))
            
            print(f"生成了 {len(w_matchups)} 个女子队伍对阵组合")
            result['women'] = w_matchups
    
    # 返回结果
    if gender == 'men':
        return result.get('men', [])
    elif gender == 'women':
        return result.get('women', [])
    else:
        return result


def batch_process_matchups(matchup_batch, features_dict, data_dict, model, model_columns, year, gender):
    """
    批量处理对阵预测，优化CPU利用和内存管理
    
    Parameters:
        matchup_batch (list): 待处理的对阵列表
        features_dict (dict): 特征数据字典
        data_dict (dict): 原始数据字典
        model: 预测模型
        model_columns: 模型所需列
        year (int): 预测年份
        gender (str): 'men'或'women'
    
    Returns:
        list: 处理结果列表
    """
    results = []
    for team1_id, team2_id in matchup_batch:
        # 确保team1_id < team2_id
        if team1_id > team2_id:
            team1_id, team2_id = team2_id, team1_id
            
        # 构建特征并预测
        matchup_features = build_matchup_features(
            team1_id, team2_id, features_dict, 
            target_year=year, round_num=2, gender=gender, data_dict=data_dict
        )
        
        # 处理可能的错误情况
        if matchup_features is None or len(matchup_features) == 0:
            results.append({
                'Team1': team1_id,
                'Team2': team2_id,
                'Pred': 0.5  # 默认预测值
            })
            continue
            
        # 确保特征格式正确
        if model_columns is not None:
            # 确保特征列顺序与模型期望的一致
            feature_values = []
            for col in model_columns:
                if col in matchup_features:
                    feature_values.append(matchup_features[col])
                else:
                    feature_values.append(0)  # 填充缺失特征
            
            # 转为numpy数组
            feature_values = np.array(feature_values).reshape(1, -1)
        else:
            feature_values = np.array(list(matchup_features.values())).reshape(1, -1)
        
        # 进行预测
        try:
            pred = model.predict_proba(feature_values)[0][1]
            
            # 添加微小随机扰动以避免预测值完全相同
            noise = np.random.uniform(-0.01, 0.01)
            pred = max(0.01, min(0.99, pred + noise))
            
            results.append({
                'Team1': team1_id,
                'Team2': team2_id,
                'Pred': pred
            })
        except Exception as e:
            print(f"预测错误 ({gender}): {e}")
            results.append({
                'Team1': team1_id,
                'Team2': team2_id,
                'Pred': 0.5  # 错误时使用默认值
            })
    
    return results


def worker_process_batches(worker_batch_list, features_dict, data_dict, model, model_columns, year, gender):
    """优化的工作进程函数，处理一批对阵"""
    results = []
    batch_size = len(worker_batch_list)
    
    # 添加进度指示器
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(worker_batch_list, desc=f"{gender}队伍对阵处理", leave=False)):
        batch_results = batch_process_matchups(
            batch, features_dict, data_dict, model, model_columns, year, gender
        )
        
        # 添加对批次结果的输出
        if i == 0:  # 只打印第一个批次的样本
            print(f"批次结果示例 ({gender}, 前5个):")
            for j, res in enumerate(batch_results[:5]):
                print(f"  {j+1}. Team1={res['Team1']}, Team2={res['Team2']}, Pred={res['Pred']:.4f}")
        
        results.extend(batch_results)
        
        # 添加定期进度报告
        if (i+1) % 10 == 0 or i == len(worker_batch_list) - 1:
            completion = (i+1) / len(worker_batch_list) * 100
            print(f"进度 ({gender}): {completion:.1f}% - 已完成 {i+1}/{len(worker_batch_list)} 批次")
            
    return results


def prepare_all_predictions(model, features_dict, data_dict, model_columns=None, year=2025, n_jobs=-1, gender='men'):
    """为所有可能的球队对阵准备预测结果，优化并行处理"""
    # 生成所有可能的对阵
    all_matchups = generate_all_possible_matchups(data_dict, gender=gender)
    
    print(f"为 {gender} 队伍生成 {len(all_matchups)} 个可能的对阵预测...")
    
    # 优化任务数量，避免使用过多CPU资源
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() - 1, 8)  # 限制最大并行数
    n_jobs = max(1, n_jobs)  # 确保至少有一个作业
    
    print(f"使用 {n_jobs} 个并行作业处理预测")
    
    # 分批处理，避免内存问题
    batch_size = calculate_optimal_batch_size(len(all_matchups), n_jobs)
    batched_matchups = []
    
    for i in range(0, len(all_matchups), batch_size):
        batch = all_matchups[i:i+batch_size]
        batched_matchups.append(batch)
    
    # 将批次分配给工作进程
    worker_batches = []
    for i in range(n_jobs):
        worker_batches.append([])
    
    # 均匀分配批次，确保负载均衡
    for i, batch in enumerate(batched_matchups):
        worker_idx = i % n_jobs
        worker_batches[worker_idx].append(batch)
    
    # 使用并行处理处理每个工作进程的批次
    try:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, timeout=3600)(
            delayed(worker_process_batches)(
                worker_batch, features_dict, data_dict, model, model_columns, year, gender
            )
            for worker_batch in worker_batches if worker_batch  # 跳过空批次
        )
        
        # 展平结果
        all_results = []
        for res_list in results:
            all_results.extend(res_list)
            
    except Exception as e:
        print(f"并行处理错误: {e}")
        print("回退到串行处理...")
        
        # 回退到串行处理
        all_results = []
        for worker_batch in worker_batches:
            if not worker_batch:
                continue
            batch_results = worker_process_batches(
                worker_batch, features_dict, data_dict, model, model_columns, year, gender
            )
            all_results.extend(batch_results)
    
    # 转换为DataFrame
    predictions_df = pd.DataFrame(all_results)
    
    # 确保列类型正确
    predictions_df['Team1'] = predictions_df['Team1'].astype(int)
    predictions_df['Team2'] = predictions_df['Team2'].astype(int)
    predictions_df['Pred'] = predictions_df['Pred'].astype(float)
    
    # 修改：添加ID列 - 直接使用年份，不需要替换
    predictions_df['ID'] = predictions_df.apply(
        lambda row: f"{year}_{int(row['Team1'])}_{int(row['Team2'])}", axis=1
    )
    
    # 检查ID格式
    print(f"生成的ID示例 ({gender}):")
    for i, id_val in enumerate(predictions_df['ID'].head(5).values):
        print(f"  {i+1}. {id_val}")
    
    # 检查预测值的分布
    if len(predictions_df) > 0:
        print(f"预测值统计 ({gender}):")
        print(f"  最小值: {predictions_df['Pred'].min():.4f}")
        print(f"  最大值: {predictions_df['Pred'].max():.4f}")
        print(f"  平均值: {predictions_df['Pred'].mean():.4f}")
        print(f"  标准差: {predictions_df['Pred'].std():.4f}")
    
    print(f"已完成 {len(predictions_df)} 个对阵的预测")
    
    # 校准预测值
    predictions_df = apply_prediction_calibration(predictions_df, gender)
    
    return predictions_df


def create_submission(predictions_df, sample_submission, data_dict=None, filename=None):
    """从预测结果创建提交文件"""
    # 验证输入
    if predictions_df is None or len(predictions_df) == 0:
        print("错误: 没有预测结果可用于创建提交文件")
        return None
        
    if sample_submission is None or 'ID' not in sample_submission.columns:
        print("错误: 样本提交文件无效")
        return None
    
    # 创建ID到预测的映射
    pred_dict = dict(zip(predictions_df['ID'], predictions_df['Pred']))
    
    print("预测数据概览:")
    print(f"  预测总数: {len(predictions_df)}")
    print(f"  ID格式样本: {predictions_df['ID'].iloc[0] if len(predictions_df) > 0 else 'N/A'}")
    print(f"  预测值范围: [{min(pred_dict.values()):.4f}, {max(pred_dict.values()):.4f}]")
    print(f"  预测值均值: {sum(pred_dict.values())/len(pred_dict):.4f}")
    
    # 显示样本ID和Pred
    print("\n预测数据样本 (前5个):")
    for i, (k, v) in enumerate(list(pred_dict.items())[:5]):
        print(f"  {k}: {v:.4f}")

    # 显示sample_submission样本
    print("\n样本提交文件 (前5行):")
    for i, id_val in enumerate(sample_submission['ID'].head(5).values):
        print(f"  {id_val}")
    
    # 创建新的提交文件，保持与样本提交文件相同的结构
    submission = sample_submission.copy()
    
    # 定义获取预测的函数
    def get_prediction(id_str):
        # 直接在pred_dict中查找ID
        if id_str in pred_dict:
            return pred_dict[id_str]
        
        # 尝试交换team1和team2顺序
        parts = id_str.split('_')
        if len(parts) == 3:  # 格式: YYYY_team1_team2
            try:
                team1 = int(parts[1])
                team2 = int(parts[2])
                
                # 确保team1 < team2的格式
                if team1 > team2:
                    alt_id = f"{parts[0]}_{team2}_{team1}"
                    if alt_id in pred_dict:
                        # 对于交换了team1和team2的情况，需要反转概率
                        return 1.0 - pred_dict[alt_id]
            except (ValueError, TypeError):
                pass
        
        # 如果找不到匹配的预测，使用中性值并添加小随机噪声
        print(f"警告: ID {id_str} 未找到对应预测，使用默认值")
        return 0.5 + np.random.uniform(-0.05, 0.05)
    
    # 使用get_prediction函数填充Pred列
    submission['Pred'] = submission['ID'].apply(get_prediction)
    
    # 打印一些关键ID的映射，帮助调试
    print("\n样本ID映射示例：")
    for i, id_str in enumerate(submission['ID'].head(5).values):
        print(f"  {i+1}. ID: {id_str} -> 预测值: {submission['Pred'].iloc[i]:.4f}")
    
    # 验证提交文件
    print(f"\n提交文件概览:")
    print(f"  行数: {len(submission)}")
    print(f"  预测值范围: [{submission['Pred'].min():.4f}, {submission['Pred'].max():.4f}]")
    print(f"  预测值均值: {submission['Pred'].mean():.4f}")
    print(f"  预测值标准差: {submission['Pred'].std():.4f}")
    
    # 检查是否所有值都相同
    if submission['Pred'].std() < 0.001:
        print("警告：预测值几乎没有变化，请检查数据处理流程")
    
    # 保存提交文件
    if filename:
        submission.to_csv(filename, index=False)
        print(f"提交文件已保存至 {filename}")
    
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
    
    # 检查行数
    if len(submission_df) != len(data_dict['sample_sub']):
        print(f"警告: 提交文件包含 {len(submission_df)} 行，样本提交包含 {len(data_dict['sample_sub'])} 行")
    
    # 解析ID获取队伍编号 - 使用正确的2025格式
    team_pairs = []
    valid_format = True
    incorrect_ids = []
    
    # 检查是否所有ID都以2025开头
    year_check = all(id_str.startswith('2025_') for id_str in submission_df['ID'])
    if not year_check:
        non_2025_ids = [id_str for id_str in submission_df['ID'] if not id_str.startswith('2025_')]
        print(f"错误: 发现{len(non_2025_ids)}个不以'2025_'开头的ID")
        if len(non_2025_ids) > 0:
            print(f"示例: {non_2025_ids[:5]}")
        valid_format = False
    
    # 检查ID格式
    for id_str in submission_df['ID']:
        parts = id_str.split('_')
        if len(parts) == 3:  # 正确的格式: 2025_team1_team2
            try:
                year = parts[0]
                if year != '2025':
                    incorrect_ids.append(id_str)
                    continue
                    
                team1 = int(parts[1])
                team2 = int(parts[2])
                
                # 验证队伍ID顺序
                if team1 >= team2:
                    print(f"错误: ID {id_str} 中 team1 >= team2")
                    valid_format = False
                
                team_pairs.append((team1, team2))
            except (ValueError, IndexError):
                incorrect_ids.append(id_str)
                valid_format = False
        else:
            incorrect_ids.append(id_str)
            valid_format = False
    
    if len(incorrect_ids) > 0:
        print(f"错误: 发现{len(incorrect_ids)}个格式不正确的ID，应为'2025_team1_team2'")
        if len(incorrect_ids) > 0:
            print(f"示例: {incorrect_ids[:5]}")
        valid_format = False
    
    # 验证预测值范围
    valid_range = ((submission_df['Pred'] >= 0) & (submission_df['Pred'] <= 1)).all()
    if not valid_range:
        out_of_range = submission_df[~((submission_df['Pred'] >= 0) & (submission_df['Pred'] <= 1))]
        print(f"错误: 有{len(out_of_range)}个预测值不在[0,1]范围内")
        if len(out_of_range) > 0:
            print(f"示例: {out_of_range['ID'].iloc[0]} -> {out_of_range['Pred'].iloc[0]}")
        return False
    
    if valid_format:
        print("提交文件验证通过!")
    
    return valid_format


@lru_cache(maxsize=10000)
def cached_build_matchup_features(team1, team2, target_year, round_num, gender):
    """缓存版本的特征构建函数，避免重复计算"""
    # 这个函数需要修改原始函数来使用，因为lru_cache需要可哈希参数
    # ...

def calculate_optimal_batch_size(total_items, n_jobs, available_memory_mb=None):
    """动态计算最优批次大小，根据数据量、作业数和可用内存"""
    # 如果未指定可用内存，尝试获取系统内存
    if available_memory_mb is None:
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            # 无法获取系统内存时使用默认值
            available_memory_mb = 2000  # 默认2GB
    
    # 每项目估计内存使用量(MB)
    est_memory_per_item = 0.01  # 假设每个对阵项占用10KB
    
    # 根据可用内存和作业数计算安全的批次大小
    memory_safe_batch = int(available_memory_mb / (est_memory_per_item * n_jobs * 1.5))
    
    # 基本批次大小
    base_size = 100
    
    # 根据数据量调整
    if total_items > 10000:
        base_size = 200
    elif total_items > 5000:
        base_size = 150
    
    # 取两种方法计算结果的较小值，确保内存安全
    optimal_size = min(base_size, memory_safe_batch)
    
    # 确保每个作业至少有一个批次
    return max(optimal_size, total_items // (n_jobs * 10))

def apply_prediction_calibration(predictions_df, gender):
    """校准预测值，保留原始分布形状但调整均值到合理范围"""
    # 确保所有预测值在[0.01, 0.99]范围内
    predictions_df['Pred'] = predictions_df['Pred'].clip(0.01, 0.99)
    
    # 如果是男队且预测值过低，或女队预测值过高，进行校准
    if gender == 'men' and predictions_df['Pred'].mean() < 0.45:
        print(f"校准{gender}队伍预测值 (当前平均: {predictions_df['Pred'].mean():.4f})")
        # 使用均值移位而非范围压缩，保留原始分布形状
        current_mean = predictions_df['Pred'].mean()
        shift = 0.5 - current_mean  # 移动到0.5均值
        predictions_df['Pred'] = predictions_df['Pred'] + shift
        # 确保所有值在[0.01, 0.99]范围内
        predictions_df['Pred'] = predictions_df['Pred'].clip(0.01, 0.99)
        print(f"校准后平均: {predictions_df['Pred'].mean():.4f}")
    elif gender == 'women' and predictions_df['Pred'].mean() > 0.55:
        print(f"校准{gender}队伍预测值 (当前平均: {predictions_df['Pred'].mean():.4f})")
        # 使用均值移位而非范围压缩
        current_mean = predictions_df['Pred'].mean()
        shift = 0.5 - current_mean
        predictions_df['Pred'] = predictions_df['Pred'] + shift
        # 确保所有值在[0.01, 0.99]范围内
        predictions_df['Pred'] = predictions_df['Pred'].clip(0.01, 0.99)
        print(f"校准后平均: {predictions_df['Pred'].mean():.4f}")
    
    return predictions_df

# 添加最终验证函数
def validate_final_submission(submission_file):
    """验证提交文件是否符合竞赛格式要求"""
    try:
        # 读取提交文件
        submission = pd.read_csv(submission_file)
        print(f"提交文件包含 {len(submission)} 行")
        
        # 检查列名
        if not all(col in submission.columns for col in ['ID', 'Pred']):
            print("错误: 提交文件必须包含'ID'和'Pred'列")
            return False
        
        # 检查ID格式
        id_format_valid = True
        year_check = all(id_str.startswith('2025_') for id_str in submission['ID'])
        
        if not year_check:
            non_2025_ids = [id_str for id_str in submission['ID'] if not id_str.startswith('2025_')]
            print(f"错误: 发现{len(non_2025_ids)}个不以'2025_'开头的ID")
            if len(non_2025_ids) > 0:
                print(f"示例: {non_2025_ids[:5]}")
            id_format_valid = False
        
        # 更详细地检查ID格式
        format_issues = []
        for id_str in submission['ID'][:100]:  # 仅检查前100个以提高效率
            parts = id_str.split('_')
            if len(parts) != 3 or not parts[0] == '2025':
                format_issues.append(id_str)
                if len(format_issues) >= 5:
                    break
        
        if format_issues:
            print(f"错误: ID格式不正确 - {format_issues[0]}，应为'2025_TeamID1_TeamID2'")
            id_format_valid = False
        
        # 检查预测值范围
        pred_range_valid = (submission['Pred'] >= 0).all() and (submission['Pred'] <= 1).all()
        if not pred_range_valid:
            out_of_range = submission[~((submission['Pred'] >= 0) & (submission['Pred'] <= 1))]
            print(f"错误: 有{len(out_of_range)}个预测值不在[0,1]范围内")
            if len(out_of_range) > 0:
                print(f"示例: {out_of_range['ID'].iloc[0]} -> {out_of_range['Pred'].iloc[0]}")
            return False
        
        # 显示预测值分布统计
        print(f"预测值分布: 最小={submission['Pred'].min():.4f}, 平均={submission['Pred'].mean():.4f}, 最大={submission['Pred'].max():.4f}")
        
        if id_format_valid and pred_range_valid:
            print("✓ 提交文件验证通过！符合2025比赛要求")
        
        return id_format_valid and pred_range_valid
    
    except Exception as e:
        print(f"验证提交文件时出错: {e}")
        return False