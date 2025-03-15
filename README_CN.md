# NCAA篮球锦标赛预测系统

## 简介

NCAA篮球锦标赛预测系统是一个全面的机器学习解决方案，旨在高精度预测NCAA篮球锦标赛比赛结果。该系统实现了一个复杂的预测流程，包括处理历史篮球数据、工程化相关特征、训练XGBoost模型，以及为锦标赛对阵生成经过校准的胜率预测。

## 系统要求

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- joblib
- tqdm
- concurrent.futures

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/ncaa-prediction-system.git
cd ncaa-prediction-system

# 创建虚拟环境（可选但推荐）
python -m venv myenv
source myenv/bin/activate  # Windows系统上：myenv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 代码结构

项目分为多个模块，每个模块处理预测流程的特定方面：

- **main.py**：编排整个工作流程并提供命令行接口
- **data_preprocessing.py**：处理数据加载、探索和训练-验证集划分
- **feature_engineering.py**：从原始数据创建特征（团队统计、种子、对战记录）
- **train_model.py**：实现带有超参数调优的XGBoost模型训练
- **submission.py**：生成锦标赛预测结果以供提交
- **evaluate.py**：包含评估指标和可视化工具
- **utils.py**：提供整个系统中使用的工具函数

## 使用方法

### 基本用法

```bash
python main.py --data_path ./data --output_path ./output --target_year 2025
```

### 高级选项

```bash
python main.py --data_path ./data \
               --output_path ./output \
               --train_start_year 2010 \
               --train_end_year 2024 \
               --target_year 2025 \
               --explore \
               --random_seed 42 \
               --n_cores 8
```

### 命令行参数

- `--data_path`：数据目录路径（默认：'./data'）
- `--output_path`：输出文件路径（默认：'./output'）
- `--train_start_year`：训练数据起始年份（默认：2010）
- `--train_end_year`：训练数据结束年份（默认：2024）
- `--target_year`：预测目标年份（默认：2025）
- `--explore`：启用数据探索（默认：False）
- `--load_model`：加载预训练模型而不是训练新模型（默认：False）
- `--load_features`：加载预计算特征而不是重新计算（默认：False）
- `--random_seed`：随机种子，用于结果可重现（默认：42）
- `--n_cores`：并行处理使用的CPU核心数（默认：自动检测）
- `--clear_cache`：清除计算缓存（默认：False）

## 数据要求

系统需要在数据目录中提供以下CSV文件：

- **MTeams.csv**：男子队伍信息
- **MRegularSeasonCompactResults.csv**：男子常规赛结果
- **MNCAATourneyCompactResults.csv**：男子锦标赛结果
- **MRegularSeasonDetailedResults.csv**：男子常规赛详细统计
- **MNCAATourneySeeds.csv**：男子锦标赛种子信息
- **SampleSubmissionStage1.csv**：样本提交格式

## 关键特性

### 高级特征工程

- 团队性能统计计算
- 种子信息处理
- 历史对战分析
- 锦标赛晋级概率估计
- 热门-冷门偏差校正

### 性能优化

- 计算密集型操作并行处理
- 内存缓存以避免重复计算
- 矢量化操作提高效率
- 大型数据集的内存使用优化

### 健壮的评估

- 多种指标（Brier分数、对数损失、准确率）
- 校准曲线分析
- 预测分布可视化
- 基于Brier分数特性的风险优化提交策略

## 预测流程

1. **数据加载**：加载并预处理历史篮球数据
2. **特征工程**：从原始数据创建预测特征
3. **模型训练**：训练具有优化超参数的XGBoost模型
4. **评估**：使用多种指标评估模型性能
5. **预测生成**：为锦标赛对阵创建预测
6. **风险策略应用**：为Brier分数应用最优风险策略
7. **提交创建**：格式化预测以供竞赛提交

## 理论洞察

系统实现了几个理论洞察以提高预测准确性：

- **Brier分数优化**：对于胜率约为33.3%的预测，应用策略性风险调整以优化预期Brier分数。
- **热门-冷门偏差校正**：系统校正了对强队（低种子）的系统性低估和对弱队（高种子）的高估。
- **时间感知验证**：使用较新赛季进行验证，以更好地反映篮球预测的时间性质。

## 示例结果

系统生成几个输出文件：

- 训练模型文件（xgb_model.pkl）
- 特征缓存（features.pkl）
- 预测提交文件（submission_YYYYMMDD_HHMMSS.csv）
- 模型评估指标（model_metrics_YYYYMMDD_HHMMSS.txt）
- 可视化结果（如果启用）

## 高级用法

### 训练自定义模型

```python
from train_model import build_xgboost_model
from utils import save_model

# 训练自定义模型
xgb_model, model_columns = build_xgboost_model(
    X_train, y_train, X_val, y_val, 
    random_seed=42,
    param_tuning=True,
    visualize=True
)

# 保存模型
save_model(xgb_model, 'custom_model.pkl', model_columns)
```

### 生成预测

```python
from submission import prepare_tournament_predictions, create_submission
from utils import load_model, load_features

# 加载模型和特征
model, model_columns = load_model('model.pkl')
features_dict = load_features('features.pkl')

# 生成预测
predictions = prepare_tournament_predictions(
    model, features_dict, sample_submission, model_columns, year=2025
)

# 创建提交文件
submission = create_submission(predictions, sample_submission, 'my_submission.csv')
```

## 性能注意事项

- 特征工程是流程中最耗时的部分；使用`--load_features`标志可重用先前计算的特征。
- 并行处理显著提高性能但增加内存使用。
- 对于极大型数据集，调整代码以使用分块处理或减少日期范围。

## 参考资料

- XGBoost：[https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- Brier分数：[https://en.wikipedia.org/wiki/Brier_score](https://en.wikipedia.org/wiki/Brier_score)
- NCAA锦标赛：[https://www.ncaa.com/march-madness](https://www.ncaa.com/march-madness)

## 作者

赵俊茗 (Junming Zhao)

## 许可证

MIT许可证

---

本README提供了NCAA篮球锦标赛预测系统的全面概述，包括设置说明、使用示例和关键技术细节。如有问题或贡献，请在仓库中提出issue。