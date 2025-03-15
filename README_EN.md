# NCAA Basketball Tournament Prediction System

## Introduction

The NCAA Basketball Tournament Prediction System is a comprehensive machine learning solution designed to predict the outcomes of NCAA basketball tournament games with high accuracy. This system implements a sophisticated prediction pipeline that processes historical basketball data, engineers relevant features, trains an XGBoost model, and generates calibrated win probability predictions for tournament matchups.

## System Requirements

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

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ncaa-prediction-system.git
cd ncaa-prediction-system

# Create a virtual environment (optional but recommended)
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Code Structure

The project is organized into several modules, each handling a specific aspect of the prediction pipeline:

- **main.py**: Orchestrates the entire workflow and provides command-line interface
- **data_preprocessing.py**: Handles data loading, exploration, and train-validation splitting
- **feature_engineering.py**: Creates features from raw data (team stats, seeds, matchups)
- **train_model.py**: Implements XGBoost model training with hyperparameter tuning
- **submission.py**: Generates tournament predictions for submission
- **evaluate.py**: Contains evaluation metrics and visualization tools
- **utils.py**: Provides utility functions used across the system

## Usage

### Basic Usage

```bash
python main.py --data_path ./data --output_path ./output --target_year 2025
```

### Advanced Options

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

### Command-line Arguments

- `--data_path`: Path to the data directory (default: './data')
- `--output_path`: Path for output files (default: './output')
- `--train_start_year`: Start year for training data (default: 2010)
- `--train_end_year`: End year for training data (default: 2024)
- `--target_year`: Target year for predictions (default: 2025)
- `--explore`: Enable data exploration (default: False)
- `--load_model`: Load pre-trained model instead of training new one (default: False)
- `--load_features`: Load pre-calculated features instead of recalculating (default: False)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--n_cores`: Number of CPU cores for parallel processing (default: auto-detect)
- `--clear_cache`: Clear computation cache (default: False)

## Data Requirements

The system expects the following CSV files in the data directory:

- **MTeams.csv**: Men's teams information
- **MRegularSeasonCompactResults.csv**: Men's regular season results
- **MNCAATourneyCompactResults.csv**: Men's tournament results
- **MRegularSeasonDetailedResults.csv**: Men's regular season detailed stats
- **MNCAATourneySeeds.csv**: Men's tournament seeds
- **SampleSubmissionStage1.csv**: Sample submission format

## Key Features

### Advanced Feature Engineering

- Team performance statistics calculation
- Seed information processing
- Historical matchup analysis
- Tournament progression probability estimation
- Favorite-longshot bias correction

### Performance Optimization

- Parallel processing for compute-intensive operations
- Memory caching to avoid redundant calculations
- Vectorized operations for improved efficiency
- Memory usage optimization for large datasets

### Robust Evaluation

- Multiple metrics (Brier score, log loss, accuracy)
- Calibration curve analysis
- Visual prediction distributions
- Risk-optimized submission strategy based on Brier score properties

## Prediction Pipeline

1. **Data Loading**: Load and preprocess historical basketball data
2. **Feature Engineering**: Create predictive features from raw data
3. **Model Training**: Train an XGBoost model with optimized hyperparameters
4. **Evaluation**: Evaluate model performance using multiple metrics
5. **Prediction Generation**: Create predictions for tournament matchups
6. **Risk Strategy Application**: Apply optimal risk strategy for Brier score
7. **Submission Creation**: Format predictions for competition submission

## Theoretical Insights

The system implements several theoretical insights to improve prediction accuracy:

- **Brier Score Optimization**: For predictions with approximately 33.3% win probability, a strategic risk adjustment is applied to optimize the expected Brier score.
- **Favorite-Longshot Bias Correction**: The system corrects for the systematic underestimation of strong teams (low seeds) and overestimation of weak teams (high seeds).
- **Time-Aware Validation**: Validation is performed using more recent seasons to better reflect the temporal nature of basketball predictions.

## Example Results

The system generates several output files:

- Trained model file (xgb_model.pkl)
- Feature cache (features.pkl)
- Prediction submission file (submission_YYYYMMDD_HHMMSS.csv)
- Model evaluation metrics (model_metrics_YYYYMMDD_HHMMSS.txt)
- Visualizations (if enabled)

## Advanced Usage

### Training a Custom Model

```python
from train_model import build_xgboost_model
from utils import save_model

# Train custom model
xgb_model, model_columns = build_xgboost_model(
    X_train, y_train, X_val, y_val, 
    random_seed=42,
    param_tuning=True,
    visualize=True
)

# Save model
save_model(xgb_model, 'custom_model.pkl', model_columns)
```

### Generating Predictions

```python
from submission import prepare_tournament_predictions, create_submission
from utils import load_model, load_features

# Load model and features
model, model_columns = load_model('model.pkl')
features_dict = load_features('features.pkl')

# Generate predictions
predictions = prepare_tournament_predictions(
    model, features_dict, sample_submission, model_columns, year=2025
)

# Create submission file
submission = create_submission(predictions, sample_submission, 'my_submission.csv')
```

## Performance Notes

- Feature engineering is the most time-consuming part of the pipeline; use the `--load_features` flag to reuse previously calculated features.
- Parallel processing significantly improves performance but increases memory usage.
- For extremely large datasets, adjust the code to use chunked processing or reduce the date range.

## References

- XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- Brier Score: [https://en.wikipedia.org/wiki/Brier_score](https://en.wikipedia.org/wiki/Brier_score)
- NCAA Tournament: [https://www.ncaa.com/march-madness](https://www.ncaa.com/march-madness)

## Author

Junming Zhao

## License

MIT License

---

This README provides a comprehensive overview of the NCAA Basketball Tournament Prediction System, including setup instructions, usage examples, and key technical details. For questions or contributions, please open an issue on the repository.