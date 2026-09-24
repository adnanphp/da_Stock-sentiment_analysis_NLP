# 📈 Multi-Source Financial Sentiment Analysis for Stock Movement Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)](https://lightgbm.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **An end-to-end financial machine learning pipeline combining stock prices, technical indicators, news sentiment, Reddit sentiment, and market indicators to study stock movement and market regimes.**

---

## 🚀 Overview

This project explores **financial market prediction through multi-source data fusion**.

The pipeline combines:

* 📊 Stock market data — OHLC, volume, and technical indicators
* 📰 Financial news — Alpha Vantage and NewsAPI
* 💬 Social sentiment — Reddit posts and comments
* 📈 Market indicators — S&P 500, VIX, and volatility measures
* 🤖 Machine learning — XGBoost, LightGBM, Random Forest, and ensembles
* 🧠 Deep learning — MLP, CNN, and LSTM
* 📉 Dimensionality reduction — PCA and t-SNE
* 📊 Backtesting — strategy simulation and financial performance analysis
* 🌐 Production components — model serialization, prediction API, and monitoring

The project is designed as a complete workflow from **data collection → preprocessing → feature engineering → modeling → evaluation → backtesting → deployment**.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                            │
├──────────────┬───────────────┬──────────────┬──────────────┤
│ Stock Prices │ Financial News│    Reddit    │ Market Data  │
│   yFinance   │ Alpha Vantage │    PRAW      │ S&P 500/VIX  │
└──────┬───────┴───────┬───────┴──────┬───────┴──────┬───────┘
       │               │              │              │
       └───────────────┴──────────────┼──────────────┘
                                      ▼
                         ┌─────────────────────┐
                         │ Data Processing     │
                         │ Cleaning / Quality   │
                         │ Missing Values       │
                         │ Standardization      │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │ Technical Indicators│
                         │ Sentiment Features  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Dimensionality      │
                         │ Reduction           │
                         │ PCA / t-SNE         │
                         └──────────┬──────────┘
                                    ▼
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
        ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
        │ Traditional │      │ Deep        │      │ Ensemble    │
        │ ML          │      │ Learning    │      │ Models      │
        │ XGBoost     │      │ MLP / CNN   │      │ Voting /    │
        │ LightGBM    │      │ LSTM        │      │ Stacking    │
        └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
               └────────────────────┼────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Evaluation &        │
                         │ Backtesting         │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Prediction / API /  │
                         │ Visualization       │
                         └─────────────────────┘
```

---

## 📁 Project Structure

```text
financial-prediction-system/
│
├── 01_data_collection/
│   ├── data_collector.py
│   ├── reddit_collector.py
│   ├── news_collector.py
│   ├── stock_data_collector.py
│   └── market_indicator_collector.py
│
├── 02_data_processing/
│   ├── datetime_standardizer.py
│   ├── data_type_converter.py
│   ├── missing_value_imputer.py
│   ├── data_quality_checker_fixed.py
│   └── text_preprocessor.py
│
├── 03_feature_engineering/
│   ├── technical_indicators.py
│   ├── sentiment_analyzer.py
│   ├── feature_selector.py
│   ├── fixed_outlier_treatment.py
│   └── normalization_pipeline.py
│
├── 04_ml_modeling/
│   ├── pca_enhanced_modeling_pipeline_fixed_v1.py
│   ├── hyperparameter_tuning.py
│   └── ...
│
├── 05_deep_learning/
│   ├── deep_learning_pipeline.py
│   ├── lstm_model.py
│   ├── cnn_financial.py
│   └── ...
│
├── 06_dimensionality_reduction/
│   └── pca_tsne_analysis.py
│
├── 07_ensemble_methods/
│   └── ensemble_modeling.py
│
├── 08_backtesting/
│   ├── backtesting_framework.py
│   ├── trading_simulation.py
│   ├── performance_metrics.py
│   └── risk_analysis.py
│
├── 09_production/
│   ├── model_serializer.py
│   ├── prediction_service.py
│   ├── monitoring_dashboard.py
│   └── retraining_pipeline.py
│
├── 10_visualization/
│   └── ...
│
├── requirements.txt
├── main.py
└── README.md
```

---

# 🔄 Pipeline

## 1. Data Collection

The system integrates multiple financial and alternative data sources.

### Market Data

* OHLC prices
* Trading volume
* Historical price movements
* S&P 500
* VIX
* Volatility indicators

### News Data

* Alpha Vantage
* NewsAPI
* Financial news articles
* NLP-derived sentiment scores

### Social Data

* Reddit posts
* Reddit comments
* Financial subreddits
* Sentiment scores

Reported dataset coverage includes:

* **30+ stock tickers**
* **10+ financial Reddit communities**
* Multiple news sources
* Market-wide indicators

---

## 2. Data Processing

The processing pipeline includes:

* Datetime standardization
* Timezone normalization
* Data type conversion
* Missing-value imputation
* Outlier detection
* Data quality checks
* Text preprocessing
* Feature normalization

Multiple outlier-handling techniques are supported, including:

```text
IQR
 │
 ├── Outlier detection
 │
Z-Score
 │
 ├── Statistical filtering
 │
Winsorization
 │
 └── Robust value treatment
```

---

## 3. Feature Engineering

The project generates **50+ technical indicators** covering multiple aspects of market behavior.

### Trend Indicators

* SMA
* EMA
* Moving-average relationships

### Momentum

* RSI
* MACD
* Stochastic indicators

### Volatility

* Bollinger Bands
* ATR
* Volatility measures

### Volume

* OBV
* Volume ratios

### Market Structure

* Support/resistance levels
* Price movement features

### Sentiment

News and Reddit text are processed to generate sentiment-related features that can be combined with market variables.

---

# 📉 Dimensionality Reduction

The original feature space contains:

| Metric                   |    Result |
| ------------------------ | --------: |
| Original features        | **1,795** |
| PCA components           |    **35** |
| Variance retained        | **95.1%** |
| Dimensionality reduction | **98.0%** |

PCA was used to compress the feature space while retaining most of the measured variance.

t-SNE was additionally used for exploratory visualization of feature relationships and cluster structure.

---

# 🤖 Machine Learning

The project evaluates multiple machine learning approaches.

### Traditional ML

* XGBoost
* LightGBM
* Random Forest
* Linear models
* Voting ensembles
* Stacking ensembles

### Deep Learning

* MLP
* CNN
* LSTM

### Ensemble Learning

```text
                 ┌──────────────┐
                 │ XGBoost      │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │ LightGBM     │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │ Random Forest│
                 └──────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Voting /      │
                │ Stacking      │
                └───────┬───────┘
                        ▼
                    Prediction
```

---

# 🎯 Prediction Targets

The pipeline investigates several prediction tasks:

| Target                       | Reported Accuracy |
| ---------------------------- | ----------------: |
| 3-day price direction        |         **63.7%** |
| 5-day trend                  |         **62.6%** |
| 10-day trend                 |         **67.6%** |
| Sharpe-ratio signal          |         **62.6%** |
| Market regime classification |         **70.3%** |

These results are experimental model-evaluation results and should not be interpreted as guaranteed future market performance.

---

# 📊 Performance Results

## Traditional Machine Learning

| Target        | Best Model        | Accuracy | F1-Score | AUC-ROC |
| ------------- | ----------------- | -------: | -------: | ------: |
| 3-day Trend   | XGBoost           |    63.7% |    0.637 |   0.688 |
| 5-day Trend   | LightGBM          |    62.6% |    0.621 |   0.725 |
| 10-day Trend  | Stacking Ensemble |    67.6% |    0.664 |   0.717 |
| Market Regime | Voting Ensemble   |    70.3% |    0.681 |   0.841 |

## Deep Learning

| Model | Accuracy | F1-Score | AUC-ROC |
| ----- | -------: | -------: | ------: |
| MLP   |    92.0% |    0.908 |   0.966 |
| CNN   |    66.4% |    0.410 |   0.839 |
| LSTM  |    68.4% |        — |   0.763 |

> **Evaluation note:** The reported MLP result is substantially higher than the other models. For financial time-series applications, unusually strong results should be carefully checked for temporal leakage, overlapping samples, feature leakage, and train/test splitting methodology before being treated as evidence of real-world predictive performance.

---

# 📈 Backtesting & Trading Simulation

The project includes a backtesting framework for evaluating prediction-driven strategies.

### Evaluation Metrics

* Sharpe ratio
* Maximum drawdown
* Win rate
* Cumulative performance
* Strategy returns
* Comparison with Buy & Hold

### Conceptual Workflow

```text
Model Prediction
       │
       ▼
Trading Signal
       │
       ▼
Position Generation
       │
       ▼
Historical Backtest
       │
       ├── Returns
       ├── Sharpe Ratio
       ├── Maximum Drawdown
       └── Win Rate
       │
       ▼
Strategy vs Buy & Hold
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/adnanphp/da_Stock-sentiment_analysis_NLP.git
cd da_Stock-sentiment_analysis_NLP
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 API Configuration

The data collection components may require API credentials.

Create a local configuration file and keep credentials outside version control.

Example:

```python
REDDIT_CONFIG = {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "user_agent": "your_user_agent"
}

NEWS_API_KEY = "your_newsapi_key"

ALPHA_VANTAGE_API_KEY = "your_alphavantage_key"
```

**Do not commit API keys or other credentials to GitHub.**

---

# ⚡ Quick Start

## Full Pipeline

```bash
python main.py
```

## Data Collection

```bash
python 01_data_collection/data_collector.py
```

## Data Processing

```bash
python 02_data_processing/datetime_standardizer.py
python 02_data_processing/data_type_converter.py
python 02_data_processing/missing_value_imputer.py
python 02_data_processing/data_quality_checker_fixed.py
```

## Feature Engineering

```bash
python 03_feature_engineering/technical_indicators.py
python 03_feature_engineering/fixed_outlier_treatment.py
python 03_feature_engineering/normalization_pipeline.py
```

## PCA / t-SNE

```bash
python 06_dimensionality_reduction/pca_tsne_analysis.py
```

## Traditional ML

```bash
python 04_ml_modeling/pca_enhanced_modeling_pipeline_fixed_v1.py
```

## Deep Learning

```bash
python 05_deep_learning/deep_learning_pipeline.py
```

## Ensemble Models

```bash
python 07_ensemble_methods/ensemble_modeling.py
```

## Backtesting

```bash
python 08_backtesting/backtesting_framework.py
python 08_backtesting/trading_simulation.py
```

---

# 📊 Visualization Gallery

## PCA Analysis

![PCA Variance Analysis](./20_images/1_pca_variance_analysis.png)

*PCA explained-variance analysis.*

## t-SNE Analysis

![t-SNE Analysis](./20_images/3_tsne_comprehensive_visualizations.png)

*t-SNE visualization of the feature space.*

## Classification Results

![Classification Results](./20_images/5_classification_price_direction_results.png)

*Model performance for price-direction classification.*

## Deep Learning Evaluation

![Deep Learning Confusion Matrix](./20_images/6_dl_confusion_matrix.png)

![Deep Learning Model Comparison](./20_images/7_dl_model_comparison.png)

![Deep Learning ROC Curve](./20_images/9_dl_roc_curve.png)

![Deep Learning Training History](./20_images/10_dl_training_history.png)

*Deep learning model evaluation and training diagnostics.*

---

# 🧪 Advanced Usage

## Custom Model Training

```python
from modeling.pipeline import FinancialPredictionPipeline

pipeline = FinancialPredictionPipeline(
    data_path="your_data.csv",
    target_variable="price_direction",
    model_type="xgboost",
    time_horizon=5
)

pipeline.train(
    test_size=0.2,
    cross_validation=True,
    hyperparameter_tuning=True,
    ensemble_method="voting"
)

results = pipeline.evaluate()

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"F1-Score: {results['f1_score']:.3f}")
```

---

# ⚙️ Hyperparameter Optimization

### XGBoost Grid Search

```bash
python hyperparameter_tuning.py \
    --model xgboost \
    --param-grid config/xgboost_params.json \
    --cv-folds 5 \
    --n-jobs 4
```

### Random Forest Random Search

```bash
python hyperparameter_tuning.py \
    --model random_forest \
    --search-method random \
    --n-iter 100 \
    --cv-folds 3
```

---

# 🔎 Feature Selection

### Correlation-Based Selection

```bash
python feature_selector.py \
    --method correlation \
    --threshold 0.8
```

### Tree-Based Feature Importance

```bash
python feature_selector.py \
    --method tree_importance \
    --top-k 50
```

### Recursive Feature Elimination

```bash
python feature_selector.py \
    --method rfe \
    --n-features 30
```

---

# 🌐 Production Components

The project includes components for turning trained models into reusable prediction services.

### Model Serialization

```text
Trained Model
     │
     ▼
   Joblib
     │
     ▼
Saved Model Artifact
```

### Prediction API

A FastAPI-based service can expose trained models through REST endpoints.

Example request:

```python
import requests

payload = {
    "features": {
        "Close": 150.25,
        "Volume": 2500000,
        "RSI": 65.3,
        "MACD": 1.24,
        "sentiment_score": 0.78
    }
}

response = requests.post(
    "http://localhost:8000/predict",
    json=payload
)

prediction = response.json()

print(prediction)
```

---

# 🛠️ Technology Stack

## Programming & Data Science

| Technology   | Purpose             |
| ------------ | ------------------- |
| Python 3.8+  | Core development    |
| Pandas       | Data manipulation   |
| NumPy        | Numerical computing |
| scikit-learn | Machine learning    |
| XGBoost      | Gradient boosting   |
| LightGBM     | Gradient boosting   |
| NLTK         | NLP processing      |

## Deep Learning

| Technology | Purpose                             |
| ---------- | ----------------------------------- |
| TensorFlow | Deep learning                       |
| Keras      | Neural networks                     |
| PyTorch    | Alternative deep-learning framework |

## Data Collection

| Technology    | Purpose             |
| ------------- | ------------------- |
| yFinance      | Stock market data   |
| PRAW          | Reddit API          |
| Alpha Vantage | Financial data/news |
| NewsAPI       | News aggregation    |

## Visualization & Deployment

| Technology | Purpose                   |
| ---------- | ------------------------- |
| Matplotlib | Static visualization      |
| Plotly     | Interactive visualization |
| Seaborn    | Statistical visualization |
| Streamlit  | Dashboard                 |
| FastAPI    | REST API                  |
| Docker     | Containerization          |
| Joblib     | Model serialization       |
| MLflow     | Experiment tracking       |

---

# 🧩 Key Technical Achievements

### Multi-Source Data Fusion

Integrated market, news, social-media, and macro/market-indicator data into a unified feature pipeline.

### Feature Engineering

Developed a feature space containing **1,795 features**, including technical indicators and sentiment-derived variables.

### Dimensionality Reduction

Reduced the feature space from **1,795 to 35 PCA components** while retaining **95.1% of variance**.

### Model Comparison

Evaluated traditional ML, deep learning, and ensemble approaches across multiple prediction targets.

### Financial Evaluation

Implemented backtesting with financial metrics including:

* Sharpe ratio
* Maximum drawdown
* Win rate
* Strategy returns
* Buy & Hold comparison

### Production Architecture

Included model serialization, prediction-service components, monitoring, and deployment-oriented architecture.

---

# ⚠️ Important Financial Disclaimer

This project is intended for **research, experimentation, and educational purposes**.

Machine learning predictions of financial markets are uncertain. Historical model performance does not guarantee future results. The reported accuracy and other evaluation metrics should not be interpreted as investment advice or as evidence that a strategy will generate profits in live trading.

Before using any model for real-world financial decisions, evaluation should account for factors such as:

* Temporal leakage
* Transaction costs
* Slippage
* Market liquidity
* Regime changes
* Look-ahead bias
* Out-of-sample performance
* Walk-forward validation

---

# 🔮 Future Improvements

* [ ] Walk-forward validation
* [ ] More rigorous temporal cross-validation
* [ ] Transaction-cost modeling
* [ ] Slippage simulation
* [ ] More robust leakage detection
* [ ] Explainable AI with SHAP
* [ ] Automated model retraining
* [ ] Real-time data ingestion
* [ ] Streaming sentiment analysis
* [ ] Model monitoring
* [ ] Dockerized prediction API
* [ ] Interactive Streamlit dashboard
* [ ] MLflow experiment tracking
* [ ] Portfolio-level optimization

---

# 📚 References & Resources

### Financial Data

* [Yahoo Finance](https://finance.yahoo.com/)
* [Alpha Vantage](https://www.alphavantage.co/)
* [Reddit](https://www.reddit.com/)

### Machine Learning

* [scikit-learn](https://scikit-learn.org/)
* [XGBoost](https://xgboost.readthedocs.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [LightGBM](https://lightgbm.readthedocs.io/)

### Visualization

* [Plotly](https://plotly.com/python/)
* [Matplotlib](https://matplotlib.org/)

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature/amazing-feature
```

3. Commit your changes.

```bash
git commit -m "Add amazing feature"
```

4. Push the branch.

```bash
git push origin feature/amazing-feature
```

5. Open a Pull Request.

### Contribution Guidelines

* Follow PEP 8.
* Add tests where appropriate.
* Update documentation.
* Use descriptive commit messages.
* Do not commit API keys or credentials.

---

# 📄 License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

# 👨‍💻 Author

**Adnan**

* GitHub: [@adnanphp](https://github.com/adnanphp)

---

## ⭐ Portfolio Project

This project demonstrates an end-to-end **data science and machine learning workflow** covering:

**Multi-source data → NLP sentiment → feature engineering → dimensionality reduction → ML/DL → ensemble learning → backtesting → production-oriented prediction.**

If you find the project useful, consider giving the repository a ⭐.
