
03 — Feature Engineering
Transforms cleaned data into predictive features. The largest single
stage by output size — the final feature space has ~2,700 columns
across stocks, news sentiment, and Reddit sentiment.

Files
File	Purpose
feature_config.py	Configuration for which features to generate
feature_engineer.py	Main feature engineering pipeline
feature_pipeline.py	Orchestrates the pipeline end-to-end
run_feature_engineering.py	Entry point
sentiment_preparer.py	Prepares sentiment scores from news and Reddit
technical_indicators.py	Generates 50+ technical indicators
Technical indicators
Trend, momentum, volatility, volume, and market-structure indicators:

SMA, EMA, MACD, RSI

Bollinger Bands, ATR

OBV, volume ratios

Support/resistance features

Sentiment features
News and Reddit text produce sentiment features that are joined to the
stock price series by (ticker, date).

Legacy
legacy/ contains earlier versions of feature_engineer.py and
technical_indicators.py.

Inputs
Cleaned data from 02_data_processing/.

Output
Feature-engineered CSVs, per source (stock, news, Reddit).

Stage boundary
Input: cleaned data.
Output: feature matrices ready for outlier handling and scaling.
