# feature_config.py
"""
Configuration file for Feature Engineering Pipeline
"""

# Technical indicator settings
TECHNICAL_CONFIG = {
    "sma_periods": [5, 10, 20, 50, 100, 200],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "stochastic_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}

# Text feature settings
TEXT_FEATURE_CONFIG = {
    "min_text_length": 2,
    "max_text_length": 10000,
    "embedding_method": "tfidf",  # 'tfidf', 'count'
    "max_embedding_features": 50,
    "sentiment_lexicon": "basic",  # 'basic', 'vader', 'custom'
    "linguistic_features": True,
    "readability_metrics": True,
}

# Time feature settings
TIME_FEATURE_CONFIG = {
    "lag_periods": [1, 2, 3, 5, 7, 14, 21, 30],
    "rolling_windows": [3, 5, 7, 10, 14, 20, 30, 50],
    "cyclical_encoding": True,
    "seasonal_features": True,
    "holiday_features": True,
    "business_day_features": True,
}

# Basic transformation settings
TRANSFORMATION_CONFIG = {
    "log_transform": True,
    "standardize": True,
    "interaction_terms": True,
    "polynomial_degree": 2,
    "handle_skewness": True,
}

# Feature selection settings
SELECTION_CONFIG = {
    "max_features": 1000,
    "correlation_threshold": 0.95,
    "variance_threshold": 0.01,
    "feature_importance_threshold": 0.001,
}

# Output settings
OUTPUT_CONFIG = {
    "include_original_features": True,
    "feature_naming_convention": "descriptive",  # 'descriptive', 'compact'
    "compression": None,
    "format": "parquet",  # 'csv', 'parquet'
}

# Dataset-specific feature engineering
DATASET_FEATURE_CONFIGS = {
    "stock_data": {
        "primary_features": ["technical_indicators", "time_features"],
        "secondary_features": ["basic_transformations"],
        "target_column": "Close",  # For supervised learning
        "group_by": ["ticker"],   # For panel data features
    },
    "reddit_data": {
        "primary_features": ["text_features", "time_features"],
        "secondary_features": ["sentiment_features"],
        "target_column": None,
        "group_by": ["subreddit"],
    },
    "news_data": {
        "primary_features": ["text_features", "time_features"],
        "secondary_features": ["sentiment_features", "readability_features"],
        "target_column": None,
        "group_by": ["source"],
    }
}

# Performance optimization
PERFORMANCE_CONFIG = {
    "parallel_processing": True,
    "batch_size": 1000,
    "memory_efficient": True,
    "chunk_size": 10000,
}
