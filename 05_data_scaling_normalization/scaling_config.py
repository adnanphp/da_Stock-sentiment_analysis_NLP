# scaling_config.py
"""
Configuration file for Data Normalization Pipeline
"""

# Scaler selection settings
SCALER_SELECTION_CONFIG = {
    "default_scaler": "standard",  # 'standard', 'minmax', 'robust', 'power'
    "auto_scaler_selection": True,
    "fallback_scaler": "minmax",
    "validation_strictness": "medium",  # 'low', 'medium', 'high'
}

# Scaler parameter configurations
SCALER_CONFIGS = {
    "standard": {
        "with_mean": True,
        "with_std": True,
        "copy": True
    },
    "minmax": {
        "feature_range": (0, 1),
        "copy": True
    },
    "robust": {
        "with_centering": True,
        "with_scaling": True,
        "quantile_range": (25.0, 75.0),
        "copy": True
    },
    "maxabs": {
        "copy": True
    },
    "power": {
        "method": "yeo-johnson",  # 'yeo-johnson', 'box-cox'
        "standardize": True,
        "copy": True
    },
    "quantile": {
        "n_quantiles": 1000,
        "output_distribution": "uniform",  # 'uniform', 'normal'
        "copy": True
    }
}

# Column exclusion patterns
EXCLUSION_PATTERNS = [
    "id", "index", "count", "flag", "binary", "dummy", "encoded",
    "year", "month", "day", "week", "quarter", "season",
    "is_", "has_", "contains_", "ratio", "percentage"
]

# Scaling analysis thresholds
ANALYSIS_THRESHOLDS = {
    "skewness_threshold": 2.0,
    "outlier_threshold": 0.05,  # 5% of data
    "sparse_threshold": 0.8,    # 80% zeros
    "constant_threshold": 1,    # unique values
    "normal_range_min": 0,
    "normal_range_max": 1,
    "standardized_mean_threshold": 0.1,
    "standardized_std_threshold": 0.1
}

# Inversion settings
INVERSION_CONFIG = {
    "inversion_tolerance": 1e-6,
    "store_original_params": True,
    "validate_inversion": True,
    "inversion_fallback": "approximate"  # 'approximate', 'original', 'error'
}

# Output settings
OUTPUT_CONFIG = {
    "naming_convention": "append",  # 'append', 'replace', 'both'
    "append_suffix": "_scaled",
    "include_original": True,
    "compression": None,
    "format": "parquet",  # 'csv', 'parquet'
}

# Dataset-specific scaling configurations
DATASET_SCALING_CONFIGS = {
    "stock_data": {
        "price_columns": ["Close", "Open", "High", "Low"],
        "volume_columns": ["Volume"],
        "indicator_columns": ["RSI", "MACD", "Stoch_K", "Stoch_D"],
        "recommended_scalers": {
            "prices": "standard",
            "volumes": "robust", 
            "indicators": "minmax"
        }
    },
    "reddit_data": {
        "text_feature_columns": ["char_count", "word_count", "sentiment"],
        "time_columns": ["year", "month", "day"],
        "recommended_scalers": {
            "text_features": "minmax",
            "time_features": "standard"
        }
    },
    "news_data": {
        "text_feature_columns": ["char_count", "word_count", "readability"],
        "sentiment_columns": ["sentiment_balance", "positive_words"],
        "recommended_scalers": {
            "text_features": "minmax", 
            "sentiment": "standard"
        }
    }
}

# Performance and memory settings
PERFORMANCE_CONFIG = {
    "batch_processing": True,
    "batch_size": 1000,
    "memory_efficient": True,
    "parallel_processing": False,  # Scikit-learn scalers are not thread-safe
    "chunk_size": 10000,
}
