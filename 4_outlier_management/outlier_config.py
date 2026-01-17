# outlier_config.py
"""
Configuration file for Outlier Detection & Treatment Pipeline
"""

# Outlier detection settings
DETECTION_CONFIG = {
    "default_methods": ["iqr", "zscore", "mad"],
    "multivariate_methods": ["isolation_forest", "lof", "elliptic_envelope"],
    "confidence_level": 0.95,
    "auto_method_selection": True,
}

# Method-specific parameters
DETECTION_METHOD_PARAMS = {
    "iqr": {
        "multiplier": 1.5,
        "use_quantiles": False
    },
    "zscore": {
        "threshold": 3.0,
        "use_absolute": True
    },
    "mad": {
        "threshold": 3.0,
        "consistency_constant": 1.4826
    },
    "isolation_forest": {
        "contamination": "auto",
        "n_estimators": 100,
        "max_samples": "auto",
        "random_state": 42
    },
    "lof": {
        "n_neighbors": 20,
        "contamination": "auto",
        "metric": "euclidean"
    },
    "elliptic_envelope": {
        "contamination": 0.1,
        "random_state": 42
    },
    "dbscan": {
        "eps": 0.5,
        "min_samples": 5,
        "metric": "euclidean"
    }
}

# Outlier treatment settings
TREATMENT_CONFIG = {
    "default_strategy": "cap",
    "aggressive_threshold": 0.1,  # 10% outliers
    "conservative_threshold": 0.05,  # 5% outliers
    "preserve_shape": True,
    "validate_treatment": True,
}

# Treatment method parameters
TREATMENT_METHOD_PARAMS = {
    "cap": {
        "method": "iqr",
        "multiplier": 2.0,  # More conservative than 1.5
        "use_quantiles": True,  # Use quantile-based bounds
        "lower_quantile": 0.05,  # 5th percentile
        "upper_quantile": 0.95   # 95th percentile
    },
    "winsorize": {
        "limits": [0.05, 0.05],  # 5% on each tail
        "inclusive": [True, True]
    },
    "transform": {
        "method": "log",
        "offset": 1e-6,
        "standardize": True
    },
    "impute": {
        "method": "median",
        "use_detection": True,
        "strategy": "conservative"
    },
    "remove": {
        "method": "iqr",
        "multiplier": 3.0,
        "handle_na": True
    }
}

# Column-specific treatment rules
COLUMN_TREATMENT_RULES = {
    "price_columns": {
        "detection_methods": ["iqr", "isolation_forest"],
        "treatment_method": "cap",
        "treatment_params": {"method": "iqr", "multiplier": 2.0}
    },
    "volume_columns": {
        "detection_methods": ["iqr", "mad"],
        "treatment_method": "transform", 
        "treatment_params": {"method": "log"}
    },
    "indicator_columns": {
        "detection_methods": ["zscore", "iqr"],
        "treatment_method": "winsorize",
        "treatment_params": {"limits": [0.05, 0.05]}
    },
    "sentiment_columns": {
        "detection_methods": ["iqr"],
        "treatment_method": "cap",
        "treatment_params": {"method": "iqr", "multiplier": 1.5}
    }
}

# Performance and validation settings
PERFORMANCE_CONFIG = {
    "batch_processing": True,
    "batch_size": 1000,
    "parallel_processing": False,  # Some ML methods are not thread-safe
    "memory_efficient": True,
    "validation_samples": 1000,
}

# Output and reporting settings
OUTPUT_CONFIG = {
    "include_detection_flags": True,
    "include_treatment_history": True,
    "generate_visualizations": False,
    "report_format": "detailed",  # 'summary', 'detailed', 'minimal'
    "save_treatment_parameters": True,
}

# Dataset-specific configurations
DATASET_OUTLIER_CONFIGS = {
    "stock_data": {
        "sensitive_columns": ["Close", "Open", "High", "Low"],
        "robust_columns": ["Volume", "RSI", "MACD"],
        "exclude_columns": ["Date", "ticker", "company_name"],
        "treatment_approach": "conservative"
    },
    "reddit_data": {
        "sensitive_columns": ["score", "num_comments"],
        "robust_columns": ["sentiment_scores", "text_length"],
        "exclude_columns": ["created_utc", "author", "subreddit"],
        "treatment_approach": "moderate"
    },
    "news_data": {
        "sensitive_columns": ["sentiment_score", "relevance_score"],
        "robust_columns": ["word_count", "readability_score"],
        "exclude_columns": ["time_published", "source", "title"],
        "treatment_approach": "conservative"
    }
}
