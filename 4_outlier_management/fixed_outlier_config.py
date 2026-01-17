# fixed_outlier_config.py
"""
Fixed configuration for outlier detection & treatment
"""

# Conservative detection settings
DETECTION_CONFIG = {
    "default_methods": ["iqr", "mad", "zscore"],
    "multivariate_methods": ["isolation_forest"],
    "confidence_level": 0.95,
    "auto_method_selection": True,
}

# Conservative method parameters
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
    }
}

# Conservative treatment settings
TREATMENT_CONFIG = {
    "default_strategy": "winsorize",
    "aggressive_threshold": 0.1,
    "conservative_threshold": 0.05,
    "preserve_shape": True,
    "validate_treatment": True,
}

# Conservative treatment parameters
TREATMENT_METHOD_PARAMS = {
    "cap": {
        "method": "iqr",
        "multiplier": 2.0,
        "use_quantiles": True,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95
    },
    "winsorize": {
        "limits": [0.05, 0.05],
        "inclusive": [True, True]
    },
    "transform": {
        "method": "log",
        "offset": 1.0,
        "standardize": False
    },
    "impute": {
        "method": "median",
        "use_detection": True,
        "strategy": "conservative"
    },
    "remove": {
        "max_removal_percentage": 0.05
    }
}
