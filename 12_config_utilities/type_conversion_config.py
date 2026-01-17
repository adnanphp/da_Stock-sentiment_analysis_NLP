# type_conversion_config.py
"""
Configuration file for data type conversion pipeline
Customize these settings for your specific use case
"""

# Confidence threshold for automatic conversion (0.0 to 1.0)
AUTOMATIC_CONVERSION_THRESHOLD = 0.7

# Manual type mappings for specific datasets
# Format: {"dataset_name": {"column_name": "target_type"}}
MANUAL_TYPE_MAPPINGS = {
    # Stock data examples
    "stock_data_all_stocks_combined.csv": {
        # "date": "datetime",
        # "volume": "integer",
        # "open": "float",
        # "high": "float", 
        # "low": "float",
        # "close": "float"
    },
    
    # Reddit data examples  
    "reddit_data_reddit_dividends_expanded.csv": {
        # "created_utc": "datetime",
        # "score": "integer",
        # "num_comments": "integer"
    },
    
    # News data examples
    "news_data_alpha_vantage_news_expanded.csv": {
        # "time_published": "datetime",
        # "sentiment_score": "float"
    }
}

# Column patterns for automatic type detection
COLUMN_PATTERNS = {
    "datetime_patterns": [
        r'.*date.*', r'.*time.*', r'.*timestamp.*', r'created.*', r'updated.*'
    ],
    "numeric_patterns": [
        r'.*id.*', r'.*count.*', r'.*number.*', r'.*age.*', r'.*score.*',
        r'.*price.*', r'.*amount.*', r'.*volume.*', r'.*rate.*', r'.*percent.*'
    ],
    "boolean_patterns": [
        r'.*is_.*', r'.*has_.*', r'.*flag.*', r'.*active.*', r'.*status.*'
    ]
}

# Type conversion preferences
CONVERSION_PREFERENCES = {
    "prefer_datetime_formats": ['%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d %H:%M:%S'],
    "prefer_integer_for_ids": True,
    "max_category_cardinality": 1000,  # Maximum unique values for category conversion
    "min_boolean_confidence": 0.9,     # Minimum confidence for boolean conversion
}

# Output settings
OUTPUT_SETTINGS = {
    "compression": None,  # 'gzip', 'snappy', etc.
    "index": False,
    "encoding": "utf-8"
}
