# datetime_config.py
"""
Configuration file for DateTime Standardization Pipeline
"""

# Multi-format parser settings
PARSER_CONFIG = {
    "preferred_formats": [
        '%Y-%m-%dT%H:%M:%S.%f%z',  # ISO with microseconds and timezone
        '%Y-%m-%dT%H:%M:%S%z',     # ISO with timezone
        '%Y-%m-%d %H:%M:%S',        # Standard datetime
        '%Y-%m-%d',                 # Date only
        '%m/%d/%Y %H:%M:%S',        # US format with time
        '%m/%d/%Y',                 # US date format
    ],
    "fallback_strategy": "aggressive",  # 'conservative', 'aggressive', 'strict'
    "handle_ambiguous_dates": True,     # Handle dates like 02/03/2023 (MM/DD vs DD/MM)
    "prefer_us_format": True,           # Prefer MM/DD/YYYY over DD/MM/YYYY for ambiguous dates
}

# Timezone normalization settings
TIMEZONE_CONFIG = {
    "default_timezone": "UTC",
    "common_timezones": {
        "US/Eastern": ["EST", "EDT", "Eastern"],
        "US/Central": ["CST", "CDT", "Central"], 
        "US/Mountain": ["MST", "MDT", "Mountain"],
        "US/Pacific": ["PST", "PDT", "Pacific"],
        "UTC": ["UTC", "GMT", "Z"],
        "Europe/London": ["BST", "GMT"],
        "Europe/Paris": ["CET", "CEST"],
    },
    "auto_detect_timezone": True,
    "handle_dst_transitions": True,  # Handle Daylight Saving Time transitions
}

# Output settings
OUTPUT_CONFIG = {
    "preferred_format": "iso",  # 'iso', 'standard', 'unix', 'original'
    "include_timezone": True,
    "handle_nulls": "keep",     # 'keep', 'drop', 'fill'
    "null_fill_value": None,
}

# Pipeline settings
PIPELINE_CONFIG = {
    "auto_detect_columns": True,
    "confidence_threshold": 0.6,  # Minimum confidence for auto-detection
    "batch_size": 1000,           # Process datasets in batches
    "enable_logging": True,
    "log_level": "INFO",          # DEBUG, INFO, WARNING, ERROR
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    "stock_data": {
        "expected_date_columns": ["Date", "Timestamp", "datetime"],
        "expected_timezone": "UTC",
        "date_range": {
            "min_year": 2000,
            "max_year": 2025
        }
    },
    "reddit_data": {
        "expected_date_columns": ["created_utc", "timestamp", "date"],
        "expected_timezone": "UTC",  # Reddit timestamps are typically UTC
        "is_unix_timestamp": True    # Reddit uses Unix timestamps
    },
    "news_data": {
        "expected_date_columns": ["time_published", "published_at", "date"],
        "expected_timezone": "UTC",
        "preferred_format": "iso"
    }
}
