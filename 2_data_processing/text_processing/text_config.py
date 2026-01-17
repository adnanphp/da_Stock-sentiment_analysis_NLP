# text_config.py
"""
Configuration file for Text Preprocessing Pipeline
"""

# General text preprocessing settings
GENERAL_CONFIG = {
    "min_text_length": 2,           # Minimum characters to keep text
    "max_text_length": 10000,       # Maximum characters (truncate longer texts)
    "handle_null_values": "keep",   # 'keep', 'drop', 'fill'
    "null_fill_value": "",          # Value to fill nulls with
    "encoding": "utf-8",            # Text encoding
}

# Reddit cleaner settings
REDDIT_CONFIG = {
    "emoji_strategy": "remove",     # 'remove', 'describe', 'keep'
    "remove_urls": True,
    "remove_user_mentions": True,
    "remove_subreddit_mentions": True,
    "remove_markdown": True,
    "expand_contractions": True,
    "normalize_case": True,
}

# News preprocessor settings  
NEWS_CONFIG = {
    "case_strategy": "lower",       # 'lower', 'sentence', 'preserve'
    "remove_html": True,
    "remove_bylines": True,
    "remove_datelines": True,
    "enable_advanced_nlp": True,
    "nlp_level": "medium",          # 'light', 'medium', 'heavy'
    "preserve_punctuation": True,   # Keep punctuation for sentence structure
}

# Sentiment preparer settings
SENTIMENT_CONFIG = {
    "handle_negations": True,
    "preserve_sentiment_words": True,
    "remove_non_sentiment_stopwords": True,
    "expand_contractions": True,
    "handle_intensifiers": True,
    "calculate_sentiment_scores": True,
}

# NLP component settings
NLP_CONFIG = {
    "language": "english",
    "stopwords_language": "english",
    "lemmatize": True,
    "stemming": False,              # Use lemmatization instead of stemming
    "tokenize_sentences": True,
    "remove_numbers": False,        # Keep numbers as they might be important
}

# Output settings
OUTPUT_CONFIG = {
    "format": "clean_text",         # 'clean_text', 'tokens', 'features'
    "include_original": False,      # Include original text in output
    "include_features": True,       # Include extracted features
    "compression": None,            # Output compression
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    "reddit_data": {
        "expected_text_columns": ["title", "selftext", "body", "content"],
        "processor": "reddit_cleaner",
        "features_to_extract": ["has_url", "has_user_mention", "has_emoji", "word_count"]
    },
    "news_data": {
        "expected_text_columns": ["title", "summary", "content", "description"],
        "processor": "news_preprocessor", 
        "features_to_extract": ["sentence_count", "word_count", "readability_score"]
    },
    "stock_data": {
        "expected_text_columns": ["description", "company_name", "sector", "industry"],
        "processor": "news_preprocessor",
        "features_to_extract": ["word_count", "char_count", "unique_word_ratio"]
    }
}

# Feature extraction settings
FEATURE_CONFIG = {
    "extract_basic_stats": True,
    "extract_linguistic_features": True,
    "extract_sentiment_features": True,
    "extract_domain_features": True,
    "max_features": 50,             # Maximum number of features to extract
}
