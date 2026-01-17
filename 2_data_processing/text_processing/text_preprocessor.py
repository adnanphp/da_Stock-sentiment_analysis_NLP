# text_preprocessor.py
import pandas as pd
import numpy as np
import re
import string
from typing import Dict, List, Any, Optional, Callable
import json
from datetime import datetime
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Main text preprocessing class with specialized pipelines for different text types.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.preprocessing_report = {}
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize specialized processors
        from reddit_cleaner import RedditTextCleaner
        from news_preprocessor import NewsArticlePreprocessor
        from sentiment_preparer import SentimentTextPreparer
        
        self.reddit_cleaner = RedditTextCleaner(config)
        self.news_preprocessor = NewsArticlePreprocessor(config)
        self.sentiment_preparer = SentimentTextPreparer(config)
    
    def detect_text_columns(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Detect columns that contain text data and analyze their characteristics.
        """
        text_columns = {}
        
        for column in df.columns:
            # Check if column contains text data
            if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                # Sample data for analysis
                sample_data = df[column].dropna().head(sample_size)
                
                if len(sample_data) > 0:
                    # Analyze text characteristics
                    text_stats = self._analyze_text_characteristics(sample_data, column)
                    
                    # Determine text type
                    text_type = self._classify_text_type(sample_data, column, text_stats)
                    
                    text_columns[column] = {
                        'text_type': text_type,
                        'statistics': text_stats,
                        'null_count': int(df[column].isnull().sum()),
                        'null_percentage': float((df[column].isnull().sum() / len(df)) * 100),
                        'sample_original': sample_data.head(3).tolist(),
                        'recommended_processor': self._get_recommended_processor(text_type)
                    }
        
        return text_columns
    
    def _analyze_text_characteristics(self, sample_data: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze text characteristics for a column."""
        text_samples = sample_data.astype(str).tolist()
        
        # Basic statistics
        word_counts = [len(str(text).split()) for text in text_samples]
        char_counts = [len(str(text)) for text in text_samples]
        
        # Text content analysis
        all_text = ' '.join(text_samples)
        words = all_text.lower().split()
        
        # Common patterns
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        mention_pattern = r'@\w+'
        hashtag_pattern = r'#\w+'
        html_pattern = r'<.*?>'
        
        stats = {
            'avg_word_count': float(np.mean(word_counts)),
            'avg_char_count': float(np.mean(char_counts)),
            'unique_words': len(set(words)),
            'total_words': len(words),
            'vocabulary_richness': len(set(words)) / len(words) if len(words) > 0 else 0,
            'contains_urls': any(re.search(url_pattern, text) for text in text_samples),
            'contains_mentions': any(re.search(mention_pattern, text) for text in text_samples),
            'contains_hashtags': any(re.search(hashtag_pattern, text) for text in text_samples),
            'contains_html': any(re.search(html_pattern, text) for text in text_samples),
            'avg_sentence_length': float(np.mean([len(re.findall(r'[.!?]+', text)) for text in text_samples])),
        }
        
        return stats
    
    def _classify_text_type(self, sample_data: pd.Series, column_name: str, stats: Dict[str, Any]) -> str:
        """Classify the type of text content."""
        column_lower = column_name.lower()
        text_samples = sample_data.astype(str).str.lower()
        
        # Column name based classification
        if any(pattern in column_lower for pattern in ['title', 'headline', 'subject']):
            return 'title'
        elif any(pattern in column_lower for pattern in ['content', 'body', 'text', 'article', 'description']):
            return 'content'
        elif any(pattern in column_lower for pattern in ['comment', 'post', 'reddit', 'social']):
            return 'social_media'
        elif any(pattern in column_lower for pattern in ['review', 'sentiment', 'opinion', 'feedback']):
            return 'sentiment'
        elif any(pattern in column_lower for pattern in ['summary', 'abstract', 'executive']):
            return 'summary'
        
        # Content based classification
        all_text = ' '.join(text_samples.tolist())
        
        # Social media patterns
        social_patterns = [r'u/\w+', r'r/\w+', r'@\w+', r'#\w+', r'^https?://']
        if any(re.search(pattern, all_text) for pattern in social_patterns):
            return 'social_media'
        
        # News patterns (longer text, formal language)
        if stats['avg_word_count'] > 50 and stats['avg_char_count'] > 200:
            return 'content'
        
        # Title patterns (shorter text)
        if stats['avg_word_count'] < 20 and stats['avg_char_count'] < 150:
            return 'title'
        
        return 'general'
    
    def _get_recommended_processor(self, text_type: str) -> str:
        """Get recommended processor based on text type."""
        processor_map = {
            'social_media': 'reddit_cleaner',
            'content': 'news_preprocessor', 
            'title': 'news_preprocessor',
            'summary': 'news_preprocessor',
            'sentiment': 'sentiment_preparer',
            'general': 'news_preprocessor'
        }
        return processor_map.get(text_type, 'news_preprocessor')
    
    def preprocess_dataset(self, df: pd.DataFrame, dataset_name: str,
                          text_columns: List[str] = None,
                          processor_type: str = 'auto') -> pd.DataFrame:
        """
        Preprocess text columns in a dataset.
        """
        processed_df = df.copy()
        preprocessing_details = {}
        
        print(f"📝 Preprocessing text columns for: {dataset_name}")
        
        # Auto-detect text columns if not provided
        if not text_columns:
            detected_columns = self.detect_text_columns(df)
            text_columns = list(detected_columns.keys())
            print(f"  🔍 Auto-detected text columns: {text_columns}")
        
        for column in text_columns:
            if column not in df.columns:
                print(f"  ⚠️  Column '{column}' not found in dataset")
                continue
            
            try:
                print(f"  📄 Processing: {column}")
                original_dtype = str(df[column].dtype)
                original_sample = df[column].head(3).tolist()
                
                # Determine processor to use
                if processor_type == 'auto':
                    detected_info = self.detect_text_columns(df[[column]])
                    if column in detected_info:
                        processor_type = detected_info[column]['recommended_processor']
                    else:
                        processor_type = 'news_preprocessor'
                
                # Apply appropriate processor
                if processor_type == 'reddit_cleaner':
                    processed_series = self.reddit_cleaner.clean_text_column(df[column])
                    processor_used = 'Reddit Cleaner'
                elif processor_type == 'sentiment_preparer':
                    processed_series = self.sentiment_preparer.prepare_sentiment_text(df[column])
                    processor_used = 'Sentiment Preparer'
                else:  # news_preprocessor or general
                    processed_series = self.news_preprocessor.preprocess_articles(df[column])
                    processor_used = 'News Preprocessor'
                
                # Update dataframe
                processed_df[column] = processed_series
                
                # Calculate processing statistics
                original_non_null = df[column].notna().sum()
                processed_non_null = processed_series.notna().sum()
                success_rate = (processed_non_null / original_non_null * 100) if original_non_null > 0 else 0
                
                # Record details
                preprocessing_details[column] = {
                    'original_dtype': original_dtype,
                    'processed_dtype': str(processed_series.dtype),
                    'processor_used': processor_used,
                    'success_rate': float(success_rate),
                    'original_sample': original_sample,
                    'processed_sample': processed_series.head(3).tolist(),
                    'null_count_before': int(df[column].isnull().sum()),
                    'null_count_after': int(processed_series.isnull().sum()),
                    'avg_length_before': float(df[column].astype(str).str.len().mean()),
                    'avg_length_after': float(processed_series.astype(str).str.len().mean())
                }
                
                print(f"  ✅ {column}: {processor_used} (success: {success_rate:.1f}%)")
                
            except Exception as e:
                print(f"  ❌ Error preprocessing {column}: {e}")
                preprocessing_details[column] = {
                    'error': str(e),
                    'original_dtype': str(df[column].dtype)
                }
        
        # Store report
        self.preprocessing_report[dataset_name] = {
            'timestamp': datetime.now().isoformat(),
            'processor_strategy': processor_type,
            'columns_processed': len(text_columns),
            'preprocessing_details': preprocessing_details,
            'original_shape': list(df.shape),
            'processed_shape': list(processed_df.shape)
        }
        
        return processed_df
    
    def batch_preprocess(self, data_dict: Dict[str, pd.DataFrame],
                        text_column_mapping: Dict[str, List[str]] = None,
                        processor_mapping: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
        """
        Preprocess multiple datasets in batch.
        """
        processed_data = {}
        
        for dataset_name, df in data_dict.items():
            text_columns = None
            processor_type = 'auto'
            
            if text_column_mapping and dataset_name in text_column_mapping:
                text_columns = text_column_mapping[dataset_name]
            
            if processor_mapping and dataset_name in processor_mapping:
                processor_type = processor_mapping[dataset_name]
            
            processed_df = self.preprocess_dataset(
                df, dataset_name, text_columns, processor_type
            )
            processed_data[dataset_name] = processed_df
        
        return processed_data
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of all preprocessing operations."""
        summary = {
            'total_datasets_processed': len(self.preprocessing_report),
            'total_columns_preprocessed': 0,
            'average_success_rate': 0.0,
            'processor_distribution': {},
            'text_type_distribution': {}
        }
        
        total_success_rate = 0.0
        total_columns = 0
        
        for dataset, report in self.preprocessing_report.items():
            processor = report.get('processor_strategy', 'unknown')
            summary['processor_distribution'][processor] = summary['processor_distribution'].get(processor, 0) + 1
            
            columns_processed = report.get('columns_processed', 0)
            summary['total_columns_preprocessed'] += columns_processed
            
            # Calculate average success rate for this dataset
            details = report.get('preprocessing_details', {})
            if details:
                dataset_success_rates = [
                    col_info.get('success_rate', 0.0) 
                    for col_info in details.values() 
                    if 'success_rate' in col_info
                ]
                if dataset_success_rates:
                    total_success_rate += sum(dataset_success_rates) / len(dataset_success_rates)
                    total_columns += len(dataset_success_rates)
        
        if total_columns > 0:
            summary['average_success_rate'] = round(total_success_rate / total_columns, 2)
        
        return summary
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types."""
        if isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)
    
    def save_preprocessing_report(self, output_path: str = "text_preprocessing_report.json"):
        """Save preprocessing report to JSON."""
        serializable_report = self._convert_to_serializable(self.preprocessing_report)
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"✓ Preprocessing report saved to: {output_path}")
