# text_feature_extractor.py
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TextFeatureExtractor:
    """
    Extract features from text data for machine learning.
    Supports basic statistics, linguistic features, and readability metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def extract_text_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract features from text columns.
        """
        feature_df = df.copy()
        details = {'features_created': 0, 'text_columns_processed': []}
        
        text_columns = config.get('text_columns', [])
        feature_types = config.get('feature_types', ['basic_stats'])
        
        for text_column in text_columns:
            if text_column not in df.columns:
                continue
            
            print(f"    📝 Extracting features from: {text_column}")
            details['text_columns_processed'].append(text_column)
            
            # Extract features based on types
            for feature_type in feature_types:
                try:
                    if feature_type == 'basic_stats':
                        feature_df, count = self._extract_basic_stats(feature_df, text_column)
                        details['features_created'] += count
                    
                    elif feature_type == 'linguistic':
                        feature_df, count = self._extract_linguistic_features(feature_df, text_column)
                        details['features_created'] += count
                    
                    elif feature_type == 'readability':
                        feature_df, count = self._extract_readability_features(feature_df, text_column)
                        details['features_created'] += count
                    
                    elif feature_type == 'sentiment':
                        feature_df, count = self._extract_sentiment_features(feature_df, text_column)
                        details['features_created'] += count
                    
                    print(f"      ✅ {feature_type}: {count} features")
                    
                except Exception as e:
                    print(f"      ❌ Error extracting {feature_type} from {text_column}: {e}")
        
        return feature_df, details
    
    def _extract_basic_stats(self, df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, int]:
        """Extract basic text statistics."""
        text_series = df[text_column].fillna('')
        features_created = 0
        
        # Character-level features
        df[f'{text_column}_char_count'] = text_series.str.len()
        df[f'{text_column}_char_count_no_whitespace'] = text_series.str.replace(r'\s', '', regex=True).str.len()
        features_created += 2
        
        # Word-level features
        df[f'{text_column}_word_count'] = text_series.str.split().str.len()
        df[f'{text_column}_avg_word_length'] = text_series.apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).strip() else 0
        )
        features_created += 2
        
        # Sentence-level features (approximate)
        df[f'{text_column}_sentence_count'] = text_series.str.count(r'[.!?]+')
        df[f'{text_column}_avg_sentence_length'] = df[f'{text_column}_word_count'] / (df[f'{text_column}_sentence_count'] + 1)
        features_created += 2
        
        # Special character features
        df[f'{text_column}_digit_count'] = text_series.str.count(r'\d')
        df[f'{text_column}_uppercase_count'] = text_series.str.count(r'[A-Z]')
        df[f'{text_column}_lowercase_count'] = text_series.str.count(r'[a-z]')
        df[f'{text_column}_special_char_count'] = text_series.str.count(r'[^\w\s]')
        features_created += 4
        
        # Ratio features
        df[f'{text_column}_digit_ratio'] = df[f'{text_column}_digit_count'] / (df[f'{text_column}_char_count'] + 1)
        df[f'{text_column}_uppercase_ratio'] = df[f'{text_column}_uppercase_count'] / (df[f'{text_column}_char_count'] + 1)
        df[f'{text_column}_special_char_ratio'] = df[f'{text_column}_special_char_count'] / (df[f'{text_column}_char_count'] + 1)
        features_created += 3
        
        return df, features_created
    
    def _extract_linguistic_features(self, df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, int]:
        """Extract linguistic features."""
        text_series = df[text_column].fillna('')
        features_created = 0
        
        # Vocabulary richness
        df[f'{text_column}_unique_words'] = text_series.apply(
            lambda x: len(set(str(x).lower().split())) if str(x).strip() else 0
        )
        df[f'{text_column}_lexical_diversity'] = (
            df[f'{text_column}_unique_words'] / (df[f'{text_column}_word_count'] + 1)
        )
        features_created += 2
        
        # Word length statistics
        def get_word_length_stats(text):
            words = str(text).split()
            if not words:
                return 0, 0, 0
            lengths = [len(word) for word in words]
            return np.mean(lengths), np.std(lengths), max(lengths)
        
        stats = text_series.apply(get_word_length_stats)
        df[f'{text_column}_word_length_mean'] = stats.apply(lambda x: x[0] if x else 0)
        df[f'{text_column}_word_length_std'] = stats.apply(lambda x: x[1] if x else 0)
        df[f'{text_column}_word_length_max'] = stats.apply(lambda x: x[2] if x else 0)
        features_created += 3
        
        # Common word patterns
        df[f'{text_column}_contains_question'] = text_series.str.contains(r'\?', na=False).astype(int)
        df[f'{text_column}_contains_exclamation'] = text_series.str.contains(r'\!', na=False).astype(int)
        df[f'{text_column}_contains_url'] = text_series.str.contains(r'http[s]?://', na=False).astype(int)
        df[f'{text_column}_contains_mention'] = text_series.str.contains(r'@\w+', na=False).astype(int)
        df[f'{text_column}_contains_hashtag'] = text_series.str.contains(r'#\w+', na=False).astype(int)
        features_created += 5
        
        return df, features_created
    
    def _extract_readability_features(self, df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, int]:
        """Extract readability and complexity features."""
        text_series = df[text_column].fillna('')
        features_created = 0
        
        # Simple readability metrics
        def calculate_readability(text):
            if not text or not str(text).strip():
                return 0, 0, 0
            
            text_str = str(text)
            words = text_str.split()
            sentences = re.split(r'[.!?]+', text_str)
            sentences = [s for s in sentences if s.strip()]
            
            if not words or not sentences:
                return 0, 0, 0
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            complex_words = sum(1 for word in words if len(word) > 6)
            
            # Flesch Reading Ease (simplified)
            flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            # Coleman-Liau Index (simplified)
            coleman = (5.89 * avg_word_length) - (0.3 * avg_sentence_length) - 15.8
            # Complex word ratio
            complex_ratio = complex_words / len(words)
            
            return flesch, coleman, complex_ratio
        
        readability_stats = text_series.apply(calculate_readability)
        df[f'{text_column}_flesch_score'] = readability_stats.apply(lambda x: x[0] if x else 0)
        df[f'{text_column}_coleman_score'] = readability_stats.apply(lambda x: x[1] if x else 0)
        df[f'{text_column}_complex_word_ratio'] = readability_stats.apply(lambda x: x[2] if x else 0)
        features_created += 3
        
        return df, features_created
    
    def _extract_sentiment_features(self, df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, int]:
        """Extract sentiment-related features."""
        text_series = df[text_column].fillna('')
        features_created = 0
        
        # Sentiment word counts (simplified)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'better', 'positive', 'happy'}
        negative_words = {'bad', 'terrible', 'awful', 'worst', 'worse', 'negative', 'sad', 'unhappy', 'poor'}
        
        def count_sentiment_words(text):
            text_lower = str(text).lower()
            words = text_lower.split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            return pos_count, neg_count
        
        sentiment_counts = text_series.apply(count_sentiment_words)
        df[f'{text_column}_positive_words'] = sentiment_counts.apply(lambda x: x[0] if x else 0)
        df[f'{text_column}_negative_words'] = sentiment_counts.apply(lambda x: x[1] if x else 0)
        df[f'{text_column}_sentiment_balance'] = (
            (df[f'{text_column}_positive_words'] - df[f'{text_column}_negative_words']) / 
            (df[f'{text_column}_word_count'] + 1)
        )
        features_created += 3
        
        # Emotional intensity (exclamation and question marks)
        df[f'{text_column}_emotional_intensity'] = (
            text_series.str.count(r'!') + text_series.str.count(r'\?')
        ) / (df[f'{text_column}_char_count'] + 1)
        features_created += 1
        
        return df, features_created
    
    def create_text_embeddings(self, df: pd.DataFrame, text_column: str, method: str = 'tfidf') -> pd.DataFrame:
        """
        Create text embeddings using specified method.
        Note: This is a simplified version. In practice, you might use more advanced embeddings.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        text_series = df[text_column].fillna('')
        
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=50,  # Limit features for demonstration
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:  # count
            vectorizer = CountVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        try:
            embeddings = vectorizer.fit_transform(text_series)
            feature_names = vectorizer.get_feature_names_out()
            
            # Convert to DataFrame
            embedding_df = pd.DataFrame(
                embeddings.toarray(),
                columns=[f'{text_column}_embed_{name}' for name in feature_names],
                index=df.index
            )
            
            return pd.concat([df, embedding_df], axis=1)
        
        except Exception as e:
            print(f"      ⚠️  Error creating embeddings: {e}")
            return df
