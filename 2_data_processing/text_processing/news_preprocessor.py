# news_preprocessor.py
import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Any, Optional
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import html

class NewsArticlePreprocessor:
    """
    Specialized preprocessor for news articles and formal content.
    Focuses on maintaining information quality while cleaning and normalizing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # News-specific patterns
        self.news_patterns = {
            'byline': r'^By\s+[\w\s]+$',
            'dateline': r'^[A-Z]+\s*[—-]\s*',
            'html_tags': r'<[^>]+>',
            'special_chars': r'[^\w\s.,!?;:()\-]',
            'multiple_spaces': r'\s+',
            'multiple_newlines': r'\n+',
        }
    
    def preprocess_articles(self, series: pd.Series) -> pd.Series:
        """Preprocess a pandas Series containing news articles."""
        print(f"    📰 Preprocessing {len(series)} news articles...")
        
        processed_series = series.copy()
        
        # Apply preprocessing pipeline
        processed_series = processed_series.apply(self._preprocess_single_article)
        
        success_rate = (processed_series.notna().sum() / len(processed_series)) * 100
        print(f"    ✅ News preprocessing success rate: {success_rate:.1f}%")
        
        return processed_series
    
    def _preprocess_single_article(self, text: Any) -> Optional[str]:
        """Preprocess a single news article."""
        if pd.isna(text) or text is None:
            return None
        
        text = str(text).strip()
        if not text:
            return None
        
        try:
            # Step 1: HTML unescape and remove HTML tags
            text = html.unescape(text)
            text = re.sub(self.news_patterns['html_tags'], ' ', text)
            
            # Step 2: Remove bylines and datelines
            text = self._remove_news_metadata(text)
            
            # Step 3: Clean and normalize text
            text = self._clean_news_text(text)
            
            # Step 4: Advanced NLP processing (optional)
            if self.config.get('enable_advanced_nlp', True):
                text = self._apply_advanced_nlp(text)
            
            # Step 5: Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text if text else None
            
        except Exception as e:
            print(f"      ⚠️  Error preprocessing article: {e}")
            return None
    
    def _remove_news_metadata(self, text: str) -> str:
        """Remove news-specific metadata like bylines and datelines."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip bylines
            if re.match(self.news_patterns['byline'], line, re.IGNORECASE):
                continue
            # Clean datelines
            line = re.sub(self.news_patterns['dateline'], '', line)
            if line:
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _clean_news_text(self, text: str) -> str:
        """Clean and normalize news text."""
        # Handle encoding issues
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Remove special characters but preserve punctuation for sentence structure
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle case normalization based on config
        case_strategy = self.config.get('case_strategy', 'lower')  # 'lower', 'sentence', 'preserve'
        if case_strategy == 'lower':
            text = text.lower()
        elif case_strategy == 'sentence':
            # Capitalize first letter of each sentence
            sentences = sent_tokenize(text)
            sentences = [sentence.capitalize() for sentence in sentences]
            text = ' '.join(sentences)
        
        return text
    
    def _apply_advanced_nlp(self, text: str) -> str:
        """Apply advanced NLP processing."""
        nlp_level = self.config.get('nlp_level', 'medium')  # 'light', 'medium', 'heavy'
        
        if nlp_level == 'light':
            return text
        
        # Tokenize
        tokens = word_tokenize(text)
        
        if nlp_level == 'medium':
            # Remove stopwords
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
            # Remove short tokens
            tokens = [token for token in tokens if len(token) > 2]
        
        elif nlp_level == 'heavy':
            # Remove stopwords
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
            
            # Remove short tokens
            tokens = [token for token in tokens if len(token) > 2]
            
            # Lemmatization
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_news_features(self, text: str) -> Dict[str, Any]:
        """Extract news-specific features from text."""
        if not text:
            return {}
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove punctuation from word count
        words_clean = [word for word in words if word not in string.punctuation]
        
        features = {
            'sentence_count': len(sentences),
            'word_count': len(words_clean),
            'avg_sentence_length': len(words_clean) / len(sentences) if sentences else 0,
            'avg_word_length': np.mean([len(word) for word in words_clean]) if words_clean else 0,
            'unique_word_ratio': len(set(words_clean)) / len(words_clean) if words_clean else 0,
            'stopword_ratio': len([w for w in words_clean if w in self.stop_words]) / len(words_clean) if words_clean else 0,
            'has_quotes': '"' in text or "'" in text,
            'has_numbers': bool(re.search(r'\d', text)),
            'readability_score': self._calculate_readability(text),
        }
        
        return features
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score (Flesch-like)."""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words_clean = [word for word in words if word not in string.punctuation]
        
        if not sentences or not words_clean:
            return 0.0
        
        avg_sentence_length = len(words_clean) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words_clean])
        
        # Simplified readability score
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        return max(0.0, min(100.0, readability))
