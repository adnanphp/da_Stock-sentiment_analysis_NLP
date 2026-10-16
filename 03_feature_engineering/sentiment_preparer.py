# sentiment_preparer.py
import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Any, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentTextPreparer:
    """
    Specialized text preparer for sentiment analysis.
    Focuses on preserving sentiment-bearing words and structures.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        
        # Sentiment-specific patterns and words to preserve
        self.sentiment_words = self._load_sentiment_words()
        self.negation_words = {'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nor'}
        
        # Enhanced stop words (remove common stop words but keep sentiment ones)
        self.enhanced_stop_words = self.stop_words - {
            'not', 'no', 'never', 'very', 'too', 'more', 'less', 'most', 'least',
            'good', 'bad', 'great', 'terrible', 'awesome', 'awful', 'excellent',
            'poor', 'best', 'worst', 'better', 'worse'
        }
    
    def _load_sentiment_words(self) -> set:
        """Load sentiment-bearing words to preserve."""
        # Basic sentiment words (in practice, you'd load from a lexicon)
        sentiment_words = {
            'good', 'bad', 'great', 'terrible', 'awesome', 'awful', 'excellent',
            'poor', 'best', 'worst', 'better', 'worse', 'amazing', 'horrible',
            'love', 'hate', 'like', 'dislike', 'wonderful', 'terrific', 'fantastic',
            'disappointing', 'satisfying', 'frustrating', 'exciting', 'boring',
            'happy', 'sad', 'angry', 'joyful', 'depressing', 'hopeful', 'fearful'
        }
        return sentiment_words
    
    def prepare_sentiment_text(self, series: pd.Series) -> pd.Series:
        """Prepare text for sentiment analysis."""
        print(f"    😊 Preparing {len(series)} texts for sentiment analysis...")
        
        prepared_series = series.copy()
        
        # Apply sentiment preparation pipeline
        prepared_series = prepared_series.apply(self._prepare_single_sentiment_text)
        
        success_rate = (prepared_series.notna().sum() / len(prepared_series)) * 100
        print(f"    ✅ Sentiment preparation success rate: {success_rate:.1f}%")
        
        return prepared_series
    
    def _prepare_single_sentiment_text(self, text: Any) -> Optional[str]:
        """Prepare a single text for sentiment analysis."""
        if pd.isna(text) or text is None:
            return None
        
        text = str(text).strip()
        if not text:
            return None
        
        try:
            # Step 1: Basic cleaning
            text = self._clean_sentiment_text(text)
            
            # Step 2: Handle negations
            text = self._handle_negations(text)
            
            # Step 3: Preserve sentiment words
            text = self._preserve_sentiment_features(text)
            
            # Step 4: Remove non-sentiment stop words
            text = self._remove_non_sentiment_stopwords(text)
            
            # Step 5: Normalize for sentiment analysis
            text = self._normalize_sentiment_text(text)
            
            return text if text else None
            
        except Exception as e:
            print(f"      ⚠️  Error preparing sentiment text: {e}")
            return None
    
    def _clean_sentiment_text(self, text: str) -> str:
        """Clean text while preserving sentiment features."""
        # HTML unescape
        import html
        text = html.unescape(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but preserve punctuation for sentiment
        text = re.sub(r'[^\w\s!?]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        return text
    
    def _handle_negations(self, text: str) -> str:
        """Handle negation patterns for sentiment analysis."""
        words = text.split()
        processed_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check if current word is a negation
            if word in self.negation_words and i + 1 < len(words):
                # Mark next word with _NEG suffix
                processed_words.append(word)
                processed_words.append(words[i + 1] + '_NEG')
                i += 2  # Skip next word since we've processed it
            else:
                processed_words.append(word)
                i += 1
        
        return ' '.join(processed_words)
    
    def _preserve_sentiment_features(self, text: str) -> str:
        """Preserve sentiment-bearing features in text."""
        words = text.split()
        
        # Identify and potentially emphasize sentiment words
        processed_words = []
        for word in words:
            if word in self.sentiment_words:
                # You could add emphasis or keep as is
                processed_words.append(word)
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def _remove_non_sentiment_stopwords(self, text: str) -> str:
        """Remove stop words that don't carry sentiment."""
        words = text.split()
        filtered_words = [word for word in words if word not in self.enhanced_stop_words]
        return ' '.join(filtered_words)
    
    def _normalize_sentiment_text(self, text: str) -> str:
        """Normalize text specifically for sentiment analysis."""
        # Expand contractions to capture full meaning
        text = self._expand_sentiment_contractions(text)
        
        # Handle intensifiers and diminishers
        text = self._handle_intensifiers(text)
        
        return text
    
    def _expand_sentiment_contractions(self, text: str) -> str:
        """Expand contractions for better sentiment analysis."""
        contractions = {
            r"won't": "will not",
            r"can't": "can not",
            r"n't": " not",
            r"'re": " are",
            r"'s": " is",
            r"'d": " would",
            r"'ll": " will",
            r"'t": " not",
            r"'ve": " have",
            r"'m": " am",
            r"what's": "what is",
            r"it's": "it is",
            r"that's": "that is",
            r"there's": "there is"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        
        return text
    
    def _handle_intensifiers(self, text: str) -> str:
        """Handle intensifiers and diminishers for sentiment analysis."""
        intensifiers = {
            'very': 'INT_',
            'really': 'INT_', 
            'extremely': 'INT_',
            'incredibly': 'INT_',
            'somewhat': 'DIM_',
            'slightly': 'DIM_',
            'barely': 'DIM_'
        }
        
        words = text.split()
        processed_words = []
        
        for word in words:
            if word in intensifiers:
                processed_words.append(intensifiers[word])
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def calculate_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores using VADER."""
        if not text:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        scores = self.sia.polarity_scores(text)
        return scores
    
    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract sentiment-specific features from text."""
        if not text:
            return {}
        
        # Basic text features
        words = word_tokenize(text.lower())
        words_clean = [word for word in words if word not in string.punctuation]
        
        # Sentiment scores
        sentiment_scores = self.calculate_sentiment_scores(text)
        
        # Sentiment word counts
        positive_words = [word for word in words_clean if self.sia.polarity_scores(word)['compound'] > 0.1]
        negative_words = [word for word in words_clean if self.sia.polarity_scores(word)['compound'] < -0.1]
        
        features = {
            'text_length': len(text),
            'word_count': len(words_clean),
            'positive_word_count': len(positive_words),
            'negative_word_count': len(negative_words),
            'positive_ratio': len(positive_words) / len(words_clean) if words_clean else 0,
            'negative_ratio': len(negative_words) / len(words_clean) if words_clean else 0,
            'sentiment_compound': sentiment_scores['compound'],
            'sentiment_positive': sentiment_scores['pos'],
            'sentiment_negative': sentiment_scores['neg'],
            'sentiment_neutral': sentiment_scores['neu'],
            'has_negation': any(word in self.negation_words for word in words_clean),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
        }
        
        return features
