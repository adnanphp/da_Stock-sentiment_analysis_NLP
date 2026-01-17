# reddit_cleaner.py
import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Any, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import html

class RedditTextCleaner:
    """
    Specialized text cleaner for Reddit and social media content.
    Handles informal language, markdown, URLs, and social media patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.stop_words = set(stopwords.words('english'))
        
        # Reddit-specific patterns
        self.reddit_patterns = {
            'user_mention': r'u/[\w-]+',
            'subreddit_mention': r'r/[\w-]+',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'markdown_link': r'\[([^\]]+)\]\(([^)]+)\)',
            'bold_italic': r'\*{1,3}([^\*]+)\*{1,3}',
            'code_block': r'`{1,3}([^`]+)`{1,3}',
            'quote_block': r'^>.*$',
            'spoiler': r'\!>([^>]+)\!<',
        }
    
    def clean_text_column(self, series: pd.Series) -> pd.Series:
        """Clean a pandas Series containing Reddit/social media text."""
        print(f"    🧹 Cleaning {len(series)} Reddit/social media texts...")
        
        cleaned_series = series.copy()
        
        # Apply cleaning pipeline
        cleaned_series = cleaned_series.apply(self._clean_single_text)
        
        success_rate = (cleaned_series.notna().sum() / len(cleaned_series)) * 100
        print(f"    ✅ Reddit cleaning success rate: {success_rate:.1f}%")
        
        return cleaned_series
    
    def _clean_single_text(self, text: Any) -> Optional[str]:
        """Clean a single text entry."""
        if pd.isna(text) or text is None:
            return None
        
        text = str(text).strip()
        if not text:
            return None
        
        try:
            # Step 1: HTML unescape
            text = html.unescape(text)
            
            # Step 2: Remove Reddit-specific formatting
            text = self._remove_reddit_formatting(text)
            
            # Step 3: Remove URLs
            text = self._remove_urls(text)
            
            # Step 4: Handle emojis
            text = self._handle_emojis(text)
            
            # Step 5: Normalize text
            text = self._normalize_text(text)
            
            # Step 6: Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text if text else None
            
        except Exception as e:
            print(f"      ⚠️  Error cleaning text: {e}")
            return None
    
    def _remove_reddit_formatting(self, text: str) -> str:
        """Remove Reddit-specific formatting and markdown."""
        # Remove user mentions but keep the username as regular text
        text = re.sub(self.reddit_patterns['user_mention'], lambda m: m.group(0).replace('u/', ''), text)
        
        # Remove subreddit mentions but keep the subreddit name
        text = re.sub(self.reddit_patterns['subreddit_mention'], lambda m: m.group(0).replace('r/', ''), text)
        
        # Remove markdown links but keep the link text
        text = re.sub(self.reddit_patterns['markdown_link'], r'\1', text)
        
        # Remove bold/italic formatting but keep the text
        text = re.sub(self.reddit_patterns['bold_italic'], r'\1', text)
        
        # Remove code blocks but keep the code content
        text = re.sub(self.reddit_patterns['code_block'], r'\1', text)
        
        # Remove quote block markers
        text = re.sub(self.reddit_patterns['quote_block'], '', text)
        
        # Remove spoiler tags but keep the content
        text = re.sub(self.reddit_patterns['spoiler'], r'\1', text)
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return re.sub(self.reddit_patterns['url'], '', text)
    
    def _handle_emojis(self, text: str) -> str:
        """Handle emojis - either remove or convert to text description."""
        strategy = self.config.get('emoji_strategy', 'remove')  # 'remove', 'describe', 'keep'
        
        if strategy == 'remove':
            # Remove all emojis
            text = emoji.replace_emoji(text, replace='')
        elif strategy == 'describe':
            # Convert emojis to text descriptions
            text = emoji.demojize(text, delimiters=(' ', ' '))
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistency."""
        # Convert to lowercase
        text = text.lower()
        
        # Expand common contractions
        text = self._expand_contractions(text)
        
        # Remove punctuation (keep some for sentiment)
        text = re.sub(r'[^\w\s!?]', '', text)
        
        # Handle repeated characters (e.g., "soooo" -> "so")
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Remove extra whitespace (again after processing)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",
            r"'re": " are",
            r"'s": " is",
            r"'d": " would",
            r"'ll": " will",
            r"'t": " not",
            r"'ve": " have",
            r"'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract Reddit-specific features from text."""
        if not text:
            return {}
        
        features = {
            'has_url': bool(re.search(self.reddit_patterns['url'], text)),
            'has_user_mention': bool(re.search(self.reddit_patterns['user_mention'], text)),
            'has_subreddit_mention': bool(re.search(self.reddit_patterns['subreddit_mention'], text)),
            'has_emoji': emoji.emoji_count(text) > 0,
            'word_count': len(text.split()),
            'char_count': len(text),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        }
        
        return features
