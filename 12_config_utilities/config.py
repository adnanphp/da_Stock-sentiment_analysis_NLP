# config.py
# API Configuration - Replace with your actual API keys

REDDIT_CONFIG = {
    'client_id': 'iZPV5FpSNQ0O3I_1MjJV8g',
    'client_secret': 'j7WQ7pLRnb085anKlJfNkBVvMk7VSg',
    'user_agent': 'financial_analysis_v1.0'
}

# Replace with your valid NewsAPI key
NEWS_API_KEY = 'c4d5881c165c48a98d5fb05e97ef043a'

# Alternative news sources (free options)
ALPHA_VANTAGE_API_KEY = 'S63A4RK16GXHSBM6'  # Free alternative

# EXPANDED Data collection settings for larger dataset
DEFAULT_SETTINGS = {
    'reddit_post_limit': 2000,  # Increased from 500
    'lookback_days': 365,       # Increased from 30 days - 1 year of data
    'stock_tickers': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA', 'JPM', 'JNJ', 
                     'V', 'WMT', 'PG', 'DIS', 'BAC', 'XOM', 'INTC', 'CSCO', 'PFE', 'VZ',
                     'ADBE', 'CRM', 'NKE', 'T', 'ABT', 'COST', 'TMO', 'LLY', 'AVGO', 'UNH'],  # 30 stocks
    'reddit_subreddits': ['stocks', 'investing', 'wallstreetbets', 'finance', 'economy', 
                         'StockMarket', 'trading', 'ValueInvesting', 'dividends', 'options'],  # 10 subreddits
    'news_queries': ['stock market', 'investing', 'economy', 'financial news', 'trading',
                    'earnings report', 'Federal Reserve', 'inflation', 'interest rates', 'cryptocurrency',
                    'tech stocks', 'banking', 'real estate', 'commodities', 'forex']  # 15 queries
}
