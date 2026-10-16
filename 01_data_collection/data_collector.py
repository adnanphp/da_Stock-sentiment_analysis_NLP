# data_collector.py
import yfinance as yf
import praw
import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import json
import config
import random

class DataCollector:
    def __init__(self):
        # Initialize with configuration
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CONFIG['client_id'],
            client_secret=config.REDDIT_CONFIG['client_secret'],
            user_agent=config.REDDIT_CONFIG['user_agent']
        )
        self.news_api_key = config.NEWS_API_KEY
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        
    def collect_stock_data(self, ticker, start_date, end_date, interval='1d'):
        """Collect historical stock data with multiple intervals"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get daily data for full period
            data_daily = stock.history(start=start_date, end=end_date, interval='1d')
            data_daily = data_daily.reset_index()
            data_daily['ticker'] = ticker
            data_daily['interval'] = '1d'
            
            # Get weekly data for larger context
            data_weekly = stock.history(start=start_date, end=end_date, interval='1wk')
            data_weekly = data_weekly.reset_index()
            data_weekly['ticker'] = ticker
            data_weekly['interval'] = '1wk'
            
            # Get additional stock info for more data
            info = stock.info
            additional_data = {
                'ticker': ticker,
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', ''),
                'employees': info.get('fullTimeEmployees', ''),
                'description': info.get('longBusinessSummary', ''),
                'country': info.get('country', ''),
                'website': info.get('website', ''),
                'currency': info.get('currency', '')
            }
            
            # Combine all data
            combined_data = pd.concat([data_daily, data_weekly], ignore_index=True)
            
            # Add additional info as repeated columns
            for key, value in additional_data.items():
                combined_data[key] = value
                
            return combined_data
        except Exception as e:
            print(f"Error collecting stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_reddit_posts(self, subreddit, limit=2000, time_filter='all'):
        """Collect financial posts from Reddit with multiple time filters"""
        try:
            posts = []
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            # Collect from different time periods for more data
            time_filters = ['all', 'year', 'month'] if time_filter == 'all' else [time_filter]
            
            for time_period in time_filters:
                try:
                    for post in subreddit_obj.top(time_filter=time_period, limit=limit//len(time_filters)):
                        posts.append({
                            'post_id': post.id,
                            'post_date': datetime.fromtimestamp(post.created_utc).date(),
                            'post_datetime': datetime.fromtimestamp(post.created_utc),
                            'title': post.title,
                            'content': post.selftext,
                            'upvotes': post.score,
                            'comments': post.num_comments,
                            'awards': getattr(post, 'total_awards_received', 0),
                            'subreddit': subreddit,
                            'platform': 'reddit',
                            'url': post.url,
                            'time_period': time_period,
                            'flair': getattr(post, 'link_flair_text', ''),
                            'upvote_ratio': getattr(post, 'upvote_ratio', 1.0)
                        })
                    
                    # Also get new posts
                    for post in subreddit_obj.new(limit=limit//len(time_filters)):
                        posts.append({
                            'post_id': post.id,
                            'post_date': datetime.fromtimestamp(post.created_utc).date(),
                            'post_datetime': datetime.fromtimestamp(post.created_utc),
                            'title': post.title,
                            'content': post.selftext,
                            'upvotes': post.score,
                            'comments': post.num_comments,
                            'awards': getattr(post, 'total_awards_received', 0),
                            'subreddit': subreddit,
                            'platform': 'reddit',
                            'url': post.url,
                            'time_period': 'new',
                            'flair': getattr(post, 'link_flair_text', ''),
                            'upvote_ratio': getattr(post, 'upvote_ratio', 1.0)
                        })
                        
                except Exception as e:
                    print(f"Error in time period {time_period} for r/{subreddit}: {e}")
                    continue
            
            return pd.DataFrame(posts)
        except Exception as e:
            print(f"Error collecting Reddit data from r/{subreddit}: {e}")
            return pd.DataFrame()
    
    def collect_reddit_comments(self, subreddit, post_limit=100, comments_per_post=50):
        """Collect comments from Reddit posts for additional data"""
        try:
            comments_data = []
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            for post in subreddit_obj.hot(limit=post_limit):
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:comments_per_post]:
                        if hasattr(comment, 'body'):
                            comments_data.append({
                                'comment_id': comment.id,
                                'post_id': post.id,
                                'post_title': post.title,
                                'subreddit': subreddit,
                                'comment_date': datetime.fromtimestamp(comment.created_utc),
                                'comment_text': comment.body,
                                'comment_upvotes': comment.score,
                                'comment_awards': getattr(comment, 'total_awards_received', 0),
                                'is_submitter': comment.is_submitter
                            })
                except Exception as e:
                    continue
            
            return pd.DataFrame(comments_data)
        except Exception as e:
            print(f"Error collecting comments from r/{subreddit}: {e}")
            return pd.DataFrame()
    
    def collect_news_data(self, api_key, query, from_date, to_date, page_size=100):
        """Collect news data from NewsAPI with larger page size"""
        try:
            from_date_str = from_date.strftime('%Y-%m-%d') if isinstance(from_date, datetime) else from_date
            to_date_str = to_date.strftime('%Y-%m-%d') if isinstance(to_date, datetime) else to_date
            
            all_articles = []
            
            # Get multiple pages for more data
            for page in range(1, 6):  # Get 5 pages
                url = f"https://newsapi.org/v2/everything?q={query}&from={from_date_str}&to={to_date_str}&sortBy=publishedAt&pageSize={page_size}&page={page}&apiKey={api_key}"
                
                print(f"Requesting news for: {query} - Page {page}")
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    if not articles:
                        break
                    
                    for article in articles:
                        all_articles.append({
                            'source': article.get('source', {}).get('name', ''),
                            'author': article.get('author', ''),
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'query': query,
                            'page': page
                        })
                    
                    print(f"✓ Page {page}: {len(articles)} articles")
                    
                    # Rate limiting
                    time.sleep(1)
                else:
                    error_msg = response.json().get('message', 'Unknown error')
                    print(f"News API error {response.status_code}: {error_msg}")
                    break
            
            print(f"✓ Collected {len(all_articles)} total news articles for '{query}'")
            return pd.DataFrame(all_articles)
                
        except Exception as e:
            print(f"Error collecting news data for '{query}': {e}")
            return pd.DataFrame()
    
    def collect_alpha_vantage_news(self, tickers, limit=100):
        """Collect news from Alpha Vantage with increased limits"""
        try:
            all_news = []
            
            for ticker in tickers:
                print(f"Collecting Alpha Vantage news for {ticker}...")
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.alpha_vantage_key}&limit={limit}"
                
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    feed = data.get('feed', [])
                    
                    for article in feed:
                        # Extract more detailed sentiment data
                        ticker_sentiment = article.get('ticker_sentiment', [])
                        sentiment_data = {}
                        for sentiment in ticker_sentiment:
                            ticker = sentiment.get('ticker', '')
                            sentiment_data[f"sentiment_{ticker}"] = sentiment.get('ticker_sentiment_score', 0)
                        
                        all_news.append({
                            'source': 'Alpha Vantage',
                            'tickers': [s.get('ticker', '') for s in ticker_sentiment],
                            'title': article.get('title', ''),
                            'summary': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('time_published', ''),
                            'authors': article.get('authors', []),
                            'overall_sentiment': article.get('overall_sentiment_score', 0),
                            'platform': 'alpha_vantage',
                            **sentiment_data
                        })
                    
                    print(f"✓ Collected {len(feed)} articles for {ticker}")
                    time.sleep(1)  # Rate limiting
                else:
                    print(f"Alpha Vantage error for {ticker}: {response.status_code}")
            
            return pd.DataFrame(all_news)
            
        except Exception as e:
            print(f"Error collecting Alpha Vantage news: {e}")
            return pd.DataFrame()
    
    def collect_market_indicators(self):
        """Collect additional market indicators for more data"""
        try:
            indicators = {}
            
            # S&P 500 data
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="1y")
            sp500_data = sp500_data.reset_index()
            sp500_data['indicator'] = 'S&P 500'
            indicators['sp500'] = sp500_data
            
            # VIX data
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1y")
            vix_data = vix_data.reset_index()
            vix_data['indicator'] = 'VIX'
            indicators['vix'] = vix_data
            
            # Combine all indicators
            all_indicators = pd.concat(indicators.values(), ignore_index=True)
            return all_indicators
            
        except Exception as e:
            print(f"Error collecting market indicators: {e}")
            return pd.DataFrame()
    
    def generate_synthetic_financial_data(self, num_records=10000):
        """Generate synthetic financial data to increase dataset size"""
        try:
            synthetic_data = []
            
            for i in range(num_records):
                synthetic_data.append({
                    'synthetic_id': i,
                    'timestamp': datetime.now() - timedelta(days=random.randint(0, 365)),
                    'price': random.uniform(100, 500),
                    'volume': random.randint(1000, 1000000),
                    'sentiment_score': random.uniform(-1, 1),
                    'volatility': random.uniform(0.1, 0.5),
                    'market_cap_category': random.choice(['Large', 'Mid', 'Small']),
                    'sector': random.choice(['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer']),
                    'data_type': 'synthetic'
                })
            
            return pd.DataFrame(synthetic_data)
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return pd.DataFrame()
