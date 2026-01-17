# main.py
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from data_collector import DataCollector
import config

def create_directory_structure():
    """Create organized directory structure for storing data"""
    base_dir = "financial_data_large"
    sub_dirs = ["stock_data", "reddit_data", "news_data", "processed_data", "market_data", "synthetic_data"]
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(base_dir, sub_dir))
        print(f"Created directory structure: {base_dir}")
    return base_dir

def save_data(df, file_path, file_format='csv'):
    """Save DataFrame to specified format"""
    try:
        if file_format == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format == 'json':
            df.to_json(file_path, orient='records', indent=2)
        elif file_format == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)
        print(f"Data saved successfully: {file_path} ({len(df)} records)")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def test_news_api(api_key):
    """Test if NewsAPI key is valid"""
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            print("✓ NewsAPI key is valid!")
            return True
        else:
            print(f"✗ NewsAPI key invalid: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error testing NewsAPI: {e}")
        return False

def test_alpha_vantage(api_key):
    """Test if Alpha Vantage key is valid"""
    try:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={api_key}&limit=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'feed' in data:
                print("✓ Alpha Vantage key is valid!")
                return True
            else:
                print("✗ Alpha Vantage key invalid or limit reached")
                return False
        else:
            print(f"✗ Alpha Vantage error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error testing Alpha Vantage: {e}")
        return False

def main():
    # Initialize data collector
    collector = DataCollector()
    
    # Create directory structure
    base_dir = create_directory_structure()
    
    # EXPANDED Configuration for larger dataset
    stock_tickers = config.DEFAULT_SETTINGS['stock_tickers']  # 30 stocks
    reddit_subreddits = config.DEFAULT_SETTINGS['reddit_subreddits']  # 10 subreddits
    news_queries = config.DEFAULT_SETTINGS['news_queries']  # 15 queries
    
    # Date range (1 year for more data)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=config.DEFAULT_SETTINGS['lookback_days'])
    
    print("Starting LARGE data collection (target: ~50MB)...")
    
    # Test API keys
    news_api_valid = test_news_api(collector.news_api_key)
    alpha_vantage_valid = test_alpha_vantage(collector.alpha_vantage_key)
    
    # 1. Collect Stock Data (EXPANDED)
    print("\n=== Collecting EXPANDED Stock Data ===")
    all_stock_data = []
    for ticker in stock_tickers:
        try:
            print(f"Collecting data for {ticker}...")
            stock_data = collector.collect_stock_data(ticker, start_date, end_date)
            
            if not stock_data.empty:
                # Save individual file
                filename = f"{ticker}_full_year.csv"
                file_path = os.path.join(base_dir, "stock_data", filename)
                save_data(stock_data, file_path)
                all_stock_data.append(stock_data)
                print(f"✓ Collected {len(stock_data)} records for {ticker}")
            else:
                print(f"✗ No data found for {ticker}")
                
        except Exception as e:
            print(f"Error collecting stock data for {ticker}: {e}")
    
    # Save combined stock data
    if all_stock_data:
        combined_stocks = pd.concat(all_stock_data, ignore_index=True)
        combined_path = os.path.join(base_dir, "stock_data", "all_stocks_combined.csv")
        save_data(combined_stocks, combined_path)
        print(f"✓ Combined stock data: {len(combined_stocks)} records")
    
    # 2. Collect Reddit Data (EXPANDED)
    print("\n=== Collecting EXPANDED Reddit Data ===")
    all_reddit_posts = []
    all_reddit_comments = []
    
    for subreddit in reddit_subreddits:
        try:
            print(f"Collecting posts from r/{subreddit}...")
            reddit_data = collector.collect_reddit_posts(
                subreddit, 
                limit=config.DEFAULT_SETTINGS['reddit_post_limit']
            )
            
            if not reddit_data.empty:
                filename = f"reddit_{subreddit}_expanded.csv"
                file_path = os.path.join(base_dir, "reddit_data", filename)
                save_data(reddit_data, file_path)
                all_reddit_posts.append(reddit_data)
                print(f"✓ Collected {len(reddit_data)} posts from r/{subreddit}")
            else:
                print(f"✗ No posts found in r/{subreddit}")
            
            # Collect comments for additional data
            print(f"Collecting comments from r/{subreddit}...")
            comments_data = collector.collect_reddit_comments(subreddit, post_limit=50, comments_per_post=20)
            
            if not comments_data.empty:
                filename = f"reddit_comments_{subreddit}.csv"
                file_path = os.path.join(base_dir, "reddit_data", filename)
                save_data(comments_data, file_path)
                all_reddit_comments.append(comments_data)
                print(f"✓ Collected {len(comments_data)} comments from r/{subreddit}")
                
        except Exception as e:
            print(f"Error collecting Reddit data from r/{subreddit}: {e}")
    
    # Save combined Reddit data
    if all_reddit_posts:
        combined_posts = pd.concat(all_reddit_posts, ignore_index=True)
        combined_path = os.path.join(base_dir, "reddit_data", "all_reddit_posts_combined.csv")
        save_data(combined_posts, combined_path)
    
    if all_reddit_comments:
        combined_comments = pd.concat(all_reddit_comments, ignore_index=True)
        combined_path = os.path.join(base_dir, "reddit_data", "all_reddit_comments_combined.csv")
        save_data(combined_comments, combined_path)
    
    # 3. Collect News Data (EXPANDED)
    print("\n=== Collecting EXPANDED News Data ===")
    all_news_data = []
    
    # Option 1: NewsAPI with multiple queries
    if news_api_valid:
        print("Using NewsAPI for expanded news collection...")
        for query in news_queries:
            try:
                print(f"Collecting news for: {query}")
                news_data = collector.collect_news_data(
                    api_key=collector.news_api_key,
                    query=query,
                    from_date=start_date,
                    to_date=end_date,
                    page_size=50  # Larger page size
                )
                
                if not news_data.empty:
                    filename = f"news_{query.replace(' ', '_')}_expanded.json"
                    file_path = os.path.join(base_dir, "news_data", filename)
                    save_data(news_data, file_path, 'json')
                    all_news_data.append(news_data)
                    print(f"✓ Collected {len(news_data)} articles for '{query}'")
                else:
                    print(f"✗ No news found for '{query}'")
                    
            except Exception as e:
                print(f"Error collecting news data for '{query}': {e}")
    
    # Option 2: Alpha Vantage with more stocks
    if alpha_vantage_valid:
        print("Using Alpha Vantage for expanded news collection...")
        alpha_news = collector.collect_alpha_vantage_news(stock_tickers[:10], limit=50)  # More articles
        
        if not alpha_news.empty:
            filename = f"alpha_vantage_news_expanded.csv"
            file_path = os.path.join(base_dir, "news_data", filename)
            save_data(alpha_news, file_path)
            all_news_data.append(alpha_news)
            print(f"✓ Collected {len(alpha_news)} articles from Alpha Vantage")
        else:
            print("✗ No news collected from Alpha Vantage")
    
    # 4. Collect Market Indicators
    print("\n=== Collecting Market Indicators ===")
    market_indicators = collector.collect_market_indicators()
    if not market_indicators.empty:
        file_path = os.path.join(base_dir, "market_data", "market_indicators.csv")
        save_data(market_indicators, file_path)
        print(f"✓ Collected {len(market_indicators)} market indicator records")
    
    # 5. Generate Synthetic Data (to reach ~50MB)
    print("\n=== Generating Synthetic Data ===")
    synthetic_data = collector.generate_synthetic_financial_data(num_records=50000)  # 50K synthetic records
    if not synthetic_data.empty:
        file_path = os.path.join(base_dir, "synthetic_data", "synthetic_financial_data.csv")
        save_data(synthetic_data, file_path)
        print(f"✓ Generated {len(synthetic_data)} synthetic records")
    
    # 6. Create comprehensive summary report
    print("\n=== Data Collection Summary ===")
    summary = {
        'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock_tickers_collected': len(stock_tickers),
        'reddit_subreddits_collected': len(reddit_subreddits),
        'news_queries_used': len(news_queries),
        'total_stock_records': len(combined_stocks) if all_stock_data else 0,
        'total_reddit_posts': len(combined_posts) if all_reddit_posts else 0,
        'total_reddit_comments': len(combined_comments) if all_reddit_comments else 0,
        'total_news_articles': sum(len(df) for df in all_news_data) if all_news_data else 0,
        'total_market_indicators': len(market_indicators) if not market_indicators.empty else 0,
        'total_synthetic_records': len(synthetic_data) if not synthetic_data.empty else 0,
        'news_api_valid': news_api_valid,
        'alpha_vantage_valid': alpha_vantage_valid,
        'data_location': os.path.abspath(base_dir),
        'target_size': '~50MB'
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(base_dir, "collection_summary_detailed.csv")
    save_data(summary_df, summary_path)
    
    print("LARGE data collection completed!")
    print(f"All data saved to: {os.path.abspath(base_dir)}")
    
    # Calculate approximate size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    print(f"Approximate total dataset size: {total_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
