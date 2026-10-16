
01 — Data Collection
Fetches raw data from four sources:

Source	API	Data
Stock prices	yFinance	OHLC, volume, adjusted close
Financial news	Alpha Vantage, NewsAPI	Article text, timestamps, tickers
Reddit sentiment	PRAW	Posts and comments from financial subreddits
Market indicators	yFinance	S&P 500, VIX, volatility measures
Files
File	Purpose
data_collector.py	Orchestrates collection from all sources
data_profiler.py	Summarizes the shape and quality of collected data
data_quality_checker.py	Basic quality checks (nulls, duplicates, ranges)
data_quality_checker_fixed.py	Corrected version
main.py	Entry point
run_data_quality_assessment.py	Runs the quality checks in isolation
config.py	API credentials and source configuration
Credentials
config.py requires API keys. Do not commit them. Use environment
variables or a .env file (gitignored):

text
ALPHA_VANTAGE_API_KEY=...
NEWS_API_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=...
Output
Raw data is written to 14_data_storage/ (gitignored — regeneratable
from the APIs).

Stage boundary
Input: none (source of truth).
Output: raw CSVs ready for 02_data_processing/.
