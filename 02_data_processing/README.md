
02 — Data Processing
Cleans and standardizes raw data from all four sources. Prepares it for
feature engineering.

Subfolders
Subfolder	Purpose
datetime_handling/	Timezone normalization, datetime standardization, time feature creation
data_quality/	Profiling, quality checks, verification
data_types/	Type conversion, manual type mapping
missing_values/	Missing-value analysis and imputation (multiple strategies)
text_processing/	Text preprocessing for news and Reddit content
Why so many subfolders
Each concern is isolated:

Datetime — every source has a different timestamp format and timezone

Types — numeric columns come as strings; dates come as strings; tickers come as objects

Missing values — different sources have different missingness patterns

Text — news and Reddit need different preprocessing (URLs, mentions, casing)

Inputs
Raw CSVs from 01_data_collection/.

Output
Cleaned CSVs, still in 14_data_storage/ (gitignored).

Stage boundary
Input: raw data.
Output: clean, typed data ready for feature engineering.
