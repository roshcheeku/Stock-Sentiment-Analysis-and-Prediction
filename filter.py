import pandas as pd

# Load your datasets
twitter_data = pd.read_csv('cleaned_twitter_data.csv')  # Twitter data
reddit_data = pd.read_csv('cleaned_reddit_data.csv')    # Reddit data
aapl_stock_data = pd.read_csv('cleaned_AAPL_stock_data.csv')  # AAPL stock data
tsla_stock_data = pd.read_csv('cleaned_TSLA_stock_data.csv')  # TSLA stock data

# Print the column names to check if they match the expected columns
print("Twitter Data Columns:", twitter_data.columns)
print("Reddit Data Columns:", reddit_data.columns)
print("AAPL Stock Data Columns:", aapl_stock_data.columns)
print("TSLA Stock Data Columns:", tsla_stock_data.columns)

# Convert 'Date' (Twitter/Reddit) or 'date' (Stock) to datetime
# For Twitter and Reddit, the column might be 'Date'
# For AAPL/TSLA stock data, the column is 'date'
twitter_data['Date'] = pd.to_datetime(twitter_data['Date'], errors='coerce')
reddit_data['Date'] = pd.to_datetime(reddit_data['Date'], errors='coerce')
aapl_stock_data['Date'] = pd.to_datetime(aapl_stock_data['Date'], errors='coerce')
tsla_stock_data['Date'] = pd.to_datetime(tsla_stock_data['Date'], errors='coerce')

# Define the target date to filter data to (2024-12-30)
target_date = pd.to_datetime('2024-12-30')

# Filter the data to only include rows with the target date
twitter_data_filtered = twitter_data[twitter_data['Date'] == target_date]
reddit_data_filtered = reddit_data[reddit_data['Date'] == target_date]
aapl_stock_data_filtered = aapl_stock_data[aapl_stock_data['Date'] == target_date]
tsla_stock_data_filtered = tsla_stock_data[tsla_stock_data['Date'] == target_date]

# Check the resulting filtered datasets
print(f"Filtered Twitter Data: {twitter_data_filtered.shape}")
print(f"Filtered Reddit Data: {reddit_data_filtered.shape}")
print(f"Filtered AAPL Stock Data: {aapl_stock_data_filtered.shape}")
print(f"Filtered TSLA Stock Data: {tsla_stock_data_filtered.shape}")

# Optionally, save these filtered datasets to new CSV files
twitter_data_filtered.to_csv('filtered_twitter_data.csv', index=False)
reddit_data_filtered.to_csv('filtered_reddit_data.csv', index=False)
aapl_stock_data_filtered.to_csv('filtered_aapl_stock_data.csv', index=False)
tsla_stock_data_filtered.to_csv('filtered_tsla_stock_data.csv', index=False)
