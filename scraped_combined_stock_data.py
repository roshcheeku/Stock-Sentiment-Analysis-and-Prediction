import yfinance as yf
import pandas as pd

# Define stock tickers and data range
stocks = ['AAPL', 'TSLA', 'AMZN', 'GOOG', 'MSFT']
start_date = '2022-01-01'  # Adjust date range as needed
end_date = '2025-01-01'

# Scrape stock data
for stock in stocks:
    stock_data = yf.download(stock, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)  # Reset index for consistency
    stock_data['Stock_Type'] = stock
    stock_data.to_csv(f'{stock}_stock_data.csv', index=False)  # Save to CSV
    print(f"Downloaded data for {stock}, Rows: {len(stock_data)}")

# Combine all stock data into one DataFrame
combined_data = pd.concat([pd.read_csv(f'{stock}_stock_data.csv') for stock in stocks], ignore_index=True)
combined_data.to_csv('scraped_combined_stock_data.csv', index=False)
print("Combined Stock Data Saved:", combined_data.shape)
