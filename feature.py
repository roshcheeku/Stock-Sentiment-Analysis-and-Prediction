import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ta
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

# Function to calculate VADER sentiment scores
def calculate_vader_sentiment(text):
    """Calculate VADER sentiment scores for given text."""
    try:
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(str(text))['compound']
    except Exception as e:
        print(f"Error calculating sentiment: {e}")
        return 0

# Function to download stock data
def get_stock_data(ticker, start_date, end_date):
    """Download stock data using yfinance."""
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        stock.reset_index(inplace=True)
        print(f"\nDownloaded {ticker} stock data:")
        print(f"Date range: {stock['Date'].min()} to {stock['Date'].max()}")
        print(f"Number of records: {len(stock)}")
        return stock
    except Exception as e:
        print(f"Error downloading stock data: {e}")
        return None

def preprocess_stock_data(stock_symbol, sentiment_file, start_date='2023-01-01'):
    """
    Preprocess stock and sentiment data with proper error checking.
    """
    print(f"\nProcessing data for {stock_symbol}:")

    # Load sentiment data
    sentiment_data = pd.read_csv(sentiment_file)
    print(f"\nLoaded sentiment data from {sentiment_file}")
    print(f"Columns available: {sentiment_data.columns.tolist()}")

    # Calculate VADER sentiment if not already present
    if 'cleaned_text' in sentiment_data.columns:
        print("Calculating VADER sentiment scores...")
        sentiment_data['vader_avg_sentiment'] = sentiment_data['cleaned_text'].apply(calculate_vader_sentiment)

    # Convert sentiment data dates to datetime without timezone
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.tz_localize(None)

    # Download stock data
    end_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = get_stock_data(stock_symbol, start_date, end_date)

    if stock_data is None:
        raise ValueError(f"Failed to download stock data for {stock_symbol}")

    # Flatten column names if needed
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns]

    # Debugging: Check stock data structure
    print("\nStock data columns:", stock_data.columns.tolist())
    print("First few rows of stock data:")
    print(stock_data.head())

    # Rename 'Date_' to 'Date' if present
    if 'Date_' in stock_data.columns:
        stock_data.rename(columns={'Date_': 'Date'}, inplace=True)

    # Check and rename the 'Close' column
    close_column = f"Close_{stock_symbol}" if f"Close_{stock_symbol}" in stock_data.columns else "Close"
    if close_column not in stock_data.columns:
        raise ValueError(f"'{close_column}' column not found in stock data. Available columns: {stock_data.columns.tolist()}")
    stock_data.rename(columns={close_column: 'close'}, inplace=True)

    # Ensure 'Date' column exists
    if 'Date' not in stock_data.columns:
        raise ValueError("'Date' column not found in stock data after renaming.")

    # Convert stock data dates to datetime without timezone
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)

    # Ensure 'close' is a numeric Series
    print("\nConverting 'close' column to numeric...")
    stock_data['close'] = pd.to_numeric(stock_data['close'], errors='coerce')
    if stock_data['close'].isnull().all():
        raise ValueError("'close' column contains all NaN values after conversion.")

    # Calculate SMA
    stock_data['SMA_50'] = ta.trend.sma_indicator(stock_data['close'], window=50)
    stock_data['SMA_200'] = ta.trend.sma_indicator(stock_data['close'], window=200)

    # Merge stock and sentiment data
    merged_data = pd.merge_asof(
        stock_data.sort_values('Date'),
        sentiment_data.sort_values('Date'),
        on='Date'
    )
    return merged_data

# Function to plot stock and sentiment data
def plot_stock_with_sentiment(data, stock_name):
    """Plot stock price and sentiment data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

    # Stock price plot
    ax1.plot(data['Date'], data['close'], label='Close Price')
    ax1.plot(data['Date'], data['SMA_50'], label='50-day SMA')
    ax1.plot(data['Date'], data['SMA_200'], label='200-day SMA')
    ax1.set_title(f'{stock_name} Stock Price with Technical Indicators')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Sentiment plot
    ax2.plot(data['Date'], data['vader_avg_sentiment'], label='Sentiment', color='purple')
    ax2.fill_between(data['Date'], data['vader_avg_sentiment'], 0, alpha=0.2)
    ax2.set_title('Sentiment Score')
    ax2.set_ylabel('VADER Sentiment')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    try:
        print("Starting data preprocessing...")

        # Process AAPL data
        aapl_data = preprocess_stock_data(
            'AAPL',
            'cleaned_twitter_data.csv'
        )

        # Process TSLA data
        tsla_data = preprocess_stock_data(
            'TSLA',
            'cleaned_reddit_data.csv'
        )

        # Plot the processed data
        print("\nGenerating plots...")
        plot_stock_with_sentiment(aapl_data, 'AAPL')
        plot_stock_with_sentiment(tsla_data, 'TSLA')

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
