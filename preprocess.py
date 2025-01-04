import pandas as pd
import numpy as np

# Load the stock data for TSLA and AAPL
tsla_data = pd.read_csv('TSLA_stock_data.csv')
aapl_data = pd.read_csv('AAPL_stock_data.csv')

# Display the first few rows to understand the structure
print("TSLA data columns:", tsla_data.columns)
print("AAPL data columns:", aapl_data.columns)

# Function to preprocess stock data
def preprocess_stock_data(stock_data):
    # Convert the date column to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')
    
    # Sort by date (just in case)
    stock_data = stock_data.sort_values('Date')
    
    # Handle missing values - forward fill for missing data
    stock_data = stock_data.fillna(method='ffill')
    
    # Add new features (e.g., moving averages)
    stock_data['MA5'] = stock_data['Close'].rolling(window=5).mean()  # 5-day moving average
    stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()  # 20-day moving average
    
    # Calculate the daily price change and percentage change
    stock_data['Price Change'] = stock_data['Close'].diff()
    stock_data['Pct Change'] = stock_data['Close'].pct_change() * 100
    
    # Fill NaN values in new features to avoid dropping rows
    stock_data['MA5'] = stock_data['MA5'].fillna(stock_data['Close'])  # Fill with close prices
    stock_data['MA20'] = stock_data['MA20'].fillna(stock_data['Close'])
    stock_data['Price Change'] = stock_data['Price Change'].fillna(0)  # Fill with 0 for no change
    stock_data['Pct Change'] = stock_data['Pct Change'].fillna(0)  # Fill with 0% for no change
    
    return stock_data

# Preprocess both TSLA and AAPL data
tsla_data = preprocess_stock_data(tsla_data)
aapl_data = preprocess_stock_data(aapl_data)

# Check the preprocessed data
print("TSLA Data after preprocessing:\n", tsla_data.head())
print("AAPL Data after preprocessing:\n", aapl_data.head())

# Save the cleaned data to new CSV files
tsla_data.to_csv('cleaned_TSLA_stock_data.csv', index=False)
aapl_data.to_csv('cleaned_AAPL_stock_data.csv', index=False)
