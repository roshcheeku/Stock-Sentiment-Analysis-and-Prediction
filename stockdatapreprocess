import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the datasets
amzn_data = pd.read_csv('AMZN_stock_data.csv')
goog_data = pd.read_csv('GOOG_stock_data.csv')
msft_data = pd.read_csv('MSFT_stock_data.csv')

# Step 2: Inspect the data
print("AMZN Data Head:")
print(amzn_data.head())
print("GOOG Data Head:")
print(goog_data.head())
print("MSFT Data Head:")
print(msft_data.head())

# Step 3: Clean the data

# Remove duplicates
amzn_data.drop_duplicates(inplace=True)
goog_data.drop_duplicates(inplace=True)
msft_data.drop_duplicates(inplace=True)

# Handle missing values - fill with forward fill method for simplicity
amzn_data.fillna(method='ffill', inplace=True)
goog_data.fillna(method='ffill', inplace=True)
msft_data.fillna(method='ffill', inplace=True)

# Check for missing values after filling
print("AMZN Missing Values:", amzn_data.isnull().sum())
print("GOOG Missing Values:", goog_data.isnull().sum())
print("MSFT Missing Values:", msft_data.isnull().sum())

# Convert 'Date' column to datetime format
amzn_data['Date'] = pd.to_datetime(amzn_data['Date'])
goog_data['Date'] = pd.to_datetime(goog_data['Date'])
msft_data['Date'] = pd.to_datetime(msft_data['Date'])

# Step 4: Feature Engineering
# Calculate daily returns for each stock (percentage change of 'Close' column)
amzn_data['Daily_Return'] = amzn_data['Close'].pct_change()
goog_data['Daily_Return'] = goog_data['Close'].pct_change()
msft_data['Daily_Return'] = msft_data['Close'].pct_change()

# Step 5: Normalize the 'Close' prices using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

amzn_data['Close'] = scaler.fit_transform(amzn_data[['Close']])
goog_data['Close'] = scaler.fit_transform(goog_data[['Close']])
msft_data['Close'] = scaler.fit_transform(msft_data[['Close']])

# Step 6: Save cleaned data to new CSV files
amzn_data.to_csv('cleaned_AMZN_stock_data.csv', index=False)
goog_data.to_csv('cleaned_GOOG_stock_data.csv', index=False)
msft_data.to_csv('cleaned_MSFT_stock_data.csv', index=False)

print("Cleaned AMZN data saved to 'cleaned_AMZN_stock_data.csv'")
print("Cleaned GOOG data saved to 'cleaned_GOOG_stock_data.csv'")
print("Cleaned MSFT data saved to 'cleaned_MSFT_stock_data.csv'")
