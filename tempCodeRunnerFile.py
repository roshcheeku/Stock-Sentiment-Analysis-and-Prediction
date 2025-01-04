import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Load social media datasets
twitter_data = pd.read_csv('cleaned_twitter_data.csv')  # Replace with actual file paths
reddit_data = pd.read_csv('cleaned_reddit_data.csv')

# Load stock datasets
aapl_data = pd.read_csv('cleaned_AAPL_stock_data.csv')
tsla_data = pd.read_csv('cleaned_TSLA_stock_data.csv')
amzn_data = pd.read_csv('cleaned_AMZN_stock_data.csv')
goog_data = pd.read_csv('cleaned_GOOG_stock_data.csv')
msft_data = pd.read_csv('cleaned_MSFT_stock_data.csv')

# 1. Merge Social Media Data
twitter_data.rename(columns={'Author ID': 'Author'}, inplace=True)
reddit_data.rename(columns={'Author': 'Author'}, inplace=True)

# Align column names for merging
twitter_data = twitter_data[['Author', 'Date', 'cleaned_text', 'sentiment']]
reddit_data = reddit_data[['Author', 'Date', 'cleaned_text', 'sentiment']]

# Combine datasets
social_media_data = pd.concat([twitter_data, reddit_data], ignore_index=True)
social_media_data.to_csv('merged_social_media_data.csv', index=False)
print("Merged Social Media Data Shape:", social_media_data.shape)

# 2. Merge Stock Data
aapl_data['Stock_Type'] = 'AAPL'
tsla_data['Stock_Type'] = 'TSLA'
amzn_data['Stock_Type'] = 'AMZN'
goog_data['Stock_Type'] = 'GOOG'
msft_data['Stock_Type'] = 'MSFT'

# Combine all stock data
all_stock_data = pd.concat([aapl_data, tsla_data, amzn_data, goog_data, msft_data], ignore_index=True)
all_stock_data.to_csv('merged_all_stock_data.csv', index=False)
print("Merged Stock Data Shape:", all_stock_data.shape)

# Summary statistics and visualization
summary_stats = all_stock_data.groupby('Stock_Type')['Close'].describe()
print(summary_stats)

plt.figure(figsize=(12, 6))
for stock in all_stock_data['Stock_Type'].unique():
    stock_prices = all_stock_data[all_stock_data['Stock_Type'] == stock]
    plt.plot(pd.to_datetime(stock_prices['Date']), stock_prices['Close'], label=stock)
plt.title('Stock Closing Price Trends (All Stocks)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Volatility analysis
all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date'])
all_stock_data['Daily_Return'] = all_stock_data.groupby('Stock_Type')['Close'].pct_change()

plt.figure(figsize=(12, 6))
sns.boxplot(data=all_stock_data, x='Stock_Type', y='Daily_Return', palette='coolwarm')
plt.title('Stock Price Volatility (Daily Returns) (All Stocks)')
plt.ylabel('Daily Return')
plt.xlabel('Stock Type')
plt.show()

# 3. LSTM Preparation Function
def preprocess_for_lstm(stock_data, stock_type, time_steps=60):
    stock_data = stock_data[stock_data['Stock_Type'] == stock_type]
    prices = stock_data[['Close']].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    
    if len(prices_scaled) > time_steps:
        X, y = create_sequences(prices_scaled, time_steps)
        print(f"{stock_type}: Shape of X: {X.shape}, Shape of y: {y.shape}")
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X, y, scaler
    else:
        raise ValueError(f"{stock_type}: Insufficient data: {len(prices_scaled)} rows available, {time_steps} required.")

# Sequence creation for LSTM
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# 4. Train LSTM for each stock
for stock in ['AAPL', 'TSLA', 'AMZN', 'GOOG', 'MSFT']:
    print(f"Processing {stock}...")
    X, y, scaler = preprocess_for_lstm(all_stock_data, stock, time_steps=60)
    
    # Build and train LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=50, verbose=1)
    
    # Save the model
    model_filename = f'lstm_{stock}_model.h5'
    model.save(model_filename)
    print(f"LSTM model for {stock} saved as '{model_filename}'")

# 5. Predict for AAPL (example)
aapl_prices = all_stock_data[all_stock_data['Stock_Type'] == 'AAPL']['Close'].values
aapl_prices_scaled = scaler.transform(aapl_prices[-60:].reshape(-1, 1)).reshape(1, -1, 1)
predicted_price = model.predict(aapl_prices_scaled)
predicted_price = scaler.inverse_transform(predicted_price)
print(f"Predicted Next Day Price for AAPL: {predicted_price[0][0]}")
