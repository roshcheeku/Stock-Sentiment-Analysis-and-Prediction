import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Load the scraped combined stock data
all_stock_data = pd.read_csv('scraped_combined_stock_data.csv')
all_stock_data['Date'] = pd.to_datetime(all_stock_data['Date'])

# Ensure 'Close' is numeric
all_stock_data['Close'] = pd.to_numeric(all_stock_data['Close'], errors='coerce')

# Drop rows with NaN values in 'Close'
all_stock_data = all_stock_data.dropna(subset=['Close'])

# Check the dataset
print("Stock Data Shape:", all_stock_data.shape)
print(all_stock_data.head())

# Summary statistics and visualization
summary_stats = all_stock_data.groupby('Stock_Type')['Close'].describe()
print(summary_stats)

plt.figure(figsize=(12, 6))
for stock in all_stock_data['Stock_Type'].unique():
    stock_prices = all_stock_data[all_stock_data['Stock_Type'] == stock]
    plt.plot(stock_prices['Date'], stock_prices['Close'], label=stock)
plt.title('Stock Closing Price Trends (All Stocks)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Volatility analysis
all_stock_data['Daily_Return'] = all_stock_data.groupby('Stock_Type')['Close'].pct_change()

plt.figure(figsize=(12, 6))
sns.boxplot(data=all_stock_data, x='Stock_Type', y='Daily_Return', palette='coolwarm')
plt.title('Stock Price Volatility (Daily Returns)')
plt.ylabel('Daily Return')
plt.xlabel('Stock Type')
plt.show()

# Sequence creation for LSTM
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# LSTM Preparation Function
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

# Train and save LSTM model for each stock
for stock in all_stock_data['Stock_Type'].unique():
    try:
        print(f"Processing {stock}...")
        X, y, scaler = preprocess_for_lstm(all_stock_data, stock, time_steps=60)
        
        # Build LSTM model
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
    except ValueError as e:
        print(e)

# Predict for a specific stock (e.g., 'AAPL')
stock_to_predict = 'AAPL'
try:
    stock_prices = all_stock_data[all_stock_data['Stock_Type'] == stock_to_predict]['Close'].values
    if len(stock_prices) >= 60:
        # Use the scaler from preprocessing
        _, _, scaler = preprocess_for_lstm(all_stock_data, stock_to_predict, time_steps=60)
        last_60_prices = stock_prices[-60:].reshape(-1, 1)
        last_60_scaled = scaler.transform(last_60_prices).reshape(1, -1, 1)
        
        # Load the saved model
        model = load_model(f'lstm_{stock_to_predict}_model.h5')
        
        predicted_price_scaled = model.predict(last_60_scaled)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        
        print(f"Predicted Next Day Price for {stock_to_predict}: {predicted_price[0][0]}")
    else:
        print(f"Insufficient data for prediction: {len(stock_prices)} rows available, 60 required.")
except NameError as e:
    print(f"Error during prediction: {e}")
