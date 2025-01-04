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

# Load Reddit data
reddit_data = pd.read_csv('cleaned_reddit_data.csv')
reddit_data['Date'] = pd.to_datetime(reddit_data['Date'])  # Ensure datetime format

# Aggregate sentiment scores for Reddit
reddit_sentiment = reddit_data.groupby('Date')['sentiment'].mean().reset_index()
reddit_sentiment.rename(columns={'sentiment': 'reddit_sentiment'}, inplace=True)

# Load Twitter data
twitter_data = pd.read_csv('cleaned_twitter_data.csv')
twitter_data['Date'] = pd.to_datetime(twitter_data['Date'])  # Ensure datetime format

# Aggregate sentiment scores for Twitter
twitter_sentiment = twitter_data.groupby('Date')['sentiment'].mean().reset_index()
twitter_sentiment.rename(columns={'sentiment': 'twitter_sentiment'}, inplace=True)

# Ensure 'Date' columns are in the same format (remove timezone if present)
all_stock_data['Date'] = all_stock_data['Date'].dt.tz_localize(None)  # Remove timezone if present
twitter_sentiment['Date'] = twitter_sentiment['Date'].dt.tz_localize(None)  # Remove timezone if present
reddit_sentiment['Date'] = reddit_sentiment['Date'].dt.tz_localize(None)  # Remove timezone if present

# Merge Reddit sentiment with stock data
all_stock_data = pd.merge(all_stock_data, reddit_sentiment, on='Date', how='left')

# Merge Twitter sentiment with stock data
all_stock_data = pd.merge(all_stock_data, twitter_sentiment, on='Date', how='left')

# Fill missing sentiment scores with neutral sentiment (e.g., 0)
all_stock_data['reddit_sentiment'].fillna(0, inplace=True)
all_stock_data['twitter_sentiment'].fillna(0, inplace=True)

# Sequence creation for LSTM with multiple features
def create_sequences_multifeature(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])  # Assuming 'Close' is the target
    return np.array(X), np.array(y)

# Preprocessing function for LSTM with sentiment data
def preprocess_for_lstm_multifeature(stock_data, stock_type, time_steps=60):
    stock_data = stock_data[stock_data['Stock_Type'] == stock_type]
    features = stock_data[['Close', 'reddit_sentiment', 'twitter_sentiment']].values
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    if len(features_scaled) > time_steps:
        X, y = create_sequences_multifeature(features_scaled, time_steps)
        print(f"{stock_type}: Shape of X: {X.shape}, Shape of y: {y.shape}")
        X = X.reshape(X.shape[0], X.shape[1], features_scaled.shape[1])
        return X, y, scaler
    else:
        raise ValueError(f"{stock_type}: Insufficient data: {len(features_scaled)} rows available, {time_steps} required.")

# Build and train the LSTM model for each stock
for stock in all_stock_data['Stock_Type'].unique():
    try:
        print(f"Processing {stock}...")
        X, y, scaler = preprocess_for_lstm_multifeature(all_stock_data, stock, time_steps=60)
        
        # Build the LSTM model
        model = Sequential([ 
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X, y, batch_size=32, epochs=50, verbose=1)
        
        # Save the model
        model_filename = f'lstm_{stock}_model_with_sentiment.h5'
        model.save(model_filename)
        print(f"LSTM model for {stock} with sentiment saved as '{model_filename}'")
    except ValueError as e:
        print(e)

# Predict next day price for a specific stock (e.g., 'AAPL')
stock_to_predict = 'AAPL'
try:
    stock_prices = all_stock_data[all_stock_data['Stock_Type'] == stock_to_predict][['Close', 'reddit_sentiment', 'twitter_sentiment']].values
    if len(stock_prices) >= 60:
        # Use the scaler from preprocessing
        last_60_features = stock_prices[-60:]
        
        # Scale the features
        last_60_scaled = scaler.transform(last_60_features).reshape(1, -1, last_60_features.shape[1])
        
        # Load the saved model
        model = load_model(f'lstm_{stock_to_predict}_model_with_sentiment.h5')
        
        # Predict next day price
        predicted_price_scaled = model.predict(last_60_scaled)
        
        # Reshape the prediction to match the number of features
        predicted_price_scaled_reshaped = np.concatenate((predicted_price_scaled, np.zeros((predicted_price_scaled.shape[0], 2))), axis=1)
        
        # Inverse transform to get the predicted 'Close' price
        predicted_price = scaler.inverse_transform(predicted_price_scaled_reshaped)
        
        print(f"Predicted Next Day Price for {stock_to_predict}: {predicted_price[0][0]}")
    else:
        print(f"Insufficient data for prediction: {len(stock_prices)} rows available, 60 required.")
except Exception as e:
    print(f"Error during prediction: {e}")
