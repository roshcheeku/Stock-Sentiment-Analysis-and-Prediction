import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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

# Remove timezone information from 'Date' columns for merging
all_stock_data['Date'] = all_stock_data['Date'].dt.tz_localize(None)
reddit_sentiment['Date'] = reddit_sentiment['Date'].dt.tz_localize(None)
twitter_sentiment['Date'] = twitter_sentiment['Date'].dt.tz_localize(None)

# Merge Reddit and Twitter sentiment with stock data
all_stock_data = pd.merge(all_stock_data, reddit_sentiment, on='Date', how='left')
all_stock_data = pd.merge(all_stock_data, twitter_sentiment, on='Date', how='left')

# Fill missing sentiment scores with neutral sentiment (e.g., 0)
all_stock_data['reddit_sentiment'] = all_stock_data['reddit_sentiment'].fillna(0)
all_stock_data['twitter_sentiment'] = all_stock_data['twitter_sentiment'].fillna(0)

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
        X = X.reshape(X.shape[0], X.shape[1], features_scaled.shape[1])
        return X, y, scaler
    else:
        raise ValueError(f"{stock_type}: Insufficient data: {len(features_scaled)} rows available, {time_steps} required.")

# Function to predict and evaluate
def predict_and_evaluate(stock_to_predict, scaler, time_steps=60):
    stock_prices = all_stock_data[all_stock_data['Stock_Type'] == stock_to_predict][['Close', 'reddit_sentiment', 'twitter_sentiment']].values
    if len(stock_prices) >= time_steps:
        # Prepare data for prediction
        last_60_features = stock_prices[-time_steps:]
        
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
        
        # Real data for MSE and other metrics
        real_data = all_stock_data[all_stock_data['Stock_Type'] == stock_to_predict].iloc[-1]['Close']
        
        # Evaluate different metrics
        mse = mean_squared_error([real_data], [predicted_price[0][0]])  # Wrap both in lists
        mae = mean_absolute_error([real_data], [predicted_price[0][0]])  # Wrap both in lists
        rmse = np.sqrt(mse)
        
        # Adjusted R-squared for single prediction
        def calculate_r2(real_data, predicted_data):
            if len(real_data) > 1:
                return r2_score(real_data, predicted_data)
            else:
                return np.nan  # Return NaN if R² cannot be calculated

        r2 = calculate_r2([real_data], [predicted_price[0][0]])

        # Plot the predicted vs. actual prices
        plt.figure(figsize=(12, 6))
        plt.plot([real_data], label='Real Price', color='blue', marker='o')
        plt.plot([predicted_price[0][0]], label='Predicted Price', color='red', marker='x')
        plt.title(f"{stock_to_predict} - Predicted vs Actual Price")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the performance metrics
        metrics = ['MSE', 'MAE', 'RMSE', 'R²']
        values = [mse, mae, rmse, r2]
        
        plt.figure(figsize=(8, 6))
        plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        plt.title(f"{stock_to_predict} - Performance Metrics")
        plt.ylabel('Values')
        plt.show()

        print(f"Predicted Next Day Price for {stock_to_predict}: {predicted_price[0][0]}")
        print(f"Mean Squared Error (MSE) for {stock_to_predict}: {mse}")
        print(f"Mean Absolute Error (MAE) for {stock_to_predict}: {mae}")
        print(f"Root Mean Squared Error (RMSE) for {stock_to_predict}: {rmse}")
        print(f"R-squared (R²) for {stock_to_predict}: {r2}")
    else:
        print(f"Insufficient data for prediction: {len(stock_prices)} rows available, {time_steps} required.")

# Allow user to input which stock to predict
stock_to_predict = input("Enter the stock symbol to predict (e.g., 'AAPL'): ")

try:
    # Get the scaler from preprocessing
    _, _, scaler = preprocess_for_lstm_multifeature(all_stock_data, stock_to_predict)
    
    # Perform prediction and evaluation
    predict_and_evaluate(stock_to_predict, scaler)
    
except ValueError as e:
    print(e)
