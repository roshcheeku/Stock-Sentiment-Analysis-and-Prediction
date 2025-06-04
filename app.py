from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
import tweepy
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
import io
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Supported stocks
SUPPORTED_STOCKS = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA']

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.sentiment_models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load pre-trained LSTM models"""
        for stock in SUPPORTED_STOCKS:
            try:
                # Load regular LSTM model
                model_path = f'lstm_{stock.lower()}_model.h5'
                if os.path.exists(model_path):
                    self.models[stock] = load_model(model_path)
                
                # Load sentiment-enhanced LSTM model
                sentiment_model_path = f'lstm_{stock.lower()}_model_with_sentiment.h5'
                if os.path.exists(sentiment_model_path):
                    self.sentiment_models[stock] = load_model(sentiment_model_path)
                
                # Initialize scaler for each stock
                self.scalers[stock] = MinMaxScaler(feature_range=(0, 1))
                
            except Exception as e:
                print(f"Error loading model for {stock}: {e}")
    
    def get_stock_data(self, symbol, period='1y'):
        """Fetch real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def preprocess_data(self, data, lookback=60):
        """Preprocess stock data for LSTM input"""
        if data is None or len(data) < lookback:
            return None, None
        
        # Use closing prices
        prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scalers[data.index.name].fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def predict_stock_price(self, symbol, use_sentiment=False):
        """Make stock price predictions"""
        try:
            # Get stock data
            stock_data = self.get_stock_data(symbol)
            if stock_data is None:
                return None, "Error fetching stock data"
            
            # Preprocess data
            X, y = self.preprocess_data(stock_data)
            if X is None:
                return None, "Insufficient data for prediction"
            
            # Select model
            model = self.sentiment_models.get(symbol) if use_sentiment else self.models.get(symbol)
            if model is None:
                return None, f"Model not available for {symbol}"
            
            # Make prediction
            last_sequence = X[-1].reshape(1, -1, 1)
            prediction = model.predict(last_sequence)
            
            # Inverse transform prediction
            prediction_price = self.scalers[symbol].inverse_transform(prediction.reshape(-1, 1))[0][0]
            current_price = stock_data['Close'].iloc[-1]
            
            return {
                'current_price': current_price,
                'predicted_price': prediction_price,
                'change': prediction_price - current_price,
                'change_percent': ((prediction_price - current_price) / current_price) * 100,
                'stock_data': stock_data
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text):
        """Analyze sentiment of text"""
        try:
            scores = self.analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None
    
    def get_sentiment_label(self, compound_score):
        """Convert compound score to sentiment label"""
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

# Initialize predictors
stock_predictor = StockPredictor()
sentiment_analyzer = SentimentAnalyzer()

def create_stock_chart(stock_data, symbol):
    """Create stock price chart"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['Close'], label='Close Price', linewidth=2)
        plt.plot(stock_data.index, stock_data['Open'], label='Open Price', alpha=0.7)
        plt.title(f'{symbol} Stock Price Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html', stocks=SUPPORTED_STOCKS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle stock prediction requests"""
    try:
        symbol = request.form.get('symbol', '').upper()
        use_sentiment = request.form.get('use_sentiment') == 'on'
        
        if symbol not in SUPPORTED_STOCKS:
            flash(f'Stock {symbol} is not supported. Please choose from: {", ".join(SUPPORTED_STOCKS)}', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        result, error = stock_predictor.predict_stock_price(symbol, use_sentiment)
        
        if error:
            flash(f'Prediction error: {error}', 'error')
            return redirect(url_for('index'))
        
        # Create chart
        chart_url = create_stock_chart(result['stock_data'], symbol)
        
        return render_template('prediction.html', 
                             symbol=symbol,
                             result=result,
                             chart_url=chart_url,
                             use_sentiment=use_sentiment)
        
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    """Sentiment analysis page"""
    if request.method == 'POST':
        text = request.form.get('text', '')
        
        if not text.strip():
            flash('Please enter some text to analyze', 'error')
            return render_template('sentiment.html')
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer.analyze_text(text)
        
        if sentiment_result:
            sentiment_label = sentiment_analyzer.get_sentiment_label(sentiment_result['compound'])
            return render_template('sentiment.html', 
                                 text=text,
                                 sentiment_result=sentiment_result,
                                 sentiment_label=sentiment_label)
        else:
            flash('Error analyzing sentiment', 'error')
    
    return render_template('sentiment.html')

@app.route('/api/stock/<symbol>')
def api_stock_data(symbol):
    """API endpoint for stock data"""
    try:
        symbol = symbol.upper()
        if symbol not in SUPPORTED_STOCKS:
            return jsonify({'error': f'Stock {symbol} not supported'}), 400
        
        stock_data = stock_predictor.get_stock_data(symbol, period='1mo')
        if stock_data is None:
            return jsonify({'error': 'Could not fetch stock data'}), 500
        
        # Convert to JSON-serializable format
        data = {
            'symbol': symbol,
            'current_price': float(stock_data['Close'].iloc[-1]),
            'open_price': float(stock_data['Open'].iloc[-1]),
            'high_price': float(stock_data['High'].iloc[-1]),
            'low_price': float(stock_data['Low'].iloc[-1]),
            'volume': int(stock_data['Volume'].iloc[-1]),
            'last_updated': stock_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """API endpoint for sentiment analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Text is required'}), 400
        
        sentiment_result = sentiment_analyzer.analyze_text(text)
        if sentiment_result:
            sentiment_label = sentiment_analyzer.get_sentiment_label(sentiment_result['compound'])
            return jsonify({
                'text': text,
                'sentiment': sentiment_result,
                'label': sentiment_label
            })
        else:
            return jsonify({'error': 'Error analyzing sentiment'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Multi-stock dashboard"""
    stock_data = {}
    
    for symbol in SUPPORTED_STOCKS:
        try:
            data = stock_predictor.get_stock_data(symbol, period='5d')
            if data is not None:
                current_price = float(data['Close'].iloc[-1])
                prev_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
                change = current_price - prev_price
                change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
                
                stock_data[symbol] = {
                    'current_price': current_price,
                    'change': change,
                    'change_percent': change_percent
                }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue
    
    return render_template('dashboard.html', stock_data=stock_data)

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
