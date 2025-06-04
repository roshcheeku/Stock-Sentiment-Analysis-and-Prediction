# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def prepare_data(stock_data, news_data):
    # Merge stock and news data
    merged_data = pd.merge(stock_data, news_data, left_index=True, right_index=True, how='left')
    merged_data['sentiment'] = merged_data['news_text'].apply(analyze_sentiment)
    merged_data['sentiment'].fillna(0, inplace=True)
    return merged_data

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user inputs
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        news_text = request.form['news_text']
        
        # Process dates
        start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Get stock data
        try:
            stock_data = get_stock_data(ticker, start_date, end_date)
            if stock_data.empty:
                raise ValueError("No data found for the given ticker and date range")
            
            # Create sample news data (in a real app, this would come from a database or API)
            news_dates = pd.date_range(start=start_date_obj, end=end_date_obj)
            news_data = pd.DataFrame({
                'date': news_dates,
                'news_text': [news_text] * len(news_dates)  # Using the same news text for all dates for demo
            })
            news_data.set_index('date', inplace=True)
            
            # Prepare data for model
            merged_data = prepare_data(stock_data, news_data)
            
            # Create features and target
            X = merged_data[['sentiment', 'Volume']].values
            y = merged_data['Close'].values
            
            # Train model
            model, X_test, y_test = train_model(X, y)
            score = model.score(X_test, y_test)
            
            # Make prediction
            current_sentiment = analyze_sentiment(news_text)
            prediction = model.predict([[current_sentiment, 1000000]])  # Using sample volume
            
            # Create plots
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            merged_data['Close'].plot(ax=ax1)
            ax1.set_title(f'{ticker} Stock Price')
            ax1.set_ylabel('Price ($)')
            plot1 = plot_to_base64(fig1)
            
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            merged_data['sentiment'].plot(ax=ax2, color='orange')
            ax2.set_title('News Sentiment Over Time')
            ax2.set_ylabel('Sentiment Score')
            plot2 = plot_to_base64(fig2)
            
            return render_template('results.html', 
                                 ticker=ticker,
                                 start_date=start_date,
                                 end_date=end_date,
                                 current_price=merged_data['Close'].iloc[-1],
                                 prediction=prediction[0],
                                 model_score=score,
                                 plot1=plot1,
                                 plot2=plot2)
            
        except Exception as e:
            error_message = f"Error processing your request: {str(e)}"
            return render_template('index.html', error=error_message)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
