import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

# ============================
# **1. Scrape Financial News**
# ============================
def scrape_financial_news():
    print("Scraping Financial News from Yahoo Finance...")
    url = "https://finance.yahoo.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    headlines = soup.find_all('h3', class_="Mb(5px)")
    print("Latest News:")
    for headline in headlines[:10]:  # Top 10 headlines
        print("- ", headline.text)
    print()

# ============================
# **2. Fetch Stock Market Data**
# ============================
def fetch_stock_data(symbol):
    print(f"Fetching stock data for {symbol}...")
    stock = yf.Ticker(symbol)
    data = stock.history(period="1mo")  # Fetch 1 month of data
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # SMA
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()  # EMA
    
    # RSI Calculation
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['20_SMA'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['20_SMA'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower_Band'] = data['20_SMA'] - 2 * data['Close'].rolling(window=20).std()
    
    print("Sample Stock Data with Indicators:")
    print(data[['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band']].tail())
    return data

# ============================
# **3. Fetch Financial News with API**
# ============================
def fetch_news_api():
    print("Fetching News using NewsAPI...")
    api_key = "0fca20cf5899402c9306dca0b6b049f7"  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q=stock%20market&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    
    print("Top News Articles:")
    for article in news_data['articles'][:5]:  # Display top 5 articles
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']['name']}")
        print(f"URL: {article['url']}\n")
    print()

# ============================
# **Main Function**
# ============================
def main():
    # Step 1: Scrape Financial News
    scrape_financial_news()
    
    # Step 2: Fetch Stock Market Data
    symbol = input("Enter stock ticker symbol (e.g., AAPL, TSLA): ").upper()
    stock_data = fetch_stock_data(symbol)
    
    # Save stock data to CSV
    stock_data.to_csv(f"{symbol}_stock_data.csv", index=True)
    print(f"Stock data saved to {symbol}_stock_data.csv\n")
    
    # Step 3: Fetch Financial News Using API
    fetch_news_api()

if __name__ == "__main__":
    main()
