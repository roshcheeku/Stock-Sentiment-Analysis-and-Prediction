import http.client
import json
import pandas as pd
import random
from datetime import datetime

def fetch_stock_data():
    conn = http.client.HTTPSConnection("yahoo-finance15.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "# your Yahoo Finance API ",
        'x-rapidapi-host': "yahoo-finance15.p.rapidapi.com"
    }

    conn.request("GET", "/api/v1/markets/stock/quotes?ticker=AAPL%2CMSFT%2C%5ESPX%2C%5ENYA%2CGAZP.ME%2CSIBN.ME%2CGEECEE.NS", headers=headers)

    res = conn.getresponse()
    data = res.read()

    # Convert the response to JSON
    stock_data = json.loads(data.decode("utf-8"))

    # Debug: Print the full API response
    print("Full API Response:", json.dumps(stock_data, indent=4))

    # If the response body contains the data (adjust according to actual structure)
    if "body" in stock_data:
        # Assuming 'body' contains the stock data, modify this according to actual response structure
        stock_body = stock_data["body"]

        # Check if it's a list of stocks or some other structure
        if isinstance(stock_body, list):
            stock_df = pd.json_normalize(stock_body)
            
            # Convert 'timestamp' to datetime, if present
            if 'timestamp' in stock_df.columns:
                stock_df['date'] = pd.to_datetime(stock_df['timestamp'], unit='s')

            # Save the stock data to CSV
            stock_df.to_csv('stock_api_response.csv', index=False)
            print(f"API Response saved to 'stock_api_response.csv'")
            return stock_df
        else:
            print("Error: Stock data in body is not a list as expected.")
    else:
        print("Error: 'body' key is missing from the response.")
        if "error" in stock_data:
            print("Error details:", stock_data["error"])
    
    return None


# ------------------- Loading and Filtering Reddit Data -------------------
def fetch_reddit_data():
    # Assuming you have the Reddit data saved in a CSV file ('cleaned_reddit_data.csv')
    reddit_data = pd.read_csv('cleaned_reddit_data.csv')

    # Convert 'Date' to datetime
    reddit_data['Date'] = pd.to_datetime(reddit_data['Date'])

    # Define the date range for Reddit data (2019-04-16 to 2024-12-13)
    start_date = pd.to_datetime('2019-04-16')
    end_date = pd.to_datetime('2024-12-13')

    # Filter Reddit data based on the date range
    filtered_reddit_data = reddit_data[(reddit_data['Date'] >= start_date) & (reddit_data['Date'] <= end_date)]
    
    return filtered_reddit_data


# ------------------- Loading and Filtering Twitter Data -------------------
def fetch_twitter_data():
    # Assuming you have the Twitter data saved in a CSV file ('cleaned_twitter_data.csv')
    twitter_data = pd.read_csv('cleaned_twitter_data.csv')
    print("Twitter Data Columns:", twitter_data.columns)

    # Check if the 'Date' column exists
    if 'Date' not in twitter_data.columns:
        print("Error: 'Date' column not found in Twitter data.")
        return None

    # Convert 'Date' to datetime with timezone awareness (assuming it's in UTC)
    twitter_data['Date'] = pd.to_datetime(twitter_data['Date'], utc=True)

    # Define the time range for Twitter data (timezone-aware in UTC)
    start_time = pd.Timestamp('2024-12-30 10:21:00', tz='UTC')
    end_time = pd.Timestamp('2024-12-30 11:03:00', tz='UTC')

    # Filter Twitter data based on the time range
    filtered_twitter_data = twitter_data[(twitter_data['Date'] >= start_time) & (twitter_data['Date'] <= end_time)]
    
    return filtered_twitter_data


# ------------------- Main Function to Fetch, Combine, and Save Data -------------------
def main():
    # Fetch and save filtered stock data
    try:
        stock_data = fetch_stock_data()
        if stock_data is not None and not stock_data.empty:
            # Save the stock data to CSV ensuring it has the same columns as filtered_reddit_data.csv
            stock_data.columns = ['symbol', 'regularMarketPrice', 'regularMarketOpen', 'regularMarketPreviousClose', 'timestamp', 'date']
            stock_data.to_csv('filtered_stock_data.csv', index=False)
            print(f"Filtered Stock Data saved to 'filtered_stock_data.csv':")
            print(stock_data)
        else:
            print("No filtered stock data to save.")
    except Exception as e:
        print(f"Error fetching or saving stock data: {e}")

    # Fetch and save filtered Reddit data
    try:
        reddit_data = fetch_reddit_data()
        if not reddit_data.empty:
            reddit_data.to_csv('filtered_reddit_data.csv', index=False)
            print(f"\nFiltered Reddit Data saved to 'filtered_reddit_data.csv':")
            print(reddit_data)
        else:
            print("No filtered Reddit data to save.")
    except Exception as e:
        print(f"Error fetching or saving Reddit data: {e}")

    # Fetch and save filtered Twitter data
    try:
        twitter_data = fetch_twitter_data()
        if twitter_data is not None and not twitter_data.empty:
            twitter_data.columns = ['Tweet', 'Date', 'Other Columns']  # Ensure column names are consistent
            twitter_data.to_csv('filtered_twitter_data.csv', index=False)
            print(f"\nFiltered Twitter Data saved to 'filtered_twitter_data.csv':")
            print(twitter_data)
        else:
            print("No filtered Twitter data to save.")
    except Exception as e:
        print(f"Error fetching or saving Twitter data: {e}")

if __name__ == "__main__":
    main()
