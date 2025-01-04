# Stock Sentiment Analysis and Prediction

## Overview
This project is designed to predict stock market trends using both stock data (e.g., from AAPL, AMZN, GOOG, MSFT, TSLA) and sentiment analysis based on social media platforms, including Twitter and Reddit. It leverages machine learning models, particularly Long Short-Term Memory (LSTM) networks, to generate predictions about stock prices. The project integrates stock data with sentiment derived from social media, providing valuable insights for stock market forecasting.

## Project Structure
The project is structured as follows:

### 1. Stock Data
Contains historical stock data for various companies (AAPL, AMZN, MSFT, etc.), stored in .csv format. These files serve as the primary dataset for training and testing the machine learning models.

**Files:**
- `AAPL_stock_data.csv`, `AMZN_stock_data.csv`, `GOOG_stock_data.csv`, `MSFT_stock_data.csv`, `TSLA_stock_data.csv`: Raw stock data for different companies.
- `cleaned_*_stock_data.csv`: Preprocessed stock data files ready for model training.
- `merged_stock_data.csv`: Consolidated dataset from all stock data sources.

### 2. Social Media Data
Data scraped from social media platforms like Twitter and Reddit, containing stock-related posts, comments, and discussions.

**Files:**
- `cleaned_reddit_data.csv`, `cleaned_twitter_data.csv`: Cleaned data from Reddit and Twitter, respectively, used for sentiment analysis.
- `sentimental.py`, `sentimentalwithstock.py`: Python scripts that perform sentiment analysis on the social media data.
- `merged_social_media_data.csv`: Combined social media data from both platforms for further analysis.

### 3. Machine Learning Models
LSTM Models: The LSTM models are used to predict stock movements by learning patterns from the historical stock data and sentiment data.

**Files:**
- `lstm_*_model.h5`: Pre-trained LSTM models for various stocks (AAPL, AMZN, GOOG, MSFT, TSLA).
- `lstm_*_model_with_sentiment.h5`: LSTM models that also incorporate sentiment analysis data for enhanced prediction accuracy.

### 4. Preprocessing and Feature Engineering
Scripts to preprocess the raw stock data and social media data, as well as generate features necessary for training machine learning models.

**Files:**
- `datapreprocess.py`, `filter.py`, `feature.py`: Scripts for data cleaning and feature extraction.
- `addstockdata.py`: Script for adding new stock data to the existing dataset.

### 5. Prediction and Evaluation
Scripts for making predictions using the trained models, including evaluating model performance.

**Files:**
- `predict.py`: Main script for making stock price predictions.
- `scrapescrape.py`: Script to combine data scraped from various sources.
- `errors.docx`: Document containing information about known errors or issues during model development.

### 6. Utilities
Miscellaneous helper scripts for tasks such as data cleaning and web scraping.

**Files:**
- `webscrape.py`: Web scraping script to gather stock-related data from the web.
- `twee.py.txt`, `tweet.py`: Scripts related to Twitter data collection.

## Why This Project?
This project was developed to address the growing interest in using machine learning and sentiment analysis for stock market predictions. By combining both traditional stock market data and social media sentiment, the goal is to improve the accuracy of stock trend predictions. Social media has become a powerful source of information that can influence stock prices, and this project aims to capitalize on that data to enhance stock market forecasting.

## What Does the Project Do?
1. **Data Collection**: The project collects stock market data for various companies (e.g., AAPL, AMZN, GOOG, TSLA) and social media data from platforms like Twitter and Reddit.
2. **Data Preprocessing**: The raw data is cleaned and processed to remove any noise and inconsistencies.
3. **Sentiment Analysis**: Sentiment analysis is performed on social media posts to determine whether the overall sentiment about a particular stock is positive, negative, or neutral.
4. **Model Training**: The processed data is used to train machine learning models, particularly LSTM networks, which predict stock price movements.
5. **Prediction**: The trained models are used to generate predictions about future stock prices based on the latest data.

## Getting Started

### Prerequisites
- Python 3.6 or higher
- Required libraries (install using `pip install -r requirements.txt`):
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `keras`
  - `tensorflow`
  - `tweepy`
  - `vaderSentiment`
  - `matplotlib`

### Running the Project
1. **Data Collection**: Use the scripts in `webscrape.py` and `reddit.py` to scrape data.
2. **Preprocess Data**: Run `datapreprocess.py` to clean the collected data.
3. **Train the Model**: Use the cleaned and preprocessed data to train the LSTM models (run `lstm.py`).
4. **Make Predictions**: Use `predict.py` to generate stock price predictions using the trained models.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
