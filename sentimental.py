import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import nltk
nltk.download('vader_lexicon')

# Read data
twitter_data = pd.read_csv('cleaned_twitter_data.csv')
reddit_data = pd.read_csv('cleaned_reddit_data.csv')
aapl_stock_data = pd.read_csv('cleaned_AAPL_stock_data.csv')
tsla_stock_data = pd.read_csv('cleaned_TSLA_stock_data.csv')

# Check the columns in the data to ensure we're referencing the correct ones
print(f"Twitter columns: {twitter_data.columns}")
print(f"Reddit columns: {reddit_data.columns}")
print(f"AAPL stock data columns: {aapl_stock_data.columns}")
print(f"TSLA stock data columns: {tsla_stock_data.columns}")

# 2. **Sentiment Analysis with VADER**
sia = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis on the 'cleaned_text' column of Twitter data
twitter_data['vader_sentiment_scores'] = twitter_data['cleaned_text'].apply(lambda x: sia.polarity_scores(str(x)))
twitter_data['vader_sentiment'] = twitter_data['vader_sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))

# Apply VADER sentiment analysis on Reddit data
reddit_data['vader_sentiment_scores'] = reddit_data['cleaned_text'].apply(lambda x: sia.polarity_scores(str(x)))
reddit_data['vader_sentiment'] = reddit_data['vader_sentiment_scores'].apply(lambda x: 'positive' if x['compound'] > 0 else ('negative' if x['compound'] < 0 else 'neutral'))

# 3. **Sentiment Analysis with TextBlob**
# Apply TextBlob sentiment analysis on the 'cleaned_text' column of Twitter data
twitter_data['textblob_polarity'] = twitter_data['cleaned_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
twitter_data['textblob_sentiment'] = twitter_data['textblob_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Apply TextBlob sentiment analysis on Reddit data
reddit_data['textblob_polarity'] = reddit_data['cleaned_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
reddit_data['textblob_sentiment'] = reddit_data['textblob_polarity'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# 4. **Sentiment Analysis with BERT**
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize Twitter and Reddit text data for BERT
def bert_tokenize(text_data):
    return tokenizer(text_data.tolist(), padding=True, truncation=True, return_tensors='pt')

twitter_inputs = bert_tokenize(twitter_data['cleaned_text'])
reddit_inputs = bert_tokenize(reddit_data['cleaned_text'])

# Dataset class for BERT
class SentimentDataset(Dataset):
    def __init__(self, inputs, labels=None):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

# Get BERT predictions (positive: 1, negative: 0)
def get_bert_predictions(inputs):
    dataset = SentimentDataset(inputs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            logits = outputs.logits
            predictions.extend(logits.argmax(dim=-1).cpu().numpy())

    return predictions

twitter_bert_preds = get_bert_predictions(twitter_inputs)
reddit_bert_preds = get_bert_predictions(reddit_inputs)

# Add BERT sentiment predictions to Twitter and Reddit data
twitter_data['bert_sentiment'] = ['positive' if pred == 1 else 'negative' for pred in twitter_bert_preds]
reddit_data['bert_sentiment'] = ['positive' if pred == 1 else 'negative' for pred in reddit_bert_preds]

# 5. **Merge Sentiment with Stock Data**
merged_twitter_data = pd.merge(twitter_data, aapl_stock_data, on='Date', how='inner')
merged_reddit_data = pd.merge(reddit_data, tsla_stock_data, on='Date', how='inner')

# 6. **Feature Engineering for Stock Data**
def calculate_technical_indicators(stock_data):
    # Check if the 'close' column exists and modify it if necessary
    if 'close' not in stock_data.columns:
        print("The 'close' column is missing from the stock data. Check the column names.")
        return stock_data

    stock_data['SMA_50'] = stock_data['close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['close'].rolling(window=200).mean()
    stock_data['RSI'] = 100 - (100 / (1 + (stock_data['close'].pct_change().rolling(window=14).mean())))
    return stock_data

merged_twitter_data = calculate_technical_indicators(merged_twitter_data)
merged_reddit_data = calculate_technical_indicators(merged_reddit_data)

# 7. **Visualization**
sentiment_counts_twitter = twitter_data['bert_sentiment'].value_counts()
sentiment_counts_reddit = reddit_data['bert_sentiment'].value_counts()

# Plot sentiment for Twitter
sentiment_counts_twitter.plot(kind='bar', title='Twitter Sentiment Distribution', color=['green', 'red'])
plt.show()

# Plot sentiment for Reddit
sentiment_counts_reddit.plot(kind='bar', title='Reddit Sentiment Distribution', color=['green', 'red'])
plt.show()
