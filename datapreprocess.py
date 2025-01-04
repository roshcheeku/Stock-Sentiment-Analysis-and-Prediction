import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize SentimentIntensityAnalyzer and Lemmatizer
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Load your datasets (adjust path if needed)
twitter_aapl_tweets = pd.read_csv("aapl_tweets.csv")  # Example CSV file for Twitter data
reddit_aapl_posts = pd.read_csv("reddit_aapl_posts.csv")  # Example CSV file for Reddit data

# Print columns to verify correct data loading
print("Twitter columns:", twitter_aapl_tweets.columns)
print("Reddit columns:", reddit_aapl_posts.columns)

# Function to clean the text
def clean_text(text):
    # Remove URLs, mentions, hashtags, special characters, and lower the text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert text to lowercase
    text = word_tokenize(text)  # Tokenize the text

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    
    # Lemmatize words
    text = [lemmatizer.lemmatize(word) for word in text]

    # Join the words back into a string
    return ' '.join(text)

# Function to get sentiment score using Vader
def get_sentiment(text):
    return sia.polarity_scores(text)['compound']

# Apply cleaning and sentiment analysis to Twitter data
twitter_aapl_tweets['cleaned_text'] = twitter_aapl_tweets['Tweet'].apply(clean_text)
twitter_aapl_tweets['sentiment'] = twitter_aapl_tweets['cleaned_text'].apply(get_sentiment)

# Print the cleaned Twitter data
print("\nCleaned Twitter Data:")
print(twitter_aapl_tweets[['Tweet', 'cleaned_text', 'sentiment']].head())

# Apply cleaning and sentiment analysis to Reddit data
# Assuming 'Title' contains the post text (adjust to 'Comments' if needed)
reddit_aapl_posts['cleaned_text'] = reddit_aapl_posts['Title'].apply(clean_text)  # Or 'Comments'
reddit_aapl_posts['sentiment'] = reddit_aapl_posts['cleaned_text'].apply(get_sentiment)

# Print the cleaned Reddit data
print("\nCleaned Reddit Data:")
print(reddit_aapl_posts[['Title', 'cleaned_text', 'sentiment']].head())

# Save cleaned data to CSV
twitter_aapl_tweets.to_csv("cleaned_twitter_data.csv", index=False)
reddit_aapl_posts.to_csv("cleaned_reddit_data.csv", index=False)

print("\nData cleaned and saved to CSV.")
