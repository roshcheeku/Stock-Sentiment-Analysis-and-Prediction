import tweepy
import csv

# Define your credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAANhqwQEAAAAAly8DSDMhQfBinZZu%2BkSamVWrFBs%3DosChdh2eouQAtAiaFJfk49R2tsRRHondwXTz3p5G4Hibe4tv3F"  # Replace with your Twitter API v2 Bearer Token

# Authenticate using Tweepy Client for Twitter API v2
client = tweepy.Client(bearer_token=bearer_token)

# Define your search parameters
search_query = 'AAPL -is:retweet lang:en'  # Filter out retweets and set language to English
tweet_count = 100

# Fetch tweets using Twitter API v2
response = client.search_recent_tweets(
    query=search_query,
    max_results=tweet_count,
    tweet_fields=['created_at', 'author_id']
)

# Open a CSV file to save the tweets
csv_file = open('aapl_tweets.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Tweet', 'Author ID', 'Date'])

# Process the tweets and save them to the CSV file
if response.data:
    for tweet in response.data:
        tweet_text = tweet.text
        author_id = tweet.author_id
        date = tweet.created_at
        csv_writer.writerow([tweet_text, author_id, date])

# Close the CSV file
csv_file.close()

print("Tweets saved to aapl_tweets.csv")
