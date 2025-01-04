import praw
import csv
import time

# Define your Reddit API credentials
reddit = praw.Reddit(
    client_id="nc0EAIS3qzu0oWRKqTeYbQ",
    client_secret="L8UDtovV2KeffNFg-ipY7tN01WfPrQ",
   user_agent = "Python:stock:v1.0 (by u/Vast-Bookkeeper-7703)"
)
# Define your search parameters
subreddit_name = "stocks"  # Subreddit to search in
search_query = "AAPL"      # Keyword to search for
post_count = 100           # Number of posts to fetch

# Search for posts
subreddit = reddit.subreddit(subreddit_name)
posts = subreddit.search(query=search_query, limit=post_count)

# Open a CSV file to save the posts
csv_file = open('reddit_aapl_posts.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Title', 'Author', 'Date', 'Score', 'Comments'])

# Process the posts and save them to the CSV file
for post in posts:
    title = post.title
    author = post.author.name if post.author else "N/A"
    created_date = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(post.created_utc))
    score = post.score
    comments = post.num_comments
    csv_writer.writerow([title, author, created_date, score, comments])

# Close the CSV file
csv_file.close()

print("Reddit posts saved to reddit_aapl_posts.csv")
