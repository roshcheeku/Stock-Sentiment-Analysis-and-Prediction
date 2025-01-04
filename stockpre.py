import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load social media data
social_media_data = pd.read_csv('social_media_data.csv')

# Check the column names
print("Columns in social media data:", social_media_data.columns)

# Clean the column names by stripping spaces and converting to lowercase
social_media_data.columns = social_media_data.columns.str.strip().str.lower()

# Define a text cleaning function
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, mentions, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        # Join the tokens back into a string
        cleaned_text = ' '.join(filtered_tokens)
        return cleaned_text
    else:
        return ""

# Apply the text cleaning function to the appropriate column ('text')
social_media_data['cleaned_text'] = social_media_data['text'].apply(clean_text)

# Check the cleaned data
print(social_media_data.head())

# Optionally, save the cleaned data to a new CSV
social_media_data.to_csv('cleaned_social_media_data.csv', index=False)
