import json
import pandas as pd
import emoji
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


with open('customer_reviews.json', encoding='utf-8') as f:
    raw = json.load(f)

rows = []
for phone, reviews in raw.items():
    for review in reviews:
        rows.append({'phone': phone, 'review': review})

df = pd.DataFrame(rows)

df['phone'] = df['phone'].str.replace(r'\s*review$', '', regex=True).str.strip()

df['review'] = df['review'].str.lower()

df['review'] = df['review'].apply(lambda text: emoji.replace_emoji(str(text), replace=''))

df['review'] = df['review'].str.replace(r'[^\w\s]', '', regex=True)

df.to_csv('cleaned_reviews.csv', index=False)

stop_words = set(stopwords.words('english'))

def remove_stop_words(review):
    words = str(review).split()
    filtered_words = [word for word in words if word not in stop_words]
    return (filtered_words)

df['review'] = df['review'].apply(remove_stop_words)


stemmer = PorterStemmer()

def stem_words(words):
    return [stemmer.stem(word) for word in words]

df['review'] = df['review'].apply(stem_words)


lemmatizer = WordNetLemmatizer()
def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

df['review'] = df['review'].apply(lemmatize_words)

df.to_csv('preprocessed_reviews.csv', index=False)