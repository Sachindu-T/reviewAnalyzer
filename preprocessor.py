import json
import pandas as pd
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

with open("customer_reviews.json", encoding="utf-8") as f:
    raw = json.load(f)

rows = [{"phone": phone, "review": review} for phone, reviews in raw.items() for review in reviews]
df = pd.DataFrame(rows)
df["phone"] = df["phone"].str.replace(r"\s*review$", "", regex=True).str.strip()

# Clean: lowercase, remove emojis, remove punctuation
df["review"] = df["review"].str.lower()
df["review"] = df["review"].apply(lambda t: emoji.replace_emoji(str(t), replace=""))
df["review"] = df["review"].str.replace(r"[^\w\s]", "", regex=True)
df.to_csv("cleaned_reviews.csv", index=False)

# Preprocess: remove stopwords, lemmatize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(review):
    words = word_tokenize(str(review))
    return [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

df["review"] = df["review"].apply(preprocess)
df.to_csv("preprocessed_reviews.csv", index=False)
