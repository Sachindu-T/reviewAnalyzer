import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("preprocessed_reviews.csv")
cleaned = pd.read_csv("cleaned_reviews.csv")

sia = SentimentIntensityAnalyzer()
data["sentiment"] = cleaned["review"].apply(lambda r: 1 if sia.polarity_scores(str(r))["compound"] > 0 else 0)
data.to_csv("labeled_preprocessed_reviews.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(
    data["review"], data["sentiment"], test_size=0.2, random_state=42, stratify=data["sentiment"])

tfidf = TfidfVectorizer()
lr = LogisticRegression()
lr.fit(tfidf.fit_transform(X_train), y_train)

y_pred = lr.predict(tfidf.transform(X_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
