import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('preprocessed_reviews.csv')
df = pd.DataFrame(data)

cleaned = pd.read_csv('cleaned_reviews.csv')

sia  = SentimentIntensityAnalyzer()
def get_sentiment(review):
    score  = sia.polarity_scores(str(review))['compound']
    return 1 if score > 0 else 0 

df['sentiment'] = cleaned['review'].apply(get_sentiment)
df.to_csv('labeled_preprocessed_reviews.csv', index=False)

x_train, x_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment'])

tfidf = TfidfVectorizer()
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

lr = LogisticRegression()
lr.fit(x_train_tfidf, y_train)

y_pred = lr.predict(x_test_tfidf)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

