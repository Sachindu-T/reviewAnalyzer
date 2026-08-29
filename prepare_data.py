import ast, os
import pandas as pd
import nltk

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sia.polarity_scores("test")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def join_tokens(tok_str):
    """Convert stringified token list like "['a','b']" to "a b"."""
    if tok_str is None:
        return ""
    s = str(tok_str).strip()
    if not s or s in ("[]", "nan"):
        return ""
    try:
        tokens = ast.literal_eval(s)
        if isinstance(tokens, list):
            return " ".join(str(t) for t in tokens if str(t).strip())
    except (ValueError, SyntaxError):
        pass
    s = s.strip("[]'\" ")
    return " ".join(part.strip().strip("'\"") for part in s.split(",") if part.strip())


def vader_label(score):
    if score >= 0.05:
        return 2  # Positive
    if score <= -0.05:
        return 0  # Negative
    return 1  # Neutral


for f in ["cleaned_reviews.csv", "preprocessed_reviews.csv"]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} not found. Run preprocessor.py first.")

clean = pd.read_csv("cleaned_reviews.csv")
pre = pd.read_csv("preprocessed_reviews.csv")
n = min(len(clean), len(pre))
clean, pre = clean.iloc[:n].reset_index(drop=True), pre.iloc[:n].reset_index(drop=True)

df = pd.DataFrame({
    "phone": pre["phone"].values,
    "text": clean["review"].astype(str).values,
    "preprocessed_text": pre["review"].apply(join_tokens).values,
})
df["text"] = df["text"].replace({"nan": "", "None": ""}).fillna("")
df = df[df["text"].str.strip().str.len() > 2].reset_index(drop=True)
df["preprocessed_text"] = df["preprocessed_text"].fillna("").replace({"nan": ""})

df["compound"] = df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment"] = df["compound"].apply(vader_label).astype(int)
df["sentiment_name"] = df["sentiment"].map(LABEL_MAP)

cols = ["phone", "text", "preprocessed_text", "sentiment", "sentiment_name", "compound"]
df[cols].to_csv("labeled_data.csv", index=False, encoding="utf-8")

print("Prepared labeled_data.csv")
print(f"Total samples: {len(df)}")
print(f"\nSentiment distribution:\n{df['sentiment_name'].value_counts()}")
