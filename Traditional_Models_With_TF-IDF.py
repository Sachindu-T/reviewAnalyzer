import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

if not os.path.exists("labeled_data.csv"):
    raise FileNotFoundError("labeled_data.csv not found. Run prepare_data.py first.")

df = pd.read_csv("labeled_data.csv").dropna(subset=["sentiment"]).reset_index(drop=True)
df["preprocessed_text"] = df["preprocessed_text"].fillna("").astype(str).replace({"nan": ""})
X, y = df["preprocessed_text"], df["sentiment"].astype(int)
labels = sorted(y.unique().tolist())
target_names = [LABEL_MAP[l] for l in labels]

print(f"Dataset: {len(df)} | Classes: {y.value_counts().sort_index().to_dict()}")

# 80/10/10 stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state=42, stratify=y_train)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(y_test)}")

tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf, X_val_tfidf, X_test_tfidf = tfidf.fit_transform(X_train), tfidf.transform(X_val), tfidf.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "SVM": LinearSVC(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

rows = []
for name, model in models.items():
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    model.fit(X_train_tfidf, y_train)
    val_pred, test_pred = model.predict(X_val_tfidf), model.predict(X_test_tfidf)

    print("\nValidation Report:")
    print(classification_report(y_val, val_pred, labels=labels, target_names=target_names, zero_division=0))
    print("Test Report:")
    print(classification_report(y_test, test_pred, labels=labels, target_names=target_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred, labels=labels))

    report = classification_report(y_test, test_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    rows.append({
        "model": name,
        "accuracy": float((test_pred == y_test).mean()),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "train_size": len(X_train),
        "test_size": len(y_test),
    })

out_path = os.path.join(RESULTS_DIR, "traditional_metrics.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")
