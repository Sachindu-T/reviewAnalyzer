import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments,
)

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 3
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
RESULTS_DIR = "results"
BERT_DIR = "bert_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BERT_DIR, exist_ok=True)


def make_dataset(encodings, labels):
    return [
        {**{k: torch.tensor(v[i]) for k, v in encodings.items()}, "labels": torch.tensor(labels[i])}
        for i in range(len(labels))
    ]


def compute_metrics(pred):
    labels, preds = pred.label_ids, pred.predictions.argmax(-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1, "precision": p, "recall": r}


def evaluate(model, tokenizer, texts, labels, max_length):
    enc = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    preds = []
    with torch.no_grad():
        for i in range(0, len(labels), 32):
            batch = {k: v[i:i+32] for k, v in enc.items()}
            preds.extend(model(**batch).logits.argmax(-1).cpu().numpy().tolist())
    labels, preds = np.array(labels), np.array(preds)
    names = [LABEL_MAP[i] for i in range(NUM_LABELS)]
    report = classification_report(labels, preds, target_names=names, zero_division=0, output_dict=True)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_LABELS))).tolist()
    print("\nBERT test report:")
    print(classification_report(labels, preds, target_names=names, zero_division=0))
    return report, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--subset", type=int, default=0, help="If >0, use first N samples.")
    args = parser.parse_args()

    df = pd.read_csv("labeled_data.csv").dropna(subset=["text", "sentiment"])
    if args.subset > 0:
        df = df.head(args.subset)
    df = df.reset_index(drop=True)

    print(f"Dataset: {len(df)} samples")
    print(f"Classes: {df['sentiment'].value_counts().sort_index().to_dict()}")

    # 80/10/10 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["sentiment"].tolist(),
        test_size=0.1, random_state=42, stratify=df["sentiment"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=1/9, random_state=42, stratify=y_train)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(y_test)}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    train_ds = make_dataset(tokenizer(X_train, truncation=True, padding=True, max_length=args.max_length), y_train)
    val_ds = make_dataset(tokenizer(X_val, truncation=True, padding=True, max_length=args.max_length), y_val)

    training_args = TrainingArguments(
        output_dir=BERT_DIR, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=32,
        learning_rate=args.lr, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        logging_dir=os.path.join(BERT_DIR, "logs"), report_to="none",
        load_best_model_at_end=True, metric_for_best_model="accuracy", seed=42)

    print("Training BERT...")
    Trainer(model=model, args=training_args, train_dataset=train_ds,
            eval_dataset=val_ds, compute_metrics=compute_metrics).train()

    model.save_pretrained(BERT_DIR)
    tokenizer.save_pretrained(BERT_DIR)

    print("Evaluating on test set...")
    report, cm = evaluate(model, tokenizer, X_test, y_test, args.max_length)

    # Test metrics
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
    logits = []
    with torch.no_grad():
        for i in range(0, len(y_test), 32):
            batch = {k: v[i:i+32] for k, v in test_enc.items()}
            logits.append(model(**batch).logits.cpu().numpy())
    preds = np.argmax(np.concatenate(logits), axis=1)
    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro", zero_division=0)
    f1w, _, _, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)

    metrics = {
        "model": "BERT", "accuracy": float(acc), "precision_macro": float(p),
        "recall_macro": float(r), "f1_macro": float(f1), "f1_weighted": float(f1w),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.lr,
        "train_size": len(X_train), "val_size": len(X_val), "test_size": len(y_test),
    }
    for name, data in [("bert_metrics.json", metrics), ("test_report.json", report), ("confusion_matrix.json", cm)]:
        with open(os.path.join(BERT_DIR, name), "w") as f:
            json.dump(data, f, indent=2)

    bert_csv = os.path.join(RESULTS_DIR, "bert_metrics.csv")
    if os.path.exists(bert_csv):
        os.remove(bert_csv)
    pd.DataFrame([{k: metrics[k] for k in ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "train_size", "test_size"]}]).to_csv(bert_csv, index=False)

    print(f"\nBERT: acc={acc:.4f} f1_macro={f1:.4f} f1_weighted={f1w:.4f}")
    print(f"Saved -> {BERT_DIR}, {bert_csv}")


if __name__ == "__main__":
    main()
