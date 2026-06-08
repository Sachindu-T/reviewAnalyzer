import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# 1. Load Data
df = pd.read_csv('labeled_preprocessed_reviews.csv')
df = df.dropna(subset=['review', 'sentiment'])

# BERT actually performs better with some context, but we will use your cleaned text here
X_train, X_test, y_train, y_test = train_test_split(
    df['review'].tolist(), df['sentiment'].tolist(), test_size=0.2, random_state=42
)

# 2. Load DistilBERT Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 3. Tokenize datasets
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# 4. Create PyTorch Dataset format
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, y_train)
test_dataset = ReviewDataset(test_encodings, y_test)

# 5. Load pre-trained DistilBERT
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 6. Define how to calculate metrics during training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 7. Configure Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # 3 Epochs is usually the sweet spot for BERT
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,               # Helps prevent overfitting
    eval_strategy="epoch",
    logging_dir='./logs'
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 9. Train and Evaluate
print("Training DistilBERT Model...")
trainer.train()

print("Evaluating Model...")
results = trainer.evaluate()
print("Transformer Model Results:", results)