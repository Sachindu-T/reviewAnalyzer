# Sentiment Analysis of Social Media Posts Using Transformer-Based Language Models

---

**Course:** Natural Language Processing
**Program:** Master of Computer Engineering
**Student ID:** 22CDS0409
**Date:** August 2026

---

## Abstract

This project presents a complete Natural Language Processing (NLP) pipeline for **3-class sentiment analysis** (Negative, Neutral, Positive) on real-world mobile phone user reviews collected from GSMArena. The study compares four traditional machine learning baselines — Naive Bayes, Logistic Regression, Support Vector Machine (SVM), and Random Forest — against a **fine-tuned BERT (Bidirectional Encoder Representations from Transformers)** transformer model. Data is collected via web scraping, preprocessed through a multi-stage pipeline (lowercasing, emoji removal, punctuation stripping, stop-word removal, lemmatization), and labeled automatically using the VADER sentiment lexicon. All models are evaluated on an identical stratified 80/10/10 train/validation/test split (3,225 / 404 / 404 samples). Results demonstrate that BERT achieves **81.68% accuracy** and **0.7926 macro F1-score**, outperforming the strongest traditional baseline (SVM, 74.01% accuracy) by **+7.7 percentage points**. The project validates that transformer-based contextual embeddings provide superior semantic understanding compared to frequency-based TF-IDF features for informal social media-style text classification.

**Keywords:** Sentiment Analysis, BERT, Transformer, NLP, TF-IDF, VADER, Text Classification, Machine Learning

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Literature Review](#2-background-and-literature-review)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Implementation](#5-implementation)
6. [Results and Discussion](#6-results-and-discussion)
7. [Conclusion](#7-conclusion)
8. [Future Work](#8-future-work)
9. [References](#9-references)
10. [Appendix](#10-appendix)

---

## 1. Introduction

### 1.1 Problem Statement

In the modern digital era, social media platforms, product review websites, and online forums generate enormous volumes of textual data daily. Organizations and businesses need to automatically understand public opinion from these large-scale text sources to make informed decisions. Manual analysis of millions of reviews, tweets, and comments is infeasible, necessitating automated sentiment analysis systems.

Sentiment analysis, a sub-task of Natural Language Processing (NLP), involves determining the emotional tone or opinion expressed in a piece of text. The challenge is particularly acute for informal text such as product reviews, which contain slang, abbreviations, emojis, mixed sentiments, and domain-specific terminology.

### 1.2 Research Question

**Can a fine-tuned BERT transformer model outperform traditional machine learning baselines (Naive Bayes, Logistic Regression, SVM, Random Forest) for 3-class sentiment analysis on informal social media-style text?**

### 1.3 Objectives

The specific objectives of this project are:

1. **Data Collection:** Scrape real mobile phone user reviews from GSMArena, a major phone review platform.
2. **Text Preprocessing:** Develop a robust preprocessing pipeline to clean and normalize raw, noisy text.
3. **Sentiment Labeling:** Automatically label data using VADER, a lexicon-based sentiment analyzer tuned for social media text.
4. **Baseline Modeling:** Build and evaluate four traditional ML classifiers using TF-IDF features.
5. **Transformer Modeling:** Fine-tune a pre-trained BERT model for 3-class sentiment classification.
6. **Comparison:** Rigorously compare all models on identical train/test splits using accuracy, precision, recall, and F1-score metrics.

### 1.4 Scope

This project focuses on English-language mobile phone reviews from GSMArena. The sentiment classification is limited to three classes (Negative, Neutral, Positive). The comparison is between four TF-IDF-based traditional classifiers and one fine-tuned transformer model (BERT).

---

## 2. Background and Literature Review

### 2.1 Natural Language Processing

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language through computational techniques. It is fundamental to building intelligent systems that can interact naturally with humans and process massive amounts of textual data. Key applications include chatbots, search engines, social media monitoring, machine translation, and text analytics.

### 2.2 Sentiment Analysis

Sentiment analysis (also called opinion mining) is the computational study of opinions, sentiments, emotions, and attitudes expressed in text. It classifies text into sentiment categories, typically:

- **Binary:** Positive vs. Negative
- **Ternary:** Positive vs. Neutral vs. Negative (used in this project)
- **Fine-grained:** Very Positive, Positive, Neutral, Negative, Very Negative

### 2.3 Evolution of Text Representation

The quality of text representation is crucial for sentiment classification performance:

| Representation | Type | Context-Aware | Limitation |
|---------------|------|:------------:|------------|
| Bag of Words (BoW) | Frequency-based | No | Loses word order and context |
| TF-IDF | Weighted importance | No | Sparse, no semantic meaning |
| Word2Vec / GloVe | Dense embeddings | Partial | Static, polysemy unhandled |
| **BERT** | Contextual embeddings | **Yes** | Computationally expensive |

### 2.4 Transformer Architecture

The Transformer architecture, introduced by Vaswani et al. (2017), revolutionized NLP through:

- **Self-Attention Mechanism:** Every token attends to all other tokens, capturing long-range dependencies regardless of distance.
- **Positional Encoding:** Injects sequence-order information since Transformers have no recurrence or convolution.
- **Multi-Head Attention:** Parallel attention operations capture diverse relationship patterns.
- **Feed-Forward Network:** Non-linear transformation over attention outputs.

### 2.5 BERT (Bidirectional Encoder Representations from Transformers)

BERT (Devlin et al., 2019) is a pre-trained Transformer model that uses bidirectional context. Unlike GPT (unidirectional), BERT reads text in both directions simultaneously, making it particularly suitable for classification tasks. Key characteristics:

- **Pre-training objectives:** Masked Language Model (MLM) and Next Sentence Prediction (NSP)
- **Architecture:** 12 Transformer layers, 768 hidden size, 12 attention heads, 30522 vocab size
- **Fine-tuning:** A classification head is added on top of the [CLS] token output for downstream tasks

### 2.6 VADER Sentiment Analyzer

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon-based sentiment analysis tool specifically designed for social media text. It provides a compound score ranging from -1 (most negative) to +1 (most positive), with thresholds at ±0.05 for classifying Positive, Negative, and Neutral sentiments.

---

## 3. Dataset

### 3.1 Data Source

Data is collected from **GSMArena** (https://www.gsmarena.com/reviews.php3), a leading mobile phone review website. User reviews are scraped across 99 pages of phone review listings using a custom web scraper (`collectReviwes.py`).

### 3.2 Data Collection Process

The scraper performs the following steps:
1. Iterates through 99 pages of GSMArena review listings
2. Extracts individual review page links and titles
3. Follows each link to scrape user comment text (CSS class `uopin`)
4. Saves results as JSON: `{phone_name: [review1, review2, ...]}`
5. Implements rate limiting (1-second delay between requests) and resume capability

### 3.3 Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Raw reviews collected | 4,071 unique phone entries |
| Labeled samples (after cleaning) | 4,033 |
| Unique products (phones) | 1,521 |
| Positive samples | 2,087 (51.8%) |
| Neutral samples | 1,004 (24.9%) |
| Negative samples | 942 (23.3%) |
| Source | GSMArena |

### 3.4 Sentiment Labeling

Labels are generated automatically using **VADER** with the following thresholds:

- **Positive (2):** compound score >= 0.05
- **Negative (0):** compound score <= -0.05
- **Neutral (1):** -0.05 < compound score < 0.05

### 3.5 Data Files Pipeline

| Stage | File | Description |
|-------|------|-------------|
| 1. Raw | `customer_reviews.json` | Scraped reviews (phone -> list of review texts) |
| 2. Cleaned | `cleaned_reviews.csv` | Lowercased, emoji/punctuation removed |
| 3. Preprocessed | `preprocessed_reviews.csv` | Stop-word removal + lemmatized tokens |
| 4. Labeled | `labeled_data.csv` | Final labeled dataset with sentiment classes |

### 3.6 Train/Validation/Test Split

All models use an identical **stratified 80/10/10 split** with `random_state=42`:

| Split | Samples | Proportion | Purpose |
|-------|---------|-----------|---------|
| Training | 3,225 | 80% | Model learning |
| Validation | 404 | 10% | Hyperparameter tuning / best checkpoint |
| Testing | 404 | 10% | Final evaluation |

The stratified split ensures proportional representation of all three sentiment classes across splits, and the fixed random seed guarantees the test set is identical across all models.

---

## 4. Methodology

### 4.1 NLP Pipeline Overview

```
Data Collection -> Cleaning -> Tokenization -> Feature Extraction -> Model Training -> Evaluation
```

### 4.2 Text Preprocessing

The preprocessing pipeline (`preprocessor.py`) applies the following steps:

1. **Lowercasing:** Convert all text to lowercase for uniformity
2. **Emoji Removal:** Strip Unicode emoji characters using the `emoji` library
3. **Punctuation Removal:** Remove non-alphanumeric characters using regex `[^\w\s]`
4. **URL Removal:** URLs are handled during BeautifulSoup text extraction
5. **Tokenization:** Word-level tokenization using NLTK `word_tokenize`
6. **Stop-Word Removal:** Remove common English stop words (NLTK stop-word list)
7. **Lemmatization:** Reduce words to base form using NLTK `WordNetLemmatizer`

### 4.3 Feature Extraction — TF-IDF

For traditional ML models, text is represented using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_features | 20,000 | Limit vocabulary size |
| ngram_range | (1, 2) | Unigrams + bigrams capture phrases |
| sublinear_tf | True | Apply log normalization |
| min_df | 2 | Ignore very rare terms |
| max_df | 0.95 | Ignore overly common terms |

### 4.4 Traditional Machine Learning Models

Four classifiers are trained on TF-IDF features:

| Model | Algorithm | Key Parameters |
|-------|-----------|----------------|
| Naive Bayes | `MultinomialNB` | Default parameters |
| Logistic Regression | `LogisticRegression` | max_iter=1000, class_weight=balanced |
| SVM | `LinearSVC` | max_iter=2000, class_weight=balanced |
| Random Forest | `RandomForestClassifier` | n_estimators=300, n_jobs=-1, class_weight=balanced |

Models with `class_weight=balanced` handle the class imbalance (52% Positive, 25% Neutral, 23% Negative) by weighting minority classes higher during training.

### 4.5 BERT Fine-Tuning

The pre-trained `bert-base-uncased` model is fine-tuned for 3-class classification:

| Hyperparameter | Value |
|----------------|-------|
| Model | `bert-base-uncased` (Hugging Face) |
| Architecture | 12 layers, 768 hidden, 12 heads |
| Vocab size | 30,522 |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 16 |
| Max sequence length | 128 |
| Weight decay | 0.01 |
| Output classes | 3 (Negative, Neutral, Positive) |

**Fine-tuning process:**
1. Tokenize text using BERT WordPiece tokenizer
2. Add classification head on [CLS] token output
3. Train with Hugging Face `Trainer` API
4. Evaluate on validation set each epoch
5. Save best checkpoint by accuracy
6. Final evaluation on held-out test set

### 4.6 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correctly classified instances |
| **Precision (macro)** | Average precision across classes (unweighted) |
| **Recall (macro)** | Average recall across classes (unweighted) |
| **F1-score (macro)** | Harmonic mean of precision and recall (unweighted) |
| **F1-score (weighted)** | F1 averaged by class support |
| **Confusion Matrix** | 3x3 matrix of true vs. predicted labels |

---

## 5. Implementation

### 5.1 System Architecture

The project is implemented as a modular Python pipeline with 7 source files:

| File | Role | Stage |
|------|------|-------|
| `collectReviwes.py` | Web scraper for GSMArena reviews | Data Collection |
| `preprocessor.py` | Text cleaning and lemmatization | Preprocessing |
| `prepare_data.py` | VADER sentiment labeling | Labeling |
| `Traditional_Models_With_TF-IDF.py` | TF-IDF + 4 baseline classifiers | Baseline Training |
| `bert.py` | BERT fine-tuning and evaluation | Transformer Training |
| `compare_results.py` | Model comparison table generation | Comparison |
| `main.py` | Pipeline orchestrator (runs all stages) | Orchestration |

### 5.2 Pipeline Orchestration

`main.py` runs the complete pipeline sequentially:
1. `prepare_data.py` — VADER labeling
2. `Traditional_Models_With_TF-IDF.py` — Train 4 baselines
3. `bert.py` — Fine-tune BERT
4. `compare_results.py` — Generate comparison table

Each stage checks for successful completion before proceeding to the next.

### 5.3 Dependencies

| Category | Package | Version |
|----------|---------|---------|
| Data processing | pandas | >= 2.0 |
| Numerical | numpy | >= 1.24 |
| Machine Learning | scikit-learn | >= 1.3 |
| NLP toolkit | nltk | >= 3.8 |
| Emoji handling | emoji | >= 2.0 |
| Web scraping | requests | >= 2.28 |
| HTML parsing | beautifulsoup4 | >= 4.12 |
| Deep Learning | torch | >= 2.0 |
| Transformers | transformers | >= 4.30 |

**NLTK data required:** `vader_lexicon`, `stopwords`, `punkt`, `wordnet`, `omw-1.4`

### 5.4 Reproducibility

- All random splits use `random_state=42` (stratified)
- VADER thresholds: `compound >= 0.05` -> Positive, `<= -0.05` -> Negative, else Neutral
- BERT model: `bert-base-uncased` (fixed pretrained weights)
- Results are saved to CSV/JSON files for reproducibility

---

## 6. Results and Discussion

### 6.1 Overall Performance Comparison

All models are evaluated on the same test set (404 samples):

| Model | Accuracy | F1 (macro) | F1 (weighted) | Precision (macro) | Recall (macro) |
|-------|:--------:|:----------:|:--------------:|:------------------:|:--------------:|
| Naive Bayes | 54.95% | 0.3158 | 0.4217 | 0.7212 | 0.3780 |
| Logistic Regression | 72.03% | 0.6882 | 0.7219 | 0.6860 | 0.6940 |
| SVM | **74.01%** | **0.7051** | **0.7372** | 0.7106 | 0.7043 |
| Random Forest | 71.78% | 0.6457 | 0.6917 | 0.7349 | 0.6499 |
| **BERT (fine-tuned)** | **81.68%** | **0.7926** | **0.8155** | **0.7998** | **0.7867** |

**Key Finding:** BERT achieves **+7.7 percentage points** accuracy improvement over the strongest traditional baseline (SVM).

### 6.2 Analysis of Traditional Models

**Naive Bayes (54.95% accuracy):** The worst performer. Despite high precision (0.7212), it has extremely low recall (0.3780) and F1 (0.3158), indicating severe bias toward the majority class (Positive). The independence assumption of Naive Bayes fails to capture word dependencies in sentiment-bearing text.

**Logistic Regression (72.03% accuracy):** A strong baseline with balanced precision (0.6860) and recall (0.6940). The balanced class weights effectively mitigate the class imbalance problem.

**SVM (74.01% accuracy):** The best traditional model. LinearSVC finds optimal hyperplanes in the high-dimensional TF-IDF space, achieving the highest F1 among baselines (0.7051 macro).

**Random Forest (71.78% accuracy):** Competitive accuracy but the lowest recall among the top three (0.6499), suggesting it overfits to certain patterns and misses minority class instances.

### 6.3 BERT Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Negative (0) | 0.728 | 0.713 | 0.720 | 94 |
| Neutral (1) | 0.817 | 0.752 | 0.784 | 101 |
| Positive (2) | 0.854 | 0.895 | 0.874 | 209 |

BERT performs best on the **Positive** class (89.5% recall, 87.4% F1) and weakest on **Negative** (71.3% recall, 72.0% F1). The performance hierarchy aligns with class frequency: Positive (51.8%) > Neutral (24.9%) > Negative (23.3%).

### 6.4 Confusion Matrix Analysis

BERT test-set confusion matrix (rows = true, columns = predicted):

| | Predicted Neg | Predicted Neu | Predicted Pos |
|---|:---:|:---:|:---:|
| **True Negative** | 67 | 10 | 17 |
| **True Neutral** | 10 | 76 | 15 |
| **True Positive** | 15 | 7 | 187 |

**Key observations:**
- **Positive class** is most accurately classified (187/209 = 89.5% correct)
- **Negative-Neutral confusion** is the primary error source (10 Negative misclassified as Neutral, 10 Neutral misclassified as Negative)
- **Negative-Positive confusion** is less common but notable (17 Negative misclassified as Positive)

### 6.5 Why BERT Outperforms Traditional Models

| Aspect | TF-IDF + Traditional ML | BERT |
|--------|------------------------|------|
| Word representation | Sparse, frequency-based | Dense, contextual |
| Context understanding | None (bag-of-words) | Full bidirectional context |
| Polysemy handling | Cannot distinguish same word in different contexts | Different embeddings per context |
| Pre-training | None | Trained on massive corpus (BookCorpus + Wikipedia) |
| Transfer learning | No | Yes (pre-trained knowledge) |

BERT's contextual embeddings capture nuanced semantic relationships that TF-IDF features miss, particularly for:
- Negation handling ("not good" -> Negative, not Positive)
- Implicit sentiment ("battery lasts 2 hours" -> Negative context)
- Domain-specific usage ("bloatware" -> Negative)

### 6.6 Error Analysis

Despite strong performance, BERT misclassifies 73 out of 404 test samples. Common failure modes include:

1. **Sarcasm:** Positive words expressing negative intent (e.g., "Great! Another software crash.")
2. **Irony:** Statements meaning the opposite of their literal interpretation
3. **Domain-Specific Language:** Technical jargon not well represented in training data
4. **Mixed Sentiments:** Text containing both positive and negative opinions (e.g., "Camera is amazing but battery dies in 2 hours")
5. **Short Text Ambiguity:** Very short reviews with insufficient context for reliable classification

---

## 7. Conclusion

This project demonstrates a complete NLP pipeline from raw web-scraped text to intelligent sentiment prediction. The key conclusions are:

1. **BERT outperforms all traditional baselines** for 3-class sentiment analysis, achieving 81.68% accuracy compared to the best baseline (SVM) at 74.01% — a significant +7.7 percentage point improvement.

2. **Contextual embeddings provide superior semantic understanding** compared to frequency-based TF-IDF features, particularly for handling negation, implicit sentiment, and domain-specific language in informal text.

3. **The complete NLP workflow** — from data collection and preprocessing through labeling and model training to evaluation and comparison — is successfully implemented as a reproducible, modular pipeline.

4. **Class imbalance** (52% Positive, 25% Neutral, 23% Negative) affects all models, with the Negative class consistently being the hardest to classify correctly. BERT mitigates this better than traditional models through its richer representation.

5. **VADER provides effective automatic labeling** for social media text, enabling large-scale dataset creation without manual annotation, though it introduces some label noise that affects all models equally.

---

## 8. Future Work

| Direction | Potential Approach | Expected Benefit |
|-----------|-------------------|------------------|
| Multilingual NLP | mBERT, XLM-R | Support for multiple languages |
| Sinhala-English Code-Switching | Indic BERT variants | Handle mixed-language text |
| Emotion Detection | EmoBERT, GoEmotions | Beyond sentiment to specific emotions |
| Aspect-Based Sentiment | ABSA-BERT | Sentiment per aspect (camera, battery, etc.) |
| Zero-shot Classification | GPT, PaLM via prompting | No labeled data required |
| Model Efficiency | DistilBERT, ALBERT | Faster inference, smaller model size |
| Human Annotation | Expert labeling | Eliminate VADER label noise |
| Active Learning | Iterative labeling | Reduce annotation cost |

---

## 9. References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171-4186.

2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

3. Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216-225.

4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

5. Wolf, T., et al. (2020). HuggingFace's Transformers: State-of-the-Art Natural Language Processing. *Proceedings of EMNLP*, 2020.

6. Loper, E., & Bird, S. (2002). NLTK: The Natural Language Toolkit. *Proceedings of the ACL-02 Workshop on Effective Tools and Methodologies for Teaching NLP*, 63-70.

---

## 10. Appendix

### Appendix A: Project File Structure

```
.
├── collectReviwes.py                          # Web scraper for GSMArena reviews
├── preprocessor.py                            # Text cleaning and lemmatization
├── prepare_data.py                            # VADER sentiment labeling
├── Traditional_Models_With_TF-IDF.py         # TF-IDF + 4 baseline classifiers
├── bert.py                                    # BERT fine-tuning
├── compare_results.py                         # Model comparison
├── main.py                                    # Pipeline orchestrator
├── TF-IDF.py                                  # Legacy: simpler 2-class version
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
├── customer_reviews.json                      # Raw scraped reviews
├── cleaned_reviews.csv                        # Cleaned text
├── preprocessed_reviews.csv                   # Lemmatized tokens
├── labeled_data.csv                           # Final labeled dataset
├── results/
│   ├── traditional_metrics.csv                # Traditional model metrics
│   ├── bert_metrics.csv                       # BERT metrics
│   └── comparison_results.csv                 # Combined comparison
└── bert_results/
    ├── config.json                            # BERT model config
    ├── model.safetensors                      # Fine-tuned weights (~418 MB)
    ├── tokenizer.json                         # BERT tokenizer
    ├── tokenizer_config.json                  # Tokenizer config
    ├── bert_metrics.json                      # BERT evaluation metrics
    ├── test_report.json                       # Per-class report
    └── confusion_matrix.json                  # 3x3 confusion matrix
```

### Appendix B: BERT Hyperparameter Configuration

```python
{
    "model_name": "bert-base-uncased",
    "num_labels": 3,
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": true,
    "metric_for_best_model": "accuracy"
}
```

### Appendix C: Confusion Matrix Values

```
Predicted ->  Negative    Neutral    Positive
True Neg      [67,         10,        17]
True Neu      [10,         76,        15]
True Pos      [15,          7,       187]
```

### Appendix D: How to Reproduce Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 3. Run the full pipeline
python main.py

# 4. Or run individual stages
python collectReviwes.py
python preprocessor.py
python prepare_data.py
python Traditional_Models_With_TF-IDF.py
python bert.py --epochs 3 --batch_size 16 --max_length 128
python compare_results.py
```

---

*Report prepared for the Natural Language Processing course, Master of Computer Engineering program.*
