# Emotion Recognition in Hinglish using Machine Learning and Transformer Models

## 1. Introduction & Problem Description
Social media platforms in India widely use Hinglish (Hindi written in Latin script mixed with English). Automatically identifying emotions in such code-mixed text is challenging due to spelling variation, mixed vocabulary, and the possibility of multiple emotions appearing in a single sentence.  
This project aims to build a **multi-label emotion classifier** that can assign one or more of 28 predefined emotions to a given Hinglish sentence.  
The goal is to compare traditional machine learning, deep learning, and transformer-based models in terms of performance and suitability for this task.

---

## 2. Dataset Source & Preprocessing

**Dataset:** EmoHi-58K (Hindi translation of the GoEmotions dataset)  
**Source:** Kaggle  
https://www.kaggle.com/datasets/debadityashome/emohi58k-finegrained-hindi-emotion-datatext/data  
**Total samples:** ~58,000  
**Labels:** 28 emotions (multi-label)  
**Format:** id | labels (list of emotion indices) | text

### Preprocessing Steps Applied
* Lowercasing
* Removal of extra punctuation and special symbols
* Basic Hindi + English stopword filtering (NLTK)
* Tokenization using:
  * TF-IDF (for SVM)
  * Keras Tokenizer (for LSTM)
  * HuggingFace AutoTokenizer (for DistilBERT)
* Label vectors converted to 28-dimensional multi-hot encoding  
* Train/Validation/Test split preserved from dataset

No samples were removed; dataset was used as provided.

---

## 3. Methods & Model Selection

Three different modeling approaches were used:

| Model | Type | Reason for Inclusion |
|--------|------|----------------------|
| TF-IDF + SVM | Traditional ML | Baseline, interpretable, fast to train |
| Bi-LSTM | Deep Learning | Captures word order and sequence patterns |
| DistilBERT (multilingual) | Transformer | Pretrained contextual embeddings, state-of-the-art for text tasks |

### Why these methods?
* SVM provides a reproducible baseline using lexical features.
* LSTM enables learning of sequential dependencies beyond word frequency.
* DistilBERT is a lightweight multilingual transformer capable of handling Hinglish code-mixed text and long-range context.

### Alternative approaches considered
* XGBoost + TF-IDF (not chosen due to similar nature to SVM)
* CNN-text classifier (excluded due to weaker sentence-level performance)
* Full BERT base model (excluded for longer training time in free Colab GPU)

---

## 4. How to Run the Code

### Clone and install
```bash
git clone https://github.com/<your-username>/emotion-recognition-hinglish.git
cd emotion-recognition-hinglish
pip install -r requirements.txt

## 5. Experiments & Results

| Model | Valid Acc | Valid F1 | Test Acc | Test F1 |
|--------|-----------|----------|----------|----------|
| TF-IDF + SVM | 0.169 | 0.193 | 0.170 | 0.193 |
| Bi-LSTM | 0.306 | 0.345 | 0.299 | 0.338 |
| **DistilBERT (m-cased)** | **0.368** | **0.418** | **0.368** | **0.416** |

### Observations
* Exact-match accuracy is naturally low in multi-label problems because a prediction is counted as correct **only when all labels match**.
* F1-score is a better metric here because it measures label-wise correctness.
* Performance increases consistently from SVM → LSTM → DistilBERT as models become more context-aware.
* DistilBERT outperforms both traditional ML and LSTM due to:
  - Pretraining on large multilingual corpora
  - Ability to handle code-mixed Hindi-English text
  - Contextual word embeddings rather than fixed vectors



