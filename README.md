# Emotion Recognition in Hinglish using ML & Transformers

This project focuses on multi-label emotion classification in Hinglish (Hindi–English code-mixed) text using three different approaches: a traditional machine learning baseline, a deep learning model, and a transformer-based model. The goal is to compare performance across these architectures and evaluate how well they recognize multiple emotions simultaneously from a single sentence.

---

## 1. Project Overview

Modern Indian social media platforms widely use Hinglish — a mix of Hindi written in Latin script along with English. Detecting emotions in such text is challenging due to:

* Code-mixing
* Spelling variations
* Emotion overlap (multi-label)
* Limited annotated datasets

To address this, three models were trained and evaluated:

| Model | Type | Highlights |
|--------|------|------------|
| TF-IDF + SVM | Classical ML | Baseline, bag-of-words features |
| Bi-LSTM | Deep Learning | Learns word order and context |
| DistilBERT (Multilingual) | Transformer | Pretrained contextual embeddings, fine-tuned |

All experiments were performed on the **EmoHi-58K** dataset (Hindi translation of GoEmotions), which contains 28 possible emotion labels per sentence.

---

## 2. Dataset

| Property | Details |
|----------|---------|
| Name | EmoHi-58K (Hindi version of GoEmotions) |
| Size | ~58,000 text samples |
| Labels | 28 emotions (multi-label) |
| Format | `id`, `label` (list of ids), `text` |
| Split | Train / Validation / Test CSVs |

Each sentence may contain **multiple emotions**, e.g.:

