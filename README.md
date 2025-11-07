# Emotion Recognition in Hinglish (Hindi–English) Text with ML & Transformers

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

Can machines understand emotions in Hindi–English code-mixed text? **Yes.**  
We trained classic ML, an LSTM, and a multilingual **DistilBERT** on the **EmoHi-58K** dataset (28 emotions) and found that **Transformers clearly win**.

**Best model:** DistilBERT (multilingual) → **F1 = XX.XX**, Accuracy = XX.XX  
**Baselines:** LSTM → F1 = XX.XX, SVM (TF-IDF) → F1 = XX.XX

---

## 🔎 What This Project Is About

Social platforms in India are full of Hinglish. Detecting **multi-label emotions** (e.g., *anger + annoyance* together) is useful for moderation, safety, and analytics.  
We compare **3 approaches**:

- **SVM + TF-IDF** (baseline)
- **Bi-LSTM** (sequence model)
- **DistilBERT (multilingual)** (state-of-the-art)

---

## 🧭 Project Workflow

```mermaid
graph LR
    A[Data & Labels/nEmoHi-58K (28 emotions)] --> B[Preprocess/ncleaning, tokenization]
    B --> C[Baselines/nTF-IDF + SVM]
    B --> D[Deep Learning/nBi-LSTM]
    B --> E[Transformer/nDistilBERT (m-cased)]
    C --> F[Evaluation]
    D --> F
    E --> F
    F --> G[Compare & Pick Best]
