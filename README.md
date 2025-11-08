# Multi-Label Emotion Detection in Hinglish Text using SVM, LSTM & DistilBERT

Can a machine understand emotions hidden inside Hinglish (Hindi+English) social media text?  
Turns out — yes. And transformers do it better than we expect.

This project builds a system that can detect **one or more of 28 human emotions** (anger, joy, love, sarcasm, confusion, etc.) from a single Hinglish sentence — even when the text is code-mixed, informal, and full of slang.

📌 Example  
`"yeh kya bakwaas hai, mujhe gussa aa raha hai"` → `[anger, annoyance]`

---

## 1. Problem Description & Motivation

Hinglish is widely used across Indian social platforms (YouTube, WhatsApp, Instagram, Reddit), but most NLP models are trained on either clean English or formal Hindi.  
Detecting emotions in Hinglish is difficult because:

* It mixes two languages in the same sentence  
* Spellings vary (e.g., "gussa", "gusa", "ghussa")  
* One sentence may express multiple emotions at once  
* No standard grammar or structure exists  

🎯 **Objective:**  
Build and compare multiple ML/DL models to perform **multi-label emotion classification** on Hinglish text and identify which modeling approach works best.

---

## 2. Dataset & Preprocessing

**Dataset:** EmoHi-58K (Hindi version of GoEmotions, by Google Research)  
**Source:** Kaggle  
https://www.kaggle.com/datasets/debadityashome/emohi58k-finegrained-hindi-emotion-datatext/data  

| Property | Details |
|----------|---------|
| Samples | ~58,000 text entries |
| Labels | 28 possible emotions (multi-label) |
| Format | id, labels list, raw text |
| Split | Train / Validation / Test CSVs provided |

### 🔧 Preprocessing Steps

* Lowercasing text  
* Removing punctuation & special symbols  
* Hindi + English stopword filtering  
* Tokenization  
  * TF-IDF (for SVM)
  * Keras Tokenizer (for LSTM)
  * HuggingFace AutoTokenizer (for DistilBERT)
* Emotion labels converted into 28-dim multi-hot vectors  
* Dataset used as-is (no samples removed)

---

## 3. Methods & Model Choices

| Model | Type | Why Included? |
|--------|------|---------------|
| TF-IDF + SVM | Classical ML | Fast, interpretable baseline model |
| Bi-LSTM | Deep Learning | Learns sequential word patterns |
| DistilBERT (Multilingual) | Transformer | Best for context + code-mixing |

### 📝 Why These Models?

* SVM gives a useful baseline for text-classification without deep learning  
* LSTM captures sentence structure and ordering  
* DistilBERT provides contextual embeddings trained on multilingual corpora — ideal for Hinglish

### ⏭ Alternatives Considered (not used)

* XGBoost + TF-IDF (similar to SVM baseline)  
* CNN Text Classifier (performs worse on sentence-level tasks)  
* Full BERT Base (too slow to fine-tune on free Colab GPU)

---

## 4. Project Workflow
```
graph LR
    A["Dataset: EmoHi-58K"] --> B["Preprocessing"]
    B --> C["SVM\n(TF-IDF)"]
    B --> D["Bi-LSTM"]
    B --> E["DistilBERT"]
    C --> F["Evaluation"]
    D --> F
    E --> F
    F --> G["Model Comparison"]
```
---

## 5. Experiments & Results
Model	Valid Acc	Valid F1	Test Acc	Test F1
TF-IDF + SVM	0.169	0.193	0.170	0.193
Bi-LSTM	0.306	0.345	0.299	0.338
DistilBERT (m-cased)	0.368	0.418	0.368	0.416

### 📊 Key Observations
* Exact-match accuracy is strict in multi-label problems — F1-score is a better metric

* Performance steadily improves from SVM → LSTM → DistilBERT

* DistilBERT performs best due to contextual understanding + multilingual pretraining

* Traditional ML alone is not sufficient for code-mixed Hinglish text

---

## 6. How to Run the Code
```
* ✅ Install dependencies
pip install -r requirements.txt

* ✅ Open notebook for training
notebooks/emotion_recognition_hinglish.ipynb

* ✅ Optional: Run inference from terminal
python src/predict.py --text "mujhe bahut gussa aa raha hai"

* Expected output
['anger', 'annoyance']
```
---

## 7. Conclusion

The experiments show that classical ML methods such as SVM struggle with multi-label Hinglish emotion classification, while sequence-based models like LSTM perform better by learning word order.
However, the best performance is achieved by the DistilBERT transformer model, which leverages contextual multilingual embeddings and handles code-mixed text effectively.
This confirms that transformer-based architectures are currently the most suitable choice for emotion recognition in Indian social media text.

---

## 8. References

EmoHi-58K Dataset – Kaggle

HuggingFace Transformers Documentation

Scikit-Learn Machine Learning Library

TensorFlow / Keras

PyTorch Documentation
