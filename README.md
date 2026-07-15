
# 🤖 Intent Classification AI Assistant

An end-to-end Natural Language Processing (NLP) and Machine Learning chatbot built using Python, Scikit-Learn, and Streamlit. Moving away from rigid rule-based logic, this project leverages statistical TF-IDF text vectorization and a Logistic Regression classifier to intelligently predict user intent and respond contextually with calculated confidence thresholds.

---

## 🛠️ Project Architecture

```text
       User Input (Web GUI)
                │
                ▼
  Text Preprocessing (Lowercasing)
                │
                ▼
     TF-IDF Vectorizer Matrix
                │
                ▼
Machine Learning Classifier (Logistic Regression)
                │
                ▼
  Predicted Intent Tag + Confidence Score
                │
                ▼
       [Confidence > 45%]
        /            \
     YES              NO
      /                \
     ▼                  ▼
Select Random      Fallback Response:
Mapped Response   "Can you rephrase?"

```

---

## 📂 Project Structure

```text
AI_Chatbot/
│
├── data/
│   └── intents.json          # Structured dataset containing tags, patterns, and responses
│
├── models/
│   └── chatbot_model.pkl     # Pickled artifact combining fitted Vectorizer & ML Model
│
├── app.py                    # Streamlit Web GUI Application
├── train.py                  # Dataset parser and ML model training script
└── README.md                 # Project documentation

```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have **Python 3.8+** installed on your system.

### 2. Installation

Clone this repository and install the required dependencies:

```bash
pip install streamlit scikit-learn pandas

```

### 3. Training the Brain

If you modify or add new pattern data to `data/intents.json`, retrain the classifier by running:

```bash
python train.py

```

### 4. Running the Web App

Launch the interactive Streamlit user interface locally:

```bash
streamlit run app.py

```

Once initialized, open `http://localhost:8501` in your web browser to chat with the bot!

---

## 📈 Key NLP Concepts Demonstrated

> 💡 **Core Mechanics:** The system relies on data-driven statistical inference rather than hardcoded string matching.

* **Text Standardization:** Global case-folding to decrease vector vocabulary variance and ensure case-insensitive matching.
* **Feature Extraction:** Transforming raw text strings into numeric matrices using **TF-IDF** (*Term Frequency-Inverse Document Frequency*).
* **Probability Scoring:** Implementing `predict_proba()` to evaluate prediction confidence thresholds, enabling graceful fallback handling for out-of-bounds queries.


---

### 🚀  How to Scale the Bot for Massive Datasets

When your dataset scales from 50 patterns to **50,000+ patterns**, your machine learning pipeline will change. Here is your roadmap for handling heavy, industrial-scale text data:

#### 1. Data Storage (Ditch JSON ──► Move to Databases)
* **The Problem:** Reading a massive `intents.json` loads the entire file into your system RAM at once, which will crash your application if the dataset grows into gigabytes.
* **The Scale Solution:** Store patterns and responses in a structured SQL database (like **PostgreSQL**) or a NoSQL database (like **MongoDB**). You can then pull data using paginated queries or directly pipeline it into your model.

#### 2. Vectorization (Ditch Count/TF-IDF ──► Move to Dense Word Embeddings)
* **The Problem:** If you have 20,000 unique words, your TF-IDF matrix becomes 20,000 columns wide. Most rows will just be a bunch of zeros (`0.0`). This creates a "Sparse Matrix" which wastes memory and slows down training.
* **The Scale Solution:** Switch to pre-trained dense **Word Embeddings** like Word2Vec, GloVe, or Hugging Face **Transformers (BERT/Gemma)**. These condense any size word or sentence into a fixed matrix (e.g., exactly 768 numbers wide) that captures contextual meanings.

#### 3. Classification Algorithm Upgrade
* While `LogisticRegression` is fast, massive multi-class intent datasets (hundreds of tags) perform much better on deep learning frameworks. You can swap Scikit-Learn out for a **TensorFlow/Keras** Sequential Neural Network containing an `Embedding Layer` followed by dense layers, using a `Softmax` output function to handle high-volume intent distributions.


