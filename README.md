# 🎭 Sentiment Analysis with NLTK

A production-ready NLP project for three-class sentiment classification
(Positive / Neutral / Negative) built with **Python**, **NLTK**, and
**scikit-learn**, complete with a Flask REST API.

---

## 📁 Project Structure

```
sentiment_analysis/
│
├── data/                        # Raw & processed CSV files
│   ├── raw_reviews.csv          # Generated / user-supplied dataset
│   └── processed_reviews.csv   # After preprocessing
│
├── models/                      # Serialised model artefacts
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── outputs/                     # Generated plots & visualisations
│   ├── label_distribution.png
│   ├── confusion_matrix.png
│   ├── cv_scores.png
│   ├── wordcloud.png
│   └── vader_distribution.png
│
├── src/                         # Core library
│   ├── __init__.py
│   ├── config.py                # All project constants & paths
│   ├── data_loader.py           # Load CSV or generate synthetic data
│   ├── preprocessor.py          # NLTK text cleaning pipeline
│   ├── feature_extractor.py     # TF-IDF + VADER scoring
│   ├── model.py                 # Train / evaluate / persist
│   └── visualizer.py           # All Matplotlib / Seaborn plots
│
├── tests/                       # Pytest unit tests
│   ├── __init__.py
│   ├── test_preprocessor.py
│   └── test_feature_and_model.py
│
├── notebooks/                   # Jupyter exploration (optional)
│
├── train.py                     # End-to-end training pipeline
├── predict.py                   # CLI inference script
├── app.py                       # Flask REST API
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / unzip the project
cd sentiment_analysis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1 — Train the model

```bash
python train.py
```

This will:
- Generate (or load) the review dataset
- Preprocess text with NLTK
- Extract TF-IDF features
- Train a Logistic Regression classifier with 5-fold CV
- Save the model and vectoriser under `models/`
- Generate five visualisation plots under `outputs/`

---

### 2 — Predict from the command line

```bash
# Single review
python predict.py "This product is absolutely fantastic!"

# Multiple reviews
python predict.py "Great quality." "Arrived broken." "It's okay I guess."
```

---

### 3 — Run the Flask API

```bash
python app.py
```

Then POST to `http://localhost:5000/predict`:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Amazing product!", "Terrible experience."]}'
```

Response:
```json
[
  {
    "text": "Amazing product!",
    "ml_label": "Positive",
    "ml_confidence": "94.3%",
    "vader_label": "Positive",
    "vader_compound": 0.5859
  },
  ...
]
```

---

### 4 — Run tests

```bash
pytest tests/ -v
```

---

## 🧠 NLP Pipeline

| Stage | Tool | Description |
|---|---|---|
| Cleaning | `re`, `string` | Lowercase, strip HTML/URLs/punctuation |
| Tokenisation | NLTK `word_tokenize` | Split into word tokens |
| Stop-word removal | NLTK `stopwords` | Remove high-frequency noise words |
| Lemmatisation | NLTK `WordNetLemmatizer` | Reduce words to base form |
| Feature extraction | `TfidfVectorizer` | Weighted bag-of-words (unigrams + bigrams) |
| Rule-based baseline | NLTK VADER | Lexicon-based compound score |
| Classifier | `LogisticRegression` | Multi-class ML model |

---

## 📊 Outputs

| File | Description |
|---|---|
| `label_distribution.png` | Bar chart of class counts |
| `confusion_matrix.png` | Heatmap of predictions vs actuals |
| `cv_scores.png` | Per-fold cross-validation accuracy |
| `wordcloud.png` | Most frequent tokens in corpus |
| `vader_distribution.png` | Histogram of VADER compound scores |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `nltk` | Tokenisation, stop-words, lemmatisation, VADER |
| `scikit-learn` | TF-IDF, Logistic Regression, evaluation |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `wordcloud` | Word frequency cloud |
| `flask` | REST API |
| `joblib` | Model serialisation |
