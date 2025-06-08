# 🔍 NLP RNN Sentiment Analysis – IMDB Movie Reviews

## 🎯 Business Objective
The goal of this project is to classify IMDB movie reviews as **Positive** or **Negative** using Recurrent Neural Network (RNN) variants. It simulates a real-world scenario where businesses, content platforms, or review aggregators need to analyze customer sentiment at scale to improve user experience, identify trends, or moderate content.

---

## 📚 Dataset
- **Source**: IMDB movie review dataset (Keras built-in)
- **Type**: Binary Sentiment Classification
- **Instances**: 50,000 (25,000 training, 25,000 test)
- **Classes**: `0` - Negative, `1` - Positive
- **Preprocessed**: Encoded as sequences of integers

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Load IMDB dataset using Keras
   - Limit vocabulary size to top 10,000 frequent words
   - Pad sequences to a uniform length (500 tokens)

2. **Model Development**
   - Implemented three deep learning architectures:
     - Simple RNN
     - LSTM (Long Short-Term Memory)
     - GRU (Gated Recurrent Unit)
   - Models trained using binary cross-entropy loss and Adam optimizer

3. **Evaluation**
   - Performance compared using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Confusion Matrix
   - Visualized via Jupyter Notebook dashboard

4. **Deployment**
   - Interactive Streamlit app allows users to input text and view predictions
   - Dockerfile for containerization
   - GitHub Actions workflow for CI/CD integration

---

## 🧠 Models Used
| Model Type | Architecture | Key Layer Size |
|------------|--------------|----------------|
| RNN        | Embedding + SimpleRNN + Dense | 32 units |
| LSTM       | Embedding + LSTM + Dense      | 64 units |
| GRU        | Embedding + GRU + Dense       | 64 units |

---

## 💻 How to Run This Project

### 🔧 Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### 🏋️ Step 2: Train Models
```bash
python scripts/train_rnn.py
python scripts/train_lstm.py
python scripts/train_gru.py
```

### 📊 Step 3: Evaluate Models
Open the notebook:
```
notebooks/comparison_dashboard.ipynb
```

### 🌐 Step 4: Launch Streamlit App
```bash
streamlit run streamlit_app/app.py
```

### 🐳 Step 5: Run with Docker
```bash
docker build -t sentiment-app .
docker run -p 8501:8501 sentiment-app
```

---

## 📦 Folder Structure
```
nlp_rnn_sentiment_analysis/
│
├── data/                    # Placeholder for raw or future data
├── models/                  # Trained model weights (RNN, LSTM, GRU)
├── notebooks/
│   └── comparison_dashboard.ipynb
├── scripts/
│   ├── train_rnn.py
│   ├── train_lstm.py
│   └── train_gru.py
├── streamlit_app/
│   └── app.py
├── utils/
│   └── data_utils.py
├── .github/workflows/       # GitHub Actions CI setup
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📈 Output
- Classification reports printed for all 3 models
- Confusion matrices for visual analysis
- Sentiment predictions live via Streamlit

---

## 🧪 Future Enhancements
- Integrate Hugging Face transformers (e.g., BERT)
- Add explainability using SHAP or LIME
- Visualize attention in LSTM/GRU
- Streamlit enhancements: prediction confidence bar, batch input

---

## 🏷️ Tags
`NLP` `RNN` `LSTM` `GRU` `Sentiment Analysis` `Streamlit` `IMDB` `Docker` `AI Deployment` `GitHub Actions`

---

## 🚀 Advanced Models & Enhancements

### 🔁 Bidirectional RNNs
- **Bidirectional LSTM**: Captures both forward and backward context for improved sentiment detection.
- **Bidirectional GRU**: Lighter alternative to LSTM with similar performance.
- Both models trained with frozen **GloVe 100D word embeddings**.

### 🧠 Pre-trained Word Embeddings
- Integrated **GloVe (Global Vectors for Word Representation)**.
- Helps model understand semantic similarity (e.g., “awesome” and “great” have similar vectors).

### 🧪 Hyperparameter Tuning
- Implemented **KerasTuner Hyperband** search for optimal model configuration:
  - Tuned embedding output dimension
  - Tuned number of LSTM units

### 🆕 Additional Scripts
| File | Purpose |
|------|---------|
| `train_bidirectional_lstm_glove.py` | Train BiLSTM with GloVe |
| `train_bidirectional_gru_glove.py`  | Train BiGRU with GloVe  |
| `tune_lstm_kerastuner.py`          | Auto-tune LSTM using KerasTuner |

---

## 📈 Evaluation Metrics
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (via dashboard notebook)

> Use the Jupyter notebook `comparison_dashboard.ipynb` to compare results of basic and advanced models visually.
