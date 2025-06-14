
---

# 🔍 NLP RNN Sentiment Analysis – IMDB Movie Reviews

## 🎯 Business Objective

This project classifies IMDB movie reviews as **Positive** or **Negative** using various Recurrent Neural Network (RNN) architectures. It simulates real-world sentiment analysis for platforms aiming to monitor user opinion at scale—enhancing recommendation systems, moderation, and marketing analytics.

---

## 📚 Dataset

* **Source**: IMDB Movie Review Dataset (Keras built-in)
* **Task**: Binary Sentiment Classification
* **Size**: 50,000 reviews (25k train / 25k test)
* **Classes**: `0` - Negative, `1` - Positive
* **Format**: Preprocessed and integer-encoded word sequences

---

## ⚙️ Methodology

1. **Data Preprocessing**

   * Load the IMDB dataset with top 10,000 frequent words.
   * Pad sequences to a fixed length of 500 tokens.

2. **Model Development**

   * Implemented the following deep learning models:

     * Simple RNN
     * LSTM (Long Short-Term Memory)
     * GRU (Gated Recurrent Unit)
     * Bidirectional LSTM (GloVe)
     * Bidirectional GRU (GloVe)
     * Tuned LSTM using KerasTuner
   * Trained using **binary cross-entropy** and the **Adam optimizer**

3. **Evaluation**

   * Used metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
   * Results visualized in Jupyter dashboard

4. **Deployment**

   * **Streamlit** app for real-time prediction
   * **Docker** for containerized deployment
   * **GitHub Actions** for CI/CD automation

---

## 🧠 Models Used

| Model Type             | Architecture                   | Key Layers/Parameters   |
| ---------------------- | ------------------------------ | ----------------------- |
| Simple RNN             | Embedding + SimpleRNN + Dense  | 32 RNN units            |
| LSTM                   | Embedding + LSTM + Dense       | 64 LSTM units           |
| GRU                    | Embedding + GRU + Dense        | 64 GRU units            |
| BiLSTM (GloVe)         | GloVe + BiLSTM + Dense         | 64 BiLSTM units, frozen |
| BiGRU (GloVe)          | GloVe + BiGRU + Dense          | 64 BiGRU units, frozen  |
| Best LSTM (KerasTuner) | Tuned Embedding + LSTM + Dense | Best via Hyperband      |

---

## 💻 How to Run This Project

### 🔧 Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🗂️ Step 2: Download GloVe Embeddings

Download GloVe (100D) from:
[http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

Extract and place `glove.6B.100d.txt` in the `data/` directory.

---

### 🏋️ Step 3: Train Models

```bash
python scripts/train_rnn.py
python scripts/train_lstm.py
python scripts/train_gru.py
python scripts/train_bidirectional_lstm_glove.py
python scripts/train_bidirectional_gru_glove.py
python scripts/tune_lstm_kerastuner.py
```

---

### 📊 Step 4: Evaluate Models

Open Jupyter Notebook:

```bash
notebooks/comparison_dashboard.ipynb
```

---

### 🌐 Step 5: Launch Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

### 🐳 Step 6: Run with Docker

```bash
docker build -t sentiment-app .
docker run -p 8501:8501 sentiment-app
```

---

## 📦 Folder Structure

```
nlp_rnn_sentiment_analysis/
│
├── data/                      # GloVe file goes here
├── models/                    # Trained model weights
├── notebooks/
│   └── comparison_dashboard.ipynb
├── scripts/
│   ├── train_rnn.py
│   ├── train_lstm.py
│   ├── train_gru.py
│   ├── train_bidirectional_lstm_glove.py
│   ├── train_bidirectional_gru_glove.py
│   └── tune_lstm_kerastuner.py
├── streamlit_app/
│   └── app.py
├── utils/
│   └── data_utils.py
├── .github/workflows/         # CI/CD pipeline
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📈 Evaluation Output

* Printed classification reports
* Confusion matrix for each model
* Live predictions through the Streamlit interface

---

## 🔮 Future Enhancements

* Integrate BERT using Hugging Face Transformers
* Explainability tools like **LIME** or **SHAP**
* Visualize LSTM/GRU attention weights
* Enhance Streamlit with:

  * Confidence visualization
  * Batch prediction input

---

## 🚀 Advanced Models & Techniques

### 🔁 Bidirectional RNNs

* **BiLSTM & BiGRU** capture past and future context.
* Powered by **pre-trained GloVe (100D)** word vectors.
* Embeddings are frozen for stable semantics.

### 🧠 Word Embeddings (GloVe)

* GloVe vectors help understand word similarity (e.g., "good" ≈ "excellent")
* Source: Stanford NLP ([Download link](http://nlp.stanford.edu/data/glove.6B.zip))

### 🎯 Hyperparameter Tuning

* Used **KerasTuner Hyperband** to optimize:

  * Embedding output dimension
  * Number of LSTM units

---

## 📈 Evaluation Metrics Used

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

> 📊 Compare results in: `notebooks/comparison_dashboard.ipynb`

---

## 📎 Repository

Clone this project:

```bash
git clone https://github.com/amitkharche/NLP_RNN_LSTM_GRU_sentiment_analysis
cd NLP_RNN_LSTM_GRU_sentiment_analysis
```

---

## 🏷️ Tags

`NLP` `Sentiment Analysis` `IMDB` `RNN` `LSTM` `GRU`
`Bidirectional` `GloVe` `KerasTuner` `Streamlit` `Docker`
`Deep Learning` `Text Classification` `AI Deployment`

---

## Contact

If you have questions or want to collaborate, feel free to connect with me on
- [LinkedIn](https://www.linkedin.com/in/amit-kharche)  
- [Medium](https://medium.com/@amitkharche14)  
- [GitHub](https://github.com/amitkharche)