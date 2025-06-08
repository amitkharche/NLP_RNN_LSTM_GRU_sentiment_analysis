import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Model paths dictionary
model_paths = {
    "Simple RNN": "../models/rnn_model.h5",
    "LSTM": "../models/lstm_model.h5",
    "GRU": "../models/gru_model.h5",
    "Bidirectional LSTM (GloVe)": "../models/bidirectional_lstm_glove.h5",
    "Bidirectional GRU (GloVe)": "../models/bidirectional_gru_glove.h5"
}

# Load IMDB word index
word_index = imdb.get_word_index()

def encode_review(text, maxlen=500):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 = <UNK>
    return pad_sequences([encoded], maxlen=maxlen)

# Streamlit UI
st.title("🎬 Sentiment Prediction on IMDB Reviews")
text_input = st.text_area("Enter a movie review:")
model_choice = st.selectbox("Select Model:", list(model_paths.keys()))

if st.button("Predict"):
    if text_input.strip():
        model = load_model(model_paths[model_choice])
        encoded = encode_review(text_input)
        prediction = model.predict(encoded)[0][0]
        sentiment = "👍 Positive" if prediction > 0.5 else "👎 Negative"
        st.write(f"### Prediction: {sentiment}")
        st.write(f"Confidence: {prediction:.4f}")
    else:
        st.warning("Please enter a review to analyze.")
