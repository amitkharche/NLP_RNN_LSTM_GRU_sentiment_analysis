import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 10000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

def model_builder(hp):
    model = Sequential()
    model.add(Embedding(max_features, hp.Int('embedding_output', min_value=32, max_value=128, step=32)))
    model.add(LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=5,
                     factor=3,
                     directory='kerastuner_dir',
                     project_name='lstm_tuning')

tuner.search(x_train, y_train, epochs=5, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best embedding output: {best_hps.get('embedding_output')}")
print(f"Best LSTM units: {best_hps.get('lstm_units')}")
