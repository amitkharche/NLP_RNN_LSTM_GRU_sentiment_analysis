from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(max_features=10000, maxlen=500):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    return x_train, y_train, x_test, y_test
