import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def train_lstm_model():
    df = pd.read_csv("xag_data.csv")
    df = df[['close']].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    real_prices = scaler.inverse_transform(y_test)

    plt.figure(figsize=(14,6))
    plt.plot(real_prices, label='Giá thực tế')
    plt.plot(predicted_prices, label='Dự báo LSTM')
    plt.title("📈 Dự báo giá bạc (XAG/USD) bằng LSTM")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá bạc")
    plt.legend()
    plt.grid()
    plt.show()
