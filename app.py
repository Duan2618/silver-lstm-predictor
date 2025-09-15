import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ta

# --- Sidebar ---
st.set_page_config(page_title="Dự báo giá & phân tích kỹ thuật", layout="wide")
st.sidebar.title("⚙️ Cài đặt")
asset = st.sidebar.selectbox("Chọn tài sản", ["XAG/USD", "XAU/USD", "BTC/USD"])
seq_length = st.sidebar.slider("Chuỗi thời gian (LSTM)", 30, 100, 60)
n_predict = st.sidebar.slider("Số bước dự báo", 1, 10, 1)

# --- Load dữ liệu ---
file_map = {"XAG/USD": "xag_data.csv", "XAU/USD": "xau_data.csv", "BTC/USD": "btc_data.csv"}
df = pd.read_csv(file_map[asset])
df = df[['close']].dropna()
df.columns = ["Close"]

# --- Phân tích kỹ thuật ---
df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
macd = ta.trend.MACD(df["Close"])
df["MACD"] = macd.macd()
df["MACD_signal"] = macd.macd_signal()
bb = ta.volatility.BollingerBands(df["Close"])
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()

# --- Hiển thị biểu đồ kỹ thuật ---
st.subheader(f"📈 Phân tích kỹ thuật: {asset}")
fig = go.Figure()
fig.add_trace(go.Scatter(y=df["Close"], name="Giá", line=dict(color="blue")))
fig.add_trace(go.Scatter(y=df["RSI"], name="RSI", line=dict(color="orange")))
fig.add_trace(go.Scatter(y=df["MACD"], name="MACD", line=dict(color="green")))
fig.add_trace(go.Scatter(y=df["MACD_signal"], name="MACD Signal", line=dict(color="red")))
fig.add_trace(go.Scatter(y=df["BB_upper"], name="BB Upper", line=dict(color="gray", dash="dot")))
fig.add_trace(go.Scatter(y=df["BB_lower"], name="BB Lower", line=dict(color="gray", dash="dot")))
fig.update_layout(height=500, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# --- Dự báo bằng LSTM ---
st.subheader(f"🤖 Dự báo giá tiếp theo: {asset}")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[["Close"]])

X, y = [], []
for i in range(seq_length, len(scaled_data)):
    X.append(scaled_data[i-seq_length:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# Dự báo
last_sequence = scaled_data[-seq_length:]
forecast = []
input_seq = last_sequence.reshape(1, seq_length, 1)
for _ in range(n_predict):
    pred = model.predict(input_seq)[0][0]
    forecast.append(pred)
    input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

forecast_prices = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
st.write(f"📌 Dự báo {n_predict} bước tiếp theo:")
st.line_chart(forecast_prices)

