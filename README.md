# 📈 Dự báo giá bạc, vàng và Bitcoin bằng LSTM + TradingView + Streamlit

Dự án này sử dụng mô hình học sâu LSTM để dự báo giá của các tài sản tài chính như bạc (XAG/USD), vàng (XAU/USD) và Bitcoin (BTC/USD). Dữ liệu được lấy trực tiếp từ TradingView thông qua thư viện `tvdatafeed`, kết hợp với phân tích kỹ thuật như RSI, MACD, Bollinger Bands. Giao diện Streamlit giúp bạn tương tác, trực quan hóa và theo dõi dự báo theo thời gian thực.

---

## 🚀 Tính năng nổi bật

- 🤖 Dự báo giá bằng mô hình LSTM
- 📊 Phân tích kỹ thuật: RSI, MACD, Bollinger Bands
- 💰 Hỗ trợ đa tài sản: XAG/USD, XAU/USD, BTC/USD
- 🖥 Giao diện Streamlit trực quan, dễ sử dụng
- 🔌 Tích hợp dữ liệu từ TradingView qua `tvdatafeed`

---

## 📁 Cấu trúc dự án

silver-lstm-predictor/ ├── app.py # Dashboard Streamlit ├── get_xag_data.py # Tải dữ liệu từ TradingView ├── lstm_model.py # Mô hình LSTM huấn luyện & dự báo ├── requirements.txt # Thư viện cần thiết ├── xag_data.csv # Dữ liệu bạc ├── xau_data.csv # Dữ liệu vàng ├── btc_data.csv # Dữ liệu Bitcoin

---

## ⚙️ Cài đặt & chạy dự án

### 1. Clone repo & cài thư viện

```bash
git clone https://github.com/yourusername/silver-lstm-predictor.git
cd silver-lstm-predictor
pip install -r requirements.txt

