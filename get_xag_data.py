from tvdatafeed import TvDatafeed, Interval
import pandas as pd

def fetch_xag_data(username=None, password=None, interval=Interval.in_daily, bars=1000):
    tv = TvDatafeed(username=username, password=password)
    df = tv.get_hist(symbol='XAGUSD', exchange='OANDA', interval=interval, n_bars=bars)
    df.to_csv("xag_data.csv")
    print("✅ Dữ liệu XAG/USD đã được lưu vào xag_data.csv")
    return df
