from get_xau_data import fetch_xau_data
from tvdatafeed import Interval

# Tải dữ liệu XAU/USD
fetch_xau_data(interval=Interval.in_1_hour, bars=500)
