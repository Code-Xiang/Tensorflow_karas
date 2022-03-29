import tushare as ts
ts_code = '000001.SZ' #第一号股票，平安
ts_code = '002069.SZ' #神奇的獐子岛
start_date = '2006-01-01'
end_date = '2020-01-01'
df = ts.pro.daily(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date)