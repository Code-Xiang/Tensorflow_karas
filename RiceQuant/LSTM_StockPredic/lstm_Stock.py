import numpy as np
import pandas as pd

# 目前riceQuant只支持中国市场
# import rqdatac
# from sqlalchemy import false
# rqdatac.init()
# order_book_ids = 'BSE'
# df = rqdatac.get_price(order_book_ids, start_date='2010-06-01', end_date='2020-01-01', frequency='1d')
# df.to_csv('.\RiceQuant\LSTM_StockPredic\BSE.csv')

import tushare as ts
ts.set_token('5cb5c9d3abea30da6b1392bd553e241bc4a4f45cdd70fc048d2328cb')
pro = ts.pro_api()
ts_code = 'AAPL' 
# ts_code = '002069.SZ' # 神奇的獐子岛？
start_date = '20100601'
end_date = '20200101'
df = pro.us_daily(
    ts_code=ts_code,
    start_date=start_date,
    end_date=end_date
)
# df1 = pro.us_basic()
# df1.to_csv('RiceQuant\LSTM_StockPredic\美股.csv')
print('df:/n',df)
df.to_csv('RiceQuant\LSTM_StockPredic\AAPL.csv')
def transform_dataset(train_set, test_set,y_train,y_test,n_input,n_output):
        # vstack竖直堆叠数组
        all_data = np.vstack((train_set, test_set))
        y_set = np.vstack((y_train, y_test))[:,0]
        X = np.empty((1, n_input, all_data.shape[1]))
        y = np.empty((1, n_output))
        for i in range(all_data.shape[0] - n_input - n_output):
            X_sample = all_data[i:i + n_input, :]
            y_sample = y_set[i + n_input:i + n_input + n_output]
            if i == 0:
                X[i] = X_sample
                y[i] = y_sample
            else:
                X = np.append(X, np.array([X_sample]), axis=0)
                y = np.append(y, np.array([y_sample.T]), axis=0)
        train_X = X[:train_set.shape[0] - n_input, :, :]
        train_y = y[:train_set.shape[0] - n_input, :]
        test_X = X[train_set.shape[0] -
                n_input:all_data.shape[0] -
                n_input -
                n_output, :, :]
        test_y = y[train_set.shape[0] -
                n_input:all_data.shape[0] -
                n_input -
                n_output, :]
        return train_X, train_y, test_X, test_y


