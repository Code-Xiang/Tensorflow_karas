from sqlalchemy import all_
import tushare as ts
import numpy as np
from sklearn.preprocessing import MinMaxScaler
ts.set_token('5cb5c9d3abea30da6b1392bd553e241bc4a4f45cdd70fc048d2328cb')
pro = ts.pro_api()

ts_code = '000001.SZ' # 平安股票
ts_code = '002069.SZ' # 神奇的獐子岛？
start_date = '2006-01-01'
end_date = '2020-01-01'
df = pro.daily(
    ts_code=ts_code,
    start_date=start_date,
    end_date=end_date
)
print('df:\n',df)
# 数据处理
# n_input为需要多少步历史数据，n_output为预测多少步未来数据
def transform_dataset(train_set, test_set,y_train,y_test,n_input,n_output):
    # vstack竖直堆叠数组
    all_data = np.vstack((train_set,test_set))
    y_set = np.vstack((y_train, y_test))
    X = np.empty((1,n_input,all_data.shape[1])) # all_data.shape[1] === 特征数
    y = np.empty((1,n_output))
    for i in range(all_data.shape[0]-n_input-n_output):
        X_sample = all_data[i:i+n_input,:]
        y_sample = y_set[i+n_input:i+n_input+n_output]
        if i == 0:
            X[i] = X_sample
            y[i] = y_sample
        else:
            X = np.append(X, np.array([X_sample]), axis=0)
            y = np.append(y, np.array([y_sample.T]), axis=0)
        train_X = X[:train_set.shape[0] - n_input,:, :]
        train_y = y[:train_set.shape[0] - n_input,:]
        test_X = X[train_set.shape[0]-
                   n_input:all_data.shape[0]-
                   n_input -
                   n_output,:,:]
        test_y = y[train_set.shape[0] -
                   n_input:all_data.shape[0]-
                   n_input -
                   n_output,:]
        return train_X, train_y,test_X, test_y

use_cols = ['trade_date','close','open','high','pre_close','vol','amount']
train_raw = df[use_cols][:2000]
print('train_raw:\n',train_raw)
test_raw = df[use_cols][2000:]
print('test_raw:\n',test_raw)
y_train_raw = train_raw[['close']]
y_test_raw = test_raw[['close']] 
# scale the data
# 归一化
sc = MinMaxScaler(feature_range=(0, 1))
y_sc = MinMaxScaler(feature_range=(0, 1))
training_scaled = sc.fit_transform(train_raw.values)
test_scaled = sc.transform(test_raw.values)
y_train_scaled = y_sc.fit_transform(y_train_raw.values)
y_test_scaled = y_sc.transform(y_test_raw.values)
n_input = 63
n_output = 1
train_X, train_y, test_X, test_y = transform_dataset(
    training_scaled, test_scaled,y_train_scaled,y_test_scaled, n_input, n_output)

# 模型准备和训练
