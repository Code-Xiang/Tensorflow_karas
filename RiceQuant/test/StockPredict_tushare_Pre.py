import pandas as pd
from sqlalchemy import all_
import tushare as ts
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import regularizers
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

use_cols = ['close','open','high','pre_close','vol','amount']
train_raw = df[use_cols][:2000]
test_raw = df[use_cols][2000:]
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
n_input = 63 # 过去3个月数据来预测未来1天
n_output = 1

train_X, train_y, test_X, test_y = transform_dataset(
    training_scaled, test_scaled,y_train_scaled,y_test_scaled, n_input, n_output)
# 模型准备和训练
# train_X.shape:(1, 63, 7) train_y.shape:(1, 1)
n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
# create a model      
model = Sequential() 
model.add(LSTM(10, input_shape=(n_timesteps, n_features),kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),return_sequences=False))     
# model.add(LSTM(20, input_shape=(n_timesteps, n_features),kernel_initializer='glorot_uniform',
#                kernel_regularizer=regularizers.l2(0.0)))

model.add(Dense(n_outputs,kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0)))

model.compile(optimizer='adam', loss='mae')
print(model.summary())

# 打乱样本
# train_X,train_y = shuffle(train_X,train_y,random_state=42)
# print('train_y:\n',train_y)
# plt.plot(train_y)

# plt.plot(history.history['loss'],
#          'b',
#          label='Training loss')
# plt.plot(history.history['val_loss'],
#          'r',
#          label='Validation loss')
# plt.legend(loc='upper right')
# plt.xlabel('Epochs')
# plt.show()
# Finalizing predictions
# 预测准确度
new_model = keras.models.load_model('path_to_my_model')
scaled_preds = new_model.predict(test_X)
test_preds = np.zeros_like(scaled_preds)
for i in range(scaled_preds.shape[1]):
    test_preds[:, i] = y_sc.inverse_transform(
        [scaled_preds[:, i]]).reshape(1, scaled_preds.shape[0])


test_preds_df = pd.DataFrame(
    test_preds, columns=[f'pred_{i+1}_step' for i in range(test_preds.shape[1])])
test_preds_df['true_value'] = test_raw.values[-len(test_preds):,0]
test_preds_df['naive_pred'] = test_raw.values[-len(test_preds) - 1:-1,0]

test_preds_df[['pred_1_step', 'true_value'
               ]].plot()
from sklearn.metrics import mean_absolute_error
err = mean_absolute_error(test_preds_df['pred_1_step'].values,test_preds_df['true_value'].values)
print(f'abs error for testset is {err}')