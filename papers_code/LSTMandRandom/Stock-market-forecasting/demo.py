from unittest import result
from wsgiref import validate
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dropout,Dense,Input,add
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from tensorflow.keras import optimizers
SP500_df = pd.read_csv('papers_code/LSTMandRandom/Stock-market-forecasting/data/SPXconst.csv')
all_companies = list(set(SP500_df.values.flatten()))
all_companies.remove(np.nan)

constituents = {'-'.join(col.split('/')[::-1]):set(SP500_df[col].dropna()) 
                for col in SP500_df.columns}

constituents_train = {} 
for test_year in range(1993,2016):
    months = [str(t)+'-0'+str(m) if m<10 else str(t)+'-'+str(m) for t in range(test_year-3,test_year) for m in range(1,13)]
    constituents_train[test_year] = [list(constituents[m]) for m in months]
    constituents_train[test_year] = set([i for sublist in constituents_train[test_year] 
                                         for i in sublist])
# print('constituents_train:\n',constituents_train[2015])
# for test_year in range(1993,2020):
test_year = 1993
print('='*50)
print(test_year)
print('='*50)

model_folder = 'papers_code/LSTMandRandom/Stock-market-forecasting/demo/models-Intraday-240-1-LSTM'
result_folder ='papers_code/LSTMandRandom/Stock-market-forecasting/demo/results-Intraday-240-1-LSTM'

filename = 'papers_code/LSTMandRandom/Stock-market-forecasting/data/Open-'+str(test_year-3)+'.csv'
df_open = pd.read_csv(filename)
filename = 'papers_code/LSTMandRandom/Stock-market-forecasting/data/Close-'+str(test_year-3)+'.csv'
df_close = pd.read_csv(filename)
perc = [0.5,0.5]
perc = [0.]+list(np.cumsum(perc))
# print('perc\n',perc)
# print('df_close.iloc[:,1:]:\n',df_close.iloc[:,1:])
# print('df_open.iloc[:,1:]:\n',df_open.iloc[:,1:])
# print('df_close.iloc[:,1:]/df_open.iloc[:,1:]:\n',df_close.iloc[:,1:]/df_open.iloc[:,1:])

label_lb = (df_close.iloc[:,1:]/df_open.iloc[:,1:]-1).apply(
            lambda x: pd.qcut(x.rank(method='first'),perc,labels=False), axis=1)
label = label_lb[1:]
stock_names = sorted(list(constituents[str(test_year-1)+'-12']))
train_data,test_data = [],[]
start = time.time()
print('stock_names:\n',stock_names[1])
# st = stock_names[1]
def reshaper(arr):
    arr = np.array(np.split(arr,3,axis=1))
    arr = np.swapaxes(arr,0,1)
    arr = np.swapaxes(arr,1,2)
    return arr.astype(np.float32)
    
def callbacks_req(model_type='LSTM'):
    csv_logger = CSVLogger(model_folder+'/training-log-'+model_type+'-'+str(test_year)+'.csv')
    filepath = model_folder+"/model-"+model_type+'-'+str(test_year)+"-E{epoch:02d}.h5"
    model_checkpoint = ModelCheckpoint(filepath,monitor="val_loss",save_best_only=False,save_freq=1)
    earlyStopping = EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True)
    return [csv_logger,earlyStopping,model_checkpoint]
def makeLSTM():
    inputs = Input(shape=(240,1))
    # return_sequences默认为false,此时返回一个hidden state的值，如果input数据包含多个时间步，则这个
    # hidden state 最后一个时间步的结果
    x = CuDNNLSTM(25,return_sequences=False)(inputs)
    x = Dropout(0.1)(x)
    outputs = Dense(2,activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(),
                        metrics=['accuracy'])
    model.summary()
    return model
def create_stock_data(df_open,df_close,st,m=240):
    st_data = pd.DataFrame([])
    st_data['Date'] = list(df_close['Date'])
    st_data['Name'] = [st]*len(st_data)
    daily_change = df_close[st]/df_open[st]-1
    # print('daily_change:\n',daily_change)
    # print('daily_change.shift(k):\n',daily_change.shift(-1))
    for k in range(m)[::-1]:
        st_data['IntraR'+str(k)] = daily_change.shift(k)
    st_data['IntraR-future'] = daily_change.shift(-1)
    st_data['label'] = list(label[st])+[np.nan] 
    st_data['Month'] = list(df_close['Date'].str[:-3])
    st_data = st_data.dropna()
    trade_year = st_data['Month'].str[:4]
    st_data = st_data.drop(columns=['Month'])
    st_train_data = st_data[trade_year<str(test_year)]
    st_test_data = st_data[trade_year==str(test_year)]
    return np.array(st_train_data),np.array(st_test_data) 
def simulate(test_data,predictions):
    rets = pd.DataFrame([],columns=['Long','Short'])
    k = 10
    for day in sorted():
        preds = predictions[day]    
for st in stock_names:
    st_train_data,st_test_data = create_stock_data(df_open,df_close,st)
    train_data.append(st_train_data)
    test_data.append(st_test_data)
train_data = np.concatenate([x for x in train_data])
test_data = np.concatenate([x for x in test_data])


# 标准化
scaler = RobustScaler()
scaler.fit(train_data[:,2:-2])
train_data[:,2:-2] = scaler.transform(train_data[:,2:-2])
test_data[:,2:-2] = scaler.transform(test_data[:,2:-2])
# trainer

np.random.shuffle(train_data)
print('train_data_2:\n',train_data)
train_x,train_y,train_ret = train_data[:,2:-2],train_data[:,-1],train_data[:,-2] # train_y: [0.0 1.0 0.0 ... 1.0 0.0 0.0]
train_x = np.reshape(train_x,(len(train_x),240,1)).astype(np.float32)
train_y = np.reshape(train_y,(-1, 1))
train_ret = np.reshape(train_ret,(-1, 1))
enc = OneHotEncoder(handle_unknown='ignore') # 它可以实现将分类特征的每个元素转换成为一个可以用来计算的值
enc.fit(train_y)
enc_y = enc.transform(train_y).toarray()
train_ret = np.hstack((np.zeros((len(train_data),1)),train_ret)) 

print('train_x:\n',train_x)
print('train_y:\n',train_y)
print('train_ret:\n',train_ret)
print('train_ret:\n',train_ret.shape)
model_type = 'LSTM'
model = makeLSTM()
callbacks = callbacks_req(model_type)
# model.fit(
#     train_x,
#     train_y,
#     epochs= 1000,
#     validation_split=0.2,
#     callbacks=callbacks,
#     batch_size=512
# )
dates = list(set(test_data[:,0]))
predictions = {}
for day in dates:
    test_d = test_data[test_data[:,0]==day]
    test_d = reshaper(test_d[:,2:-2])
    predictions[day] = model.predict(test_d)[:,1]
print('predictions:\n',predictions)

# return model,predictions
# print('train_data:\n',train_data)
# print('test_data:\n',test_data)
# print(train_data.shape,test_data.shape,time.time()-start)
# print('df_close["Date"]:\n',df_close['Date'])
# print('df_close["Date"].str[:-3]:\n',df_close['Date'].str[:-3])
# print('df_close[st]:\n',df_close[st]kk)
# print('daily_change:\n',daily_change)
# for st in stock_names:

# print('lambda x:\n', x)
# lambda x: pd.qcut(x.rank(method='first'))
    # if not np.all(df_close.iloc[:,1:]/df_open.iloc[:,1:]-1).apply()
    #     return
    # label = create_label(df_open,df_close)

