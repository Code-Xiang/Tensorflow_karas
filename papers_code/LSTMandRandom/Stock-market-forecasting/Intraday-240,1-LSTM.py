import pandas as pd
import numpy as np
import random
import time
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from Statistics import Statistics

import tensorflow as tf
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dropout,Dense,Input,add,LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import optimizers
import warnings
warnings.filterwarnings("ignore")

import os
SEED = 9
os.environ['PYTHONHASHSEED']=str(SEED)

# print('random.seed(SEED):\n',np.random.seed(SEED))
random.seed(SEED)
np.random.seed(SEED)

SP500_df = pd.read_csv('data/SPXconst.csv')
all_companies = list(set(SP500_df.values.flatten()))
all_companies.remove(np.nan)

constituents = {'-'.join(col.split('/')[::-1]):set(SP500_df[col].dropna()) 
                for col in SP500_df.columns}
constituents_train = {} 
for test_year in range(1993,2016):
    months = [str(t)+'-0'+str(m) if m<10 else str(t)+'-'+str(m) 
              for t in range(test_year-3,test_year) for m in range(1,13)]
    constituents_train[test_year] = [list(constituents[m]) for m in months]
    constituents_train[test_year] = set([i for sublist in constituents_train[test_year] 
                                         for i in sublist])
# 1990-01 1990-12
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

def cpuLSTM():
    inputs = Input(shape=(240,1))
    x = LSTM(25,activation='tanh', recurrent_activation='sigmoid',
                   input_shape=(240,1),
                   dropout=0.1,recurrent_dropout=0.1)(inputs)
    outputs = Dense(2,activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(),
                          metrics=['accuracy'])
    model.summary()
    return model    

def callbacks_req(model_type='LSTM'):
    csv_logger = CSVLogger(model_folder+'/training-log-'+model_type+'-'+str(test_year)+'.csv')
    filepath = model_folder+"/model-" + model_type + '-' + str(test_year) + "-E{epoch:02d}.h5"
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',save_best_only=False, save_freq=1)
    earlyStopping = EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True)
    # monitor:监控的数据接口
    # 因为monitor='val_loss'，所以mode要min
    # patience 能够容忍多少个epoch内都没有improvement
    # restore_best_weights：是否从具有检测数据的最佳值的时期恢复模型权重。如果为False，则使用在训练的最后一步获得的模型权重
    return [csv_logger,earlyStopping,model_checkpoint]
#  models-Intraday-240-1-LSTM/training-log-LSTM-1990-01.csv
#  models-Intraday-240-1-LSTM/model-training-log-LSTM-1990-01-E{day}.h5
    arr = np.array(np.split(arr,3,axis=1))
    arr = np.swapaxes(arr,0,1)
    arr = np.swapaxes(arr,1,2)
    return arr



def trained(filename,train_data,test_data):
    model = load_model(filename)
    dates = list(set(test_data[:,0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:,0]==day]
        test_d = np.reshape(test_d[:,2:-2],(len(test_d),240,1))
        predictions[day] = model.predict(test_d)[:,1]
    return model,predictions     

      
def create_label(df_open,df_close,perc=[0.5,0.5]):
    if not np.all(df_close.iloc[:,0]==df_open.iloc[:,0]):
        print('Date Index issue')
        return
    perc = [0.]+list(np.cumsum(perc))
    label = (df_close.iloc[:,1:]/df_open.iloc[:,1:]-1).apply(
            lambda x: pd.qcut(x.rank(method='first'),perc,labels=False), axis=1)
    return label[1:]

def create_stock_data(df_open,df_close,st,m=240):
    st_data = pd.DataFrame([])
    st_data['Date'] = list(df_close['Date'])
    st_data['Name'] = [st]*len(st_data)
    daily_change = df_close[st]/df_open[st]-1
    for k in range(m)[::-1]:
        st_data['IntraR'+str(k)] = daily_change.shift(k)

    st_data['IntraR-future'] = daily_change.shift(-1)  # 将后一天赋值给当前的日期  
    st_data['label'] = list(label[st])+[np.nan] #最后一个加一个nan
    st_data['Month'] = list(df_close['Date'].str[:-3]) # 去掉后面的天，留月份
    st_data = st_data.dropna()
    
    trade_year = st_data['Month'].str[:4] # 取年份
    st_data = st_data.drop(columns=['Month'])
    st_train_data = st_data[trade_year<str(test_year)] # 交易年份小于测试年份的都是训练年份
    st_test_data = st_data[trade_year==str(test_year)] # 交易年份是测试年份的则是测试年份
    return np.array(st_train_data),np.array(st_test_data) 

def scalar_normalize(train_data,test_data):
    scaler = RobustScaler()
    scaler.fit(train_data[:,2:-2])
    train_data[:,2:-2] = scaler.transform(train_data[:,2:-2])
    test_data[:,2:-2] = scaler.transform(test_data[:,2:-2])

def trainer(train_data,test_data,model_type='LSTM'):
    np.random.shuffle(train_data) # 打乱训练数据
    train_x,train_y,train_ret = train_data[:,2:-2],train_data[:,-1],train_data[:,-2]
    train_x = np.reshape(train_x,(len(train_x),240,1)).astype(np.float32)

    train_y = np.reshape(train_y,(-1, 1))
    train_ret = np.reshape(train_ret,(-1, 1))
    enc = OneHotEncoder(handle_unknown='ignore') # 它可以实现将分类特征的每个元素转换成为一个可以用来计算的值
    enc.fit(train_y)
    enc_y = enc.transform(train_y).toarray()
    train_ret = np.hstack((np.zeros((len(train_data),1)),train_ret))  # hstack将参数元组的元素组按水平方向进行叠加

    if model_type == 'LSTM':
        model = cpuLSTM()
    else:
        return
    callbacks = callbacks_req(model_type)
    
    model.fit(train_x,
              enc_y,
              epochs=1000,
              validation_split=0.2,
              callbacks=callbacks,
              batch_size=512
              )
    
    dates = list(set(test_data[:,0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:,0]==day]
        test_d = np.reshape(test_d[:,2:-2], (len(test_d),240,1))
        test_d = test_d.astype(np.float32)
        predictions[day] = model.predict(test_d)[:,1]
        # model.predict 返回值：每个测试集的所预测的各个类别的概率
    return model,predictions

def simulate(test_data,predictions):
    rets = pd.DataFrame([],columns=['Long','Short'])
    k = 10
    for day in sorted(predictions.keys()):
        preds = predictions[day]
        test_returns = test_data[test_data[:,0]==day][:,-2]
        top_preds = predictions[day].argsort()[-k:][::-1] 
        # argsort(),表示对数据进行从小到大进行排序，返回数据的索引值
        # [::-1] 表示对数组a进行从大到小排序，返回索引值
        trans_long = test_returns[top_preds]
        worst_preds = predictions[day].argsort()[:k][::-1] 
        trans_short = -test_returns[worst_preds]
        rets.loc[day] = [np.mean(trans_long),np.mean(trans_short)] 
    print('Result : ',rets.mean())  
    return rets      
# 生成目录
model_folder = 'models-Intraday-240-1-LSTM'
result_folder = 'results-Intraday-240-1-LSTM'
for directory in [model_folder,result_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

for test_year in range(1993,2020):
    
    print('-'*40)
    print(test_year)
    print('-'*40)
    
    filename = 'data/Open-'+str(test_year-3)+'.csv'
    df_open = pd.read_csv(filename)
    filename = 'data/Close-'+str(test_year-3)+'.csv'
    df_close = pd.read_csv(filename)
    colums_open = df_open.columns
    df_open[colums_open] = df_open[colums_open].replace(0,np.nan)
    colums_close = df_open.columns
    df_open[colums_close] = df_open[colums_close].replace(0,np.nan)
    label = create_label(df_open,df_close)
    stock_names = sorted(list(constituents[str(test_year-1)+'-12']))
    train_data,test_data = [],[]

    start = time.time()
    for st in stock_names:
        st_train_data,st_test_data = create_stock_data(df_open,df_close,st)
        train_data.append(st_train_data)
        test_data.append(st_test_data)
        
    train_data = np.concatenate([x for x in train_data])
    test_data = np.concatenate([x for x in test_data])
    # 标准化
    scalar_normalize(train_data,test_data)
    print(train_data.shape,test_data.shape,time.time()-start)
    
    model,predictions = trainer(train_data,test_data)
    # 模拟？
    returns = simulate(test_data,predictions)
    returns.to_csv(result_folder+'/avg_daily_rets-'+str(test_year)+'.csv')
    
    result = Statistics(returns.sum(axis=1))
    print('\nAverage returns prior to transaction charges')
    result.report()
    
    with open(result_folder+"/avg_returns.txt", "a") as myfile:
        res = '-'*30 + '\n'
        res += str(test_year) + '\n'
        res += 'Mean = ' + str(result.mean()) + '\n'
        res += 'Sharpe = '+str(result.sharpe()) + '\n'
        res += 'std = '+str(result.std()) + '\n'
        res += 'MDD = '+str(result.MDD()) + '\n'
        res += '-'*30 + '\n'
        myfile.write(res)
        

