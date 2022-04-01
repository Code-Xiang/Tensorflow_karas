from posixpath import split
from sklearn.preprocessing import StandardScaler
import tushare as ts
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.layers import Input,Dense,LSTM,GRU,BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import openpyxl
# from openpyxl.workbook import Workbook
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus']=False

#路径目录
baseDir = 'RiceQuant/test' #当前目录
staticDir = os.path.join(baseDir,'Static') # 静态文件目录
resultDir = os.path.join(baseDir,'Result') # 结果文件目录

# 读取数据 通过tushare获取数据
ts.set_token('5cb5c9d3abea30da6b1392bd553e241bc4a4f45cdd70fc048d2328cb')
pro = ts.pro_api()
ts_code = '000001.SZ' # 平安股票
start_date = '2006-01-01'
end_date = '2020-01-01'
df = pro.daily(
    ts_code=ts_code,
    start_date=start_date,
    end_date=end_date
)

df = df.set_index(['trade_date'])
ColName = df.columns.values
use_cols = ['close','open','high','low','vol','amount']
data = df[use_cols]
# 标准化数据集
outputCol = ['close'] # 输出列
inputCol = ['close','high','low','open'] # 输入列
X = data[inputCol]
Y = data[outputCol]
xScaler = StandardScaler()
yScaler = StandardScaler()
X = xScaler.fit_transform(X) 
Y = yScaler.fit_transform(Y)


#按时间
timeStep = 5 # 输入天数
outStep = 1 # 输出天数
xAll = list()
yAll = list()
# 按时间步骤整理数据 输入数据尺寸是(timeStep, 5) 输出尺寸是(outSize)
for row in range(data.shape[0]-timeStep-outStep+1):
    x = X[row:row+timeStep]
    y = Y[row+timeStep:row+timeStep+outStep]
    xAll.append(x)
    yAll.append(y)
xAll = np.array(xAll).reshape(-1,timeStep,len(inputCol))
yAll = np.array(yAll).reshape(-1,outStep)


# 分成测试集，训练集
testRate = 0.2 #测试比例
splitIndex = int(xAll.shape[0]*(1-testRate)) # 所有数据条数*训练集比例 = 训练集条数
xTrain = xAll[:splitIndex]
xTest = xAll[splitIndex:]
yTrain = yAll[:splitIndex]
yTest = yAll[splitIndex:]

# 搭建简单的LSTM网络
'''
    搭建LSTM网络，激活函数为tanh
    timeStep:输入时间同步
    inputColNum: 输入列数d
    outStep: 输入时间步
    learnRate: 学习率
'''
def buildLSTM(timeStep, inputColNum,outStep,learnRate=1e-4):
    # 输入层
    inputLayer = Input(shape=(timeStep,inputColNum))
    # 中间层
    middle = LSTM(100,activation='tanh')(inputLayer)
    middle = Dense(100,activation='tanh')(middle)
    # 输出层 全连接
    outputLayer = Dense(outStep)(middle)
    # 建模
    model = Model(inputs=inputLayer,outputs=outputLayer)
    optimizer = Adam(lr=learnRate)
    model.compile(optimizer=optimizer,loss='mse') 
    model.summary()
    return model
#搭建LSTM
lstm = buildLSTM(timeStep=timeStep,inputColNum=len(inputCol),outStep=outStep,learnRate=1e-4)
#训练网络
epochs = 1000 #迭代次数
batchSize = 500 #批处理量
lstm.fit(xTrain,yTrain,epochs=epochs,verbose=0,batch_size=batchSize) 
#预测 测试集对比
yPredict = lstm.predict(xTest)
yPredict = yScaler.inverse_transform(yPredict)[:,0]
yTest = yScaler.inverse_transform(yTest)[:,0]
result = {'观测值':yTest,'预测值':yPredict}
result = pd.DataFrame(result)
result.index = data.index[timeStep+xTrain.shape[0]:result.shape[0]+timeStep+xTrain.shape[0]]
result.to_excel(resultDir+'\预测结果.xlsx')
result.head()
mae = MAE(result['观测值'],result['预测值'])
print('模型测试集MAE',mae)
#可视化
fig,ax = plt.subplots(1,1)
ax.plot(result.index,result['预测值'],label='预测值')
ax.plot(result.index,result['观测值'],label='观测值')
ax.set_title('LSTM预测效果，MAE：%2f'%mae)
ax.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
fig.savefig(resultDir+'/预测折线图.png',dpi=500)
