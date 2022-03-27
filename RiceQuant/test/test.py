# import imp
# from matplotlib.pyplot import close
from doctest import Example
import matplotlib.pyplot as plt
import rqdatac as rq
import pandas as pd
rq.init()
data = rq.get_price('000001.XSHE', start_date='2020-04-01', end_date='2021-04-03', adjust_type='none',expect_df=False)
# data = pd.read_csv('000001.csv')
# print(data.head()) 
# 数据归一化：归一化之后的数据服从正态分布
# X_std = (X - X.min(axis=0)) / (X.max(axis=0)-X.min(axis=0))
# X_scaled = X_std*(max-min)+min
from sklearn.preprocessing import MinMaxScaler
# 默认 feature_range=(0, 1) 0-1是标准的正态分布

# 获取收盘价
originData = data.iloc[:,3:4].astype('float32')
scaler = MinMaxScaler()
# Compute the minimum and maximum to be used for later scaling.
scaler.fit(originData)
# Scaler features of X according to feature_range.
returnData = scaler.transform(originData)

# fit_transform(X[,y]) => Fit to data, then transform it.
df_log = pd.DataFrame(returnData)
print(df_log.head(5))

#特征工程
def change(column):
    res = [0]
    buyPrice = column[0]
    print(len(column))
    for i in range(1, len(column)):
        res.append((column[i] - buyPrice) / buyPrice)
    return res

# 获取收盘价Close这一列数据
closeCol = data['close']
# 构建累计涨计跌幅这一特征
babaChange = change(closeCol)

# 数据可视化
# plt.rcParams['figure.figsize']=(10,6)
# data[]
data['cumulativeChange'] = babaChange

data['cumulativeChange'].plot(grid=True,c='r') #调整后的收盘价
# plt.plot(label='baba')
plt.title("BABA")
plt.legend()
plt.show()
# ----
