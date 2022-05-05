import pandas as pd
import numpy as np
import random
import time
import os




SP500_df = pd.read_csv('LSTM_SP500/Data/SP 500 Stock Price 2013-2017.csv')
# corporate_name = list(set(SP500_df['Name'].values))
# print('corporate_name:\n',corporate_name)
# names = ['corporate_name']
# test = pd.DataFrame(columns = names,data = corporate_name)
# test.to_csv('LSTM_SP500/Data/corporate_name.csv')
# for i in range():

corporate_name = pd.read_csv('LSTM_SP500/Data/corporate_name.csv')
stockDate = pd.read_csv('LSTM_SP500/Data/SPXconst_2020.csv')
# print('stockDate:\n',stockDate.columns.values)
rgdate = stockDate.columns.values
SP_names = list(set(corporate_name['corporate_name'].values))
StockName = SP_names[0]
StockPrice = pd.read_csv('LSTM_SP500/Data/Stock/'+str(StockName)+'.csv')
print('StockName:\n',StockName)
list = []
for col in StockPrice['Date']:
    # print('col:\n',col)
    # print('++++++++\n',StockPrice['Date'].values)
    # print('======\n',col.replace('-',''))
    t = '/'.join(col.split('-')[::-1])[3:]
    # t = col.replace('-','/')
    list.append(t)
print('list:\n',list)
# names = ['date']
# Date = pd.DataFrame(columns=names, data=list)
# print('Date_4:\n',Date['date'])
# ext = list(set(Date['date'].str[3:]))
# print('ext:\n',ext)
# print('Date_6:\n',Date['date'].str[:6])
# print('StockPrice:\n',StockPrice['Date'])
mon = ['01','02','03','04','05','06','07','08','09','10','11','12']

for year in rgdate:
    # print('year:\n',year)
    if year in list:
        print('year:\n',year)
    else:
        print('Nothing!!')
    # for month in range(1,10):
    #     td = int(str(year)+'0'+str(month))
    #     # print('td:\n',td)
    #     if td in Date['date'].str[:6]:
    #         print('//////////\n',td)
    #     # print('year:\n',year)
    #     # for month in range(1,13):
    #         # print('=====================\n', int(str(year) + str(month)) )
    #     # for month in range(1,10):
    #     #     td = int(str(year)+'0'+str(month))
    #     #     # print('td:',td)
    #     #     # print('date',)
    #     #     if td in Date['date'].str[:6]:
    #     #         print('===============\n',td)sa
    #     # # for month in range(10,13):
    #     # #     td = int(str(year)+str(month))
    #     # #     if td in Date['date'].str[:6]:
    #     # #         print('==========///////////\n',td)
    #     #     else:
    #     #         print('nothing!')

            
    #     else:
    #         print('False')

# print('===='*20+'\n',StockName)
# for StockName in SP_names:
#     print('StockName:\n',StockName)

