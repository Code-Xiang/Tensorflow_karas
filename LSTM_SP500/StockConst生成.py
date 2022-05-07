
import pandas as pd
import numpy as np
import random
import time
import os
import csv


corporate_name = pd.read_csv('LSTM_SP500/Data/corporate_name.csv')
stockDate = pd.read_csv('LSTM_SP500/Data/SPXconst_2020.csv')
rgdate1 = stockDate.columns.values
# rgdate = list(set(stockDate.columns.values))
print('rgdate:\n',rgdate1)
SP_names = list(set(corporate_name['corporate_name'].values))

# StockName = SP_names[0]


sp_name_true = os.listdir("LSTM_SP500/Data/Stock")
print('sp_name_true:',sp_name_true[66][:-4])

data = []
SP_2020 = pd.DataFrame(data,columns=rgdate1,index= [i for i in range(0,505)])
# SP_2020['01/1990'].append('a')
# df = pd.DataFrame([StockName],columns=['01/1990'],index = [0])
# SP_2020.append(df)
# SP_2020.loc[0,'01/1990'] = str(StockName)


i = 0
for StockName in sp_name_true: 
    StockPrice = pd.read_csv('LSTM_SP500/Data/Stock/'+str(StockName))
    print('StockName:\n',StockName)
    list = []
    for col in StockPrice['Date']:
        t = '/'.join(col.split('-')[::-1])[3:]
        list.append(t)
    
    for year in rgdate1:
        if year in list:
            SP_2020.loc[i,str(year)] = str(StockName[:-4])
            # print('year:\n',year)
        else:
            print('Nothing!!')
    i = i +1
print('SP_2020\n',SP_2020)
SP_2020.to_csv('LSTM_SP500/Data/SPXconst_2020_new.csv')
