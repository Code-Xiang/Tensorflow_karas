import pandas as pd
import numpy as np
import random
import time
import os




# SP500_df = pd.read_csv('LSTM_SP500/Data/SP 500 Stock Price 2013-2017.csv')
# corporate_name = list(set(SP500_df['Name'].values))
# names = ['corporate_name']
# test = pd.DataFrame(columns = names,data = corporate_name)
# test.to_csv('LSTM_SP500/Data/corporate_name.csv')
# for i in range():

corporate_name = pd.read_csv('LSTM_SP500/Data/corporate_name.csv')
SP_names = list(set(corporate_name['corporate_name'].values))
# print('SP500_df:\n',SP_names)
StockName = SP_names[0]
StockPrice = pd.read_csv('LSTM_SP500/Data/Stock/'+str(StockName)+'.csv')
print('StockName:\n',StockName)
print('StockPrice:\n',StockPrice)
# print('===='*20+'\n',StockName)
# for StockName in SP_names:
#     print('StockName:\n',StockName)