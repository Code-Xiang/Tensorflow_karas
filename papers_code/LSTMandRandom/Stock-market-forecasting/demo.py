import pandas as pd
import numpy as np
import time

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
st = stock_names[1]
st_data = pd.DataFrame([])
st_data['Date'] = list(df_close['Date'])
st_data['Name'] = [st]*len(st_data)
daily_change = df_close[st]/df_open[st]-1
print('daily_change:\n',daily_change)
print('daily_change.shift(k):\n',daily_change.shift(2))
for k in range(240)[::-1]:
    st_data['IntraR'+str(k)] = daily_change.shift(k)

# print('st_data:\n',st_data)
# print('df_close[st]:\n',df_close[st])
# print('daily_change:\n',daily_change)
# for st in stock_names:

# print('lambda x:\n', x)
# lambda x: pd.qcut(x.rank(method='first'))
    # if not np.all(df_close.iloc[:,1:]/df_open.iloc[:,1:]-1).apply()
    #     return
    # label = create_label(df_open,df_close)

