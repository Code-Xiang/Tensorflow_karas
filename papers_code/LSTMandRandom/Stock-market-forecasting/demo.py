import pandas as pd
import numpy as np
SP500_df = pd.read_csv('papers_code/LSTMandRandom/Stock-market-forecasting/data/SPXconst.csv')
all_companies = list(set(SP500_df.values.flatten()))
all_companies.remove(np.nan)

constituents = {'-'.join(col.split('/')[::-1]):set(SP500_df[col].dropna()) 
                for col in SP500_df.columns}
constituents_train = {} 
for test_year in range(1993,2016):
    months = [str(t)+'-0'+str(m) if m<10 else str(t)+'-'+str(m) for t in range(test_year-3,test_year) for m in range(1,13)]