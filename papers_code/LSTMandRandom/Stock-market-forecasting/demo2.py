from Statistics import Statistics
from unittest import result
from wsgiref import validate
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
import os
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dropout,Dense,Input,add
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
from tensorflow.keras import optimizers
SP500_df = pd.read_csv('papers_code/LSTMandRandom/Stock-market-forecasting/data/SPXconst.csv')
all_companies = list(set(SP500_df.values.flatten()))
all_companies.remove(np.nan)
print('SP500_df.columns:\n',SP500_df['01/1990'])
constituents = {'-'.join(col.split('/')[::-1]):set(SP500_df[col].dropna()) 
                for col in SP500_df.columns}
# print('constituents:\n',constituents)