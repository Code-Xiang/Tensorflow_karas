import pandas as pd
import numpy as np
import random
import time
import pickle

# import tensorflow as tf
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dropout,Dense,Input,add
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler
# from tensorflow.keras.models import Model, Sequential, load_model
# from tensorflow.keras import optimizers
# import warnings
# warnings.filterwarnings("ignore")

# import os
# SEED = 9
# os.environ['PYTHONHASHSEED']=str(SEED)

# random.seed(SEED)
# np.random.seed(SEED)

# SP500_df = pd.read_csv('papers_code/LSTMandRandom/Stock-market-forecasting/data/SPXconst.csv')
# print('-----------------------')
# print('SP500_df\n',SP500_df.head)
# zer = np.zeros((245,1))
# train_ret = np.arange(245,dtype=np.float32)
# train_ret = train_ret.reshape(245,1)
# print('train_ret:\n',train_ret)
# print('zero:\n',zer)
# df = np.hstack((np.zeros((245,1)),train_ret))
# print('df:\n',df)

rets = pd.DataFrame([],columns=['Long','Short'])
print('rets:\n',rets)