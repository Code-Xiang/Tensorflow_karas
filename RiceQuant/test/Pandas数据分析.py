import pandas as pd
import numpy as np
s = pd.Series([1,3,5,np.nan,6,8])
# print('s:\n',s)
# s:
#  0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64

#--------------
# print(s[1:3])
# 1    3.0
# 2    5.0
# dtype: float64
#--------------

#--------------
# print(s[2])
# 5.0
#--------------

dates = pd.date_range('20220101',periods=6)
# print('dates:\n',dates)
# dates:
#  DatetimeIndex(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04',
#                '2022-01-05', '2022-01-06'],
#               dtype='datetime64[ns]', freq='D')j
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
# print('df:\n',df)
# df:
#                     A         B         C         D
# 2022-01-01 -1.149241  0.681800 -2.674275  1.040423
# 2022-01-02 -1.085501 -0.336165  0.192069 -0.086606
# 2022-01-03  0.625990  1.135149 -0.223805  0.316152
# 2022-01-04  1.079751  0.016024 -0.625309  0.147470
# 2022-01-05  0.330139 -0.222871 -1.498270  0.293469
# 2022-01-06  2.123219  0.477126  0.524279 -0.007619
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
# print('df2:\n',df2)
#      A          B    C  D      E    F
# 0  1.0 2013-01-02  1.0  3   test  foo
# 1  1.0 2013-01-02  1.0  3  train  foo
# 2  1.0 2013-01-02  1.0  3   test  foo
# 3  1.0 2013-01-02  1.0  3  train  foo
# print(df2.dtypes)
# A           float64
# B    datetime64[ns]
# C           float32
# D             int32
# E          category
# F            object
# dtype: object

# print(df2['E'])
# 0     test
# 1    train
# 2     test
# 3    train
# Name: E, dtype: category
# Categories (2, object): ['test', 'train']

