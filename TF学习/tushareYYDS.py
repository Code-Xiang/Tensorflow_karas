import tushare as ts
ts.set_token('5cb5c9d3abea30da6b1392bd553e241bc4a4f45cdd70fc048d2328cb')
pro = ts.pro_api()
# df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
# print(df)

#获取CU1811合约20180101～20181113期间的行情
# df = pro.fut_daily(ts_code='CU1811.SHF', start_date='20180101', end_date='20191113')

# print(ts.__version__)

# pro = ts.pro_api('your token')
# df = pro.fut_basic(exchange='DCE', fut_type='1', fields='ts_code,symbol,name,list_date,delist_date')
# df = ts.get_hist_data('600848') #一次性获取全部日k线数据
# print(df)
# fut_wsr
# df = pro.fut_holding(trade_date='20181113', symbol='C', exchange='DCE')
# print(df)

df_list = []
for exchange in ['DCE', 'CZCE', 'SHFE', 'INE']:
    df = pro.fut_basic(exchange=exchange, fut_type='2', fields='ts_code,symbol,name')
    df['is_zhuli'] = df['name'].apply(myfun1)
    df = df[df['is_zhuli'] == True]
    df['exchange'] = exchange
    df_list.append(df[['ts_code', 'symbol', 'name', 'exchange']])
df_list = pd.concat(df_list).reset_index(drop=True)

df_zhuli = []
for i in tqdm(range(len(df_list))):
    code = df_list['ts_code'][i]
    exchange = df_list['exchange'][i]
    symbol = df_list['symbol'][i]

    # 获取主力合约TF.CFX每日对应的月合约
    df = pro.fut_mapping(ts_code=code)
    df['trade_date'] = pd.to_datetime(df['trade_date'], infer_datetime_format=True)

    # 筛选到制定时间段
    df = df[(df['trade_date'].dt.date >= trade_date_min) & (df['trade_date'].dt.date <= trade_date_max)]

    # 获得成为主力合约的时间段
    df = df.groupby('mapping_ts_code')['trade_date'].agg({'max', 'min'}).reset_index()
    df['exchange'] = exchange
    df['symbol'] = symbol
    # df.columns = ['code', 'date_max', 'date_min', 'exchange', 'symbol']
    df = df.rename(columns={'mapping_ts_code': 'code', 'max': 'date_max', 'min': 'date_min'})
    df_zhuli.append(df)
df_zhuli = pd.concat(df_zhuli).reset_index(drop=True)