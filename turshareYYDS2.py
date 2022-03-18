import tushare as ts
ts.set_token('5cb5c9d3abea30da6b1392bd553e241bc4a4f45cdd70fc048d2328cb')
pro = ts.pro_api()


def get_day(df_zhuli):
    for i in tqdm(range(len(df_zhuli))):
        # if i < 299: continue
        day1 = df_zhuli['date_max'][i]
        et = str(day1.year) + str(day1.month).zfill(2) + str(day1.day).zfill(2)
        day2 = df_zhuli['date_min'][i]
        st = str(day2.year) + str(day2.month).zfill(2) + str(day2.day).zfill(2)

        num = myfun2(df_zhuli['code'][i])

        symbol = df_zhuli['symbol'][i]
        exchange = df_zhuli['exchange'][i]

        #
        if exchange != 'CZCE':
            code = exchange + '.' + symbol.lower() + num
            save_path = os.path.join('data/day', code + ".csv")
        else:
            code = exchange + '.' + symbol + num[1:]
            save_path = os.path.join('data/day', code + ".csv")

        code = df_zhuli['code'][i]

        df = pro.fut_daily(ts_code=code, start_date=st, end_date=et)
        df.to_csv(save_path)
        time.sleep(0.3)