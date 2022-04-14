import tushare as ts
import numpy as np
ts.set_token('5cb5c9d3abea30da6b1392bd553e241bc4a4f45cdd70fc048d2328cb')
pro = ts.pro_api()
df = pro.us_daily(ts_code='BSE', start_date='20100101', end_date='20200601')

def transform_dataset(train_set, test_set,y_train,y_test,n_input,n_output):
        # vstack竖直堆叠数组
        all_data = np.vstack((train_set, test_set))
        y_set = np.vstack((y_train, y_test))[:,0]
        X = np.empty((1, n_input, all_data.shape[1]))
        y = np.empty((1, n_output))
        for i in range(all_data.shape[0] - n_input - n_output):
            X_sample = all_data[i:i + n_input, :]
            y_sample = y_set[i + n_input:i + n_input + n_output]
            if i == 0:
                X[i] = X_sample
                y[i] = y_sample
            else:
                X = np.append(X, np.array([X_sample]), axis=0)
                y = np.append(y, np.array([y_sample.T]), axis=0)
        train_X = X[:train_set.shape[0] - n_input, :, :]
        train_y = y[:train_set.shape[0] - n_input, :]
        test_X = X[train_set.shape[0] -
                n_input:all_data.shape[0] -
                n_input -
                n_output, :, :]
        test_y = y[train_set.shape[0] -
                n_input:all_data.shape[0] -
                n_input -
                n_output, :]
        return train_X, train_y, test_X, test_y


