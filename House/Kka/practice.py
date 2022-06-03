import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

pd.set_option('display.max_columns', None)
GangNam = pd.read_csv('TotalData_Gangnam.csv')
GangNam.set_index('ymd', inplace=True)
print(GangNam.head(5))
print(GangNam.info())


# 표준화
scaler1 = StandardScaler()
GN1 = scaler1.fit_transform(GangNam)
print('StandardScaler : ', GN1[:10])
# 정규화
scaler2 = MinMaxScaler(feature_range=(0,1))
GN2 = scaler2.fit_transform(GN1)
np.set_printoptions(precision=6, suppress=True)
print('MinMaxScaler : ', GN2[:10])