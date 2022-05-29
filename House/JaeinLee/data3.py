'''
Created on 2022. 5. 25. 월별 데이터
'''
import pandas as pd
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt

plt.rc('font', family='malgun gothic')
pd.set_option('display.max_columns', None)

data1 = pd.read_csv('../datas/부동산 아파트 거래량 2011.01-2021.12.csv', encoding='cp949')
data2 = pd.read_csv('../datas/시간별 데이터 합.csv', encoding='cp949')
data3 = pd.read_csv('../datas/행정구별_인구수.csv', encoding='cp949')
data3 = data3.dropna()
# print(data1.head(1))
# print(data2.head(1))
# print(data3.head(1))

data1['ymd'] = data1['구분'].apply(lambda x:int(datetime.strptime(x,'%b-%y').strftime('%Y%m')))
data1 = data1.iloc[:, 2:]
data1 = pd.DataFrame(data1.set_index('ymd').unstack(0)).reset_index()
data1.columns = ['구', 'ymd', '거래량']

data3 = data3.iloc[1:, :]
data3['ymd'] = data3['시점'].apply(lambda x:int(datetime.strptime(x, '%Y-%m').strftime("%Y%m")))
data3 = data3.iloc[:, 2:]
data3 = pd.DataFrame(data3.set_index('ymd').unstack(0)).reset_index()
data3.columns = ['구', 'ymd', '인구수']

data_sum = pd.merge(data1, data2, how='left', on='ymd')
data_sum = pd.merge(data_sum, data3, how='left', on=('ymd', '구'))

print(data_sum)
