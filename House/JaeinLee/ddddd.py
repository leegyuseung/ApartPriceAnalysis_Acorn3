import pandas as pd
from datetime import datetime
import time


pd.set_option('display.max_columns', None)

buy_data = pd.read_csv('../datas/부동산 아파트 거래량 2011.01-2021.12.csv', encoding='cp949')
buy_data['ymd'] = buy_data['구분'].apply(lambda x:int(datetime.strptime(x,'%b-%y').strftime("%Y%m")))
buy_data = buy_data.iloc[:,2:]
buy_data = pd.DataFrame(buy_data.set_index('ymd').unstack(0)).reset_index()
buy_data.columns = ['구','ymd','거래량']


people = pd.read_csv('../datas/행정구별_인구수.csv', encoding='cp949')
people = people.iloc[1:,:]
people['ymd'] = people['시점'].apply(lambda x:int(datetime.strptime(x,'%Y-%m').strftime("%Y%m")))
people = people.iloc[:,2:]
people = pd.DataFrame(people.set_index('ymd').unstack(0)).reset_index()
people.columns = ['구','ymd','인구수']


data = pd.read_csv('../datas/시간별 데이터 합.csv', encoding='cp949')


apart = pd.read_csv('../datas/아파트2.csv', encoding='cp949')
apart = pd.DataFrame(apart.set_index('ymd').unstack(0)).reset_index()
apart.columns = ['구', 'ymd', '면적가']


result = pd.merge(apart, data, how='left', on='ymd')
result = pd.merge(result, buy_data, how='left', on=('ymd', '구'))
result = pd.merge(result , people, how='left', on=('ymd','구'))

# print(result.head(2))
print(result.corr())
# Arima 모델 활용
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as sgt

plt.rc('font', family='malgun gothic')

# plt.plot(result['m2(abs)'], result['ymd'])
# plt.show()

# 차분
# train_x = result['m2(abs)'][0:600000]
# test_x = result['m2(abs)'][600000:]
# train_y = result['거래금액'][0:600000]
# test_y = result['거래금액'][600000:]
#
# diff_1 = train_y.diff().dropna()
# plt.plot(diff_1)
# plt.show()
#
# diff_2 = diff_1.diff().dropna()
# plt.plot(diff_2)
# plt.show()
#
# # ACF(자기상관함수), PACF(편자기상관함수)
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax2 = fig.add_subplot(2, 1, 2)
#
# sgt.plot_acf(train_y, lags = 20, zero=False, ax=ax1)
# ax1.set_title("ACF 거래금액")
#
# sgt.plot_pacf(train_y, lags = 20, zero=False, method = ('ols'), ax=ax2 )
# ax2.set_title("PACF 거래금액")
#
# plt.show()
#
# model = sm.tsa.arima.ARIMA(test_y, order = (1, 2, 0))
# model_fit = model.fit(test_y)
# print(model_fit.summary())