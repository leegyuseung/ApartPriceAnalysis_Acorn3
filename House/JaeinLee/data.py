'''
Created on 2022. 5. 24.
@author: Jain
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
pd.set_option('display.max_columns', None)

# 강남구합계, 시계열데이터 정보 불러오기
apart = pd.read_csv('../datas/아파트2.csv', encoding='cp949', low_memory = False)
del apart['년']
print(apart.head(1))
apart.astype({'ymd':'object'})
print(apart.dtypes)

# apart = apart[apart['구']=='강남구']
# apart = apart.loc[:,['거래금액','ymd']]
# print(apart)
# print(apart.info())

# 행정구별 인구수 불러오기
people = pd.read_csv('../datas/행정구별_인구수.csv', encoding='cp949')
people = people.dropna()
people.rename(columns={'시점':'ymd'}, inplace=True)

# merge_data = pd.merge(apart, people, how = 'outer', on='ymd')
# print(merge_data.head(1))
# 통화량, cpi, 기대인플레 데이터 불러오기
data = pd.read_csv('../datas/시간별 데이터 합.csv', encoding='cp949')
print(data.head(2)) # interest rating : 고정금리, m2 : 통화량, cpi : 미국 소비자물가지수
print

# 부동산 아파트 거래량
buy_data = pd.read_csv('../datas/부동산 아파트 거래량 2011.01-2021.12.csv', encoding='cp949')
print(buy_data.head(2))
print(buy_data['구분'].isnull())
result = pd.merge(apart, data, how='left', on='ymd')
print(result)
print(result.corr())
# ---------------------------------------------------------------------------------------------------
# Arima 모델 활용
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as sgt

# plt.plot(result['m2(abs)'], result['ymd'])
# plt.show()

# 차분
train = result['거래금액'][0:600000]
test = result['거래금액'][60000:]
diff_1 = train.diff().dropna()
plt.plot(diff_1)
plt.show()

diff_2 = diff_1.diff().dropna()
plt.plot(diff_2)
plt.show()

# ACF(자기상관함수), PACF(편자기상관함수)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

sgt.plot_acf(train, lags = 20, zero=False, ax=ax1)
ax1.set_title("ACF 거래금액")

sgt.plot_pacf(train, lags = 20, zero=False, method = ('ols'), ax=ax2 )
ax2.set_title("PACF 거래금액")

plt.show()

model = ARIMA(test, order = (1, 2, 0))
model_fit = model.fit(trend = 'c', full_output = True, disp = True)
print(model_fit.summary())