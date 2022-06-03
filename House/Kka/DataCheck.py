import os   # provides functions for interacting with the operating system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import seaborn as sns
import itertools
import warnings
import datetime
from datetime import datetime
warnings.filterwarnings('ignore')

from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat, to_datetime

# %matplotlib inline

# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})


# 데이터 불러오기
data = pd.read_csv('구별,월별 평당가격.csv', parse_dates=['ymd'])
Jongro = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
Jongro['yymm'] = yymm
Jongro['price'] = data['종로구']
Jongro.set_index('yymm', inplace=True)
print(Jongro.head(3), Jongro.tail(3))
print(Jongro.shape)     # (132, 1)


# ACF, PACF 플롯 그리기
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# plot_acf(Jongro)
# plot_pacf(Jongro)
# plt.show()


# ARIMA 모델 패키지 임포트
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
# Auto Arima 모델 패키지
# Anaconda Prompt로 설치 : (Anaconda3 prompt)>>pip install pmdarima
import pmdarima as pm
from pmdarima.model_selection import train_test_split

# train 데이터와 validation 데이터 나누기
# train-test 수동으로 나누기
J_train = Jongro[Jongro.index < '2021-01-31']
print(J_train.tail(3), J_train.shape)
J_test = Jongro[Jongro.index >= '2021-01-31']
print(J_test.head(3), J_test.tail(3), J_test.shape)
print(J_train.index[-1])    # 2020-12-31 00:00:00

plt.plot(J_train)
plt.title('J_train')
plt.show()


# Differencing(차분)
# Train 데이터
# 1차
diff1 = J_train.diff().dropna()
plt.plot(diff1, color='red')
# plt.title('1st_diff')
# plt.show()
# 2차
diff2 = diff1.diff().dropna()
plt.plot(diff2)
plt.title('2nd_diff')
plt.show()
# test 데이터
test_diff1 = J_test.diff().dropna()
test_diff2 = test_diff1.diff().dropna()


# 향후 1년치 값을 예측할 것이므로 예측 날짜들을 인덱스로 한 dataframe 만들기
index_12_weeks = pd.date_range('2021-01-31', freq='M', periods = 12, tz = None)
# 확인
print(index_12_weeks)   # dtype='datetime64[ns]'


# 최적의 (p, d, q) 값 찾기
# 1. Auto-ARIMA 모델 사용 : 계측값이 일별이면 m=1, 월별이면 m=12, 주별이면 m=52
auto_arima_model = pm.auto_arima(diff2, seasonal=False, m=12)
# 모델 예측
fcast = auto_arima_model.predict(12)
fcast = pd.Series(fcast, index=index_12_weeks)
fcast = fcast.rename("Auto Arima")
# 예측값 시각화
fig, ax = plt.subplots(figsize=(15, 5))
chart = sns.lineplot(x='yymm', y='price', data=diff2)
chart.set_title('Seoul APT price')
fcast.plot(ax=ax, color='red', marker="o", legend=True)
test_diff2.plot(ax=ax, color='blue', marker="o", legend=True)
plt.show()
# AIC 프린트
print('The MSE of auto-arima is : ', mean_squared_error(test_diff2['price'].values, fcast.values, squared=False))
# ㄴ-> The MSE of auto-arima is :  413.09429819645584


""" 뭔가 이상함... 울고 싶다 ㅠㅠㅠㅠ
# 모델 돌리기!
# order에 파라미터 넣어주기
model1 = ARIMA(J_train, order=(1, 1, 0)).fit()
# 에측한 값들을 저장
fcast1 = model1.forecast(12)[0]
fcast1 = pd.Series(fcast1, index = index_12_weeks)
# for i in range(0, 12):
#     fcast1 = pd.DataFrame(index = index_12_weeks)
#     fcast1 = fcast1.append(model1.forecast(12)[i])

fcast1 = fcast1.rename("Arima")
# 확인
print(fcast1)

''' NotImplementedError: 
    statsmodels.tsa.arima_model.ARMA
    statsmodels.tsa.arima_model.ARIMA
    위에 거 말고 아래 거 써서 임포트 해야 함.
    statsmodels.tsa.arima.model.ARIMA '''
print(model1.forecast(12)[0], model1.forecast(12)[1], model1.forecast(12)[3])
"""



