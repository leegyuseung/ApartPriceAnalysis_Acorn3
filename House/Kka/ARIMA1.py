# ARIMA 모델로 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rc('axes', unicode_minus=False)
import seaborn as sns
import datetime
from dateutil.relativedelta import relativedelta
import statsmodels
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

data = pd.read_csv('구별,월별 평당가격.csv', parse_dates=['ymd'])
print(data.head(5))
print(data.info()) # ymd는 int64, 평당가격은 float64    # ymd와 구 칼럼 모두 다 132행

Jongro = pd.DataFrame()
# yymm을 pd.date_range로 만들어 보려고 했으나.. 잘 안 됨...
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
Jongro['yymm'] = yymm
Jongro['price'] = data['종로구']
Jongro.set_index('yymm', inplace=True)
# Jongro['yymm'] = data['ymd']
print(Jongro.info())
print(Jongro.head(3), Jongro.tail(3))
# print(Jongro.head(3), Jongro.tail(3), Jongro.shape)


# 시계열 데이터 정상성 확인
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(10, 6))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(Jongro['price'])


# 정상화
# 1. 차분
Jongro['first_difference'] = Jongro['price'] - Jongro['price'].shift(1)  
# Or Alternatively,
Jongro.diff().plot()
test_stationarity(Jongro.first_difference.dropna(inplace=False))



print()
# 데이터 전처리
# 표준화
scaler1 = StandardScaler()
JR1 = scaler1.fit_transform(Jongro)
print('StandardScaler : ', JR1[:10])
# 정규화
scaler2 = MinMaxScaler(feature_range=(0,1))
JR2 = scaler2.fit_transform(JR1)
np.set_printoptions(precision=6, suppress=True)
print('MinMaxScaler : ', JR2[:10])


# 시각화
Jongro.plot(figsize=(15,10), title = 'Monthly Price', fontsize=14)
plt.xlabel('년-월')
plt.ylabel('구별 평당가격')
plt.legend()
plt.show()


# Seasonal Decomposition
decomposition = seasonal_decompose(data['종로구'], period=12)  
fig = plt.figure()  
fig = decomposition.plot()
plt.show()

# ARIMA 모수 설정 : ACF / PACH
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(JR2)
plot_pacf(JR2)
plt.show()


# train-test 수동으로 나누기
J_train = JR2[:33]      # (33, 1)
# print(JR2[:-3])
J_test = JR2[33:132]    # (99, 1)











