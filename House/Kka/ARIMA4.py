import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from IPython.core.pylabtools import figsize
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

# 데이터 불러오기
data = pd.read_csv('구별,월별 평당가격.csv', parse_dates=['ymd'])
Jongro = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
Jongro['yymm'] = yymm
Jongro['price'] = data['종로구']
# Jongro.set_index('yymm', inplace=True)
# print(Jongro.head(3), Jongro.tail(3), Jongro.shape)
'''                   price                           price
    yymm                           yymm
    2011-01-31  1878.779446        2021-10-31  4057.966887
    2011-02-28  1909.280698        2021-11-30  3687.941989
    2011-03-31  1786.382560        2021-12-31  3907.407257    (132, 1)
'''
print(Jongro.info())
timeSeries = Jongro.loc[:, ['yymm', 'price']]
timeSeries.index = timeSeries.yymm
ts = timeSeries.drop("yymm", axis=1)


# # 2011-01부터 2021-12 까지 구별-월별 아파트 평당가격 그래프
# plt.figure(figsize=(15, 8))
# plt.plot(Jongro)
# plt.title("Seoul APT price 2011-01 ~ 2021-12")
# plt.xlabel("Year-Month")
# plt.ylabel("price")
# plt.show()


# # seasonal_decompose()
from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(ts.price, model='additive')
# fig = plt.figure()
# fig = result.plot()
# fig.set_size_inches(15, 10)
# plt.show()


# # ACF, PACF 그래프
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# fig = plt.figure(figsize=(18, 10))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# fig = plot_acf(ts, ax = ax1)
# ax1.set_title("ACF")
# fig = plot_pacf(ts, ax = ax2)
# ax2.set_title("PACF")
# plt.show()


# # 정상성 확인 : ADF(Augmented Dickey-Fuller test)
# # 귀무가설 : 자료가 정상성을 만족하지 않는다. / 대립가설 : 자료가 정상성을 만족한다.
from statsmodels.tsa.stattools import adfuller
# result = adfuller(ts)
# print('ADF Statistic : %f'% result[0])
# print('p-value : %f'% result[1])
# print('Critical Values : ')
# for key, value in result[4].items():
#     print('\t%s: %.3f'%(key, value))
''' ADF Statistic : 3.745713
    p-value : 1.000000
    Critical Values : 
        1%: -3.487
        5%: -2.886
        10%: -2.580            '''


# # 1차 차분
ts_diff = ts - ts.shift()
# plt.figure(figsize=(15, 8))
# plt.plot(ts_diff)
# plt.title("1st Differencing")
# plt.xlabel('Year-Month')
# plt.ylabel('APT price')
# plt.show()
# 1차 차분 데이터로 다시 정상성 검사
result = adfuller(ts_diff[1:])
print('\n1차 차분 후')
print('ADF Statistic : %f'% result[0])
print('p-value : %f'% result[1])
print('Critical Values : ')
for key, value in result[4].items():
    print('\t%s: %.3f'%(key, value))
''' ADF Statistic : -8.329827
    p-value : 0.000000
    Critical Values : 
        1%: -3.483
        5%: -2.885
        10%: -2.579            '''


# # 1차 차분 데이터로 ACF, PACF 그려서 ARIMA 모형의 p, q 결정
# fig = plt.figure(figsize=(18, 10))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# fig = plot_acf(ts_diff[1:], ax = ax1)
# ax1.set_title("ACF_diff1")
# fig = plot_pacf(ts_diff[1:], ax = ax2)
# ax2.set_title("PACF_diff1")
# plt.show()
# # ==> ACF는 1 이후로 0에 수렴, PACF도 1 이후로 0에 수렴.


# # ARIMA 모델 만들기
from statsmodels.tsa.arima.model import ARIMA
# fit model
model = ARIMA(ts, order=(1, 1, 0))
model_fit = model.fit()
# predict
start_index = yymm[120] # 2021-01-31 00:00:00
end_index = yymm[131]   # 2021-12-31 00:00:00
forecast = model_fit.predict(start=start_index, end=end_index, typ='levels')
# 시각화
plt.figure(figsize=(15, 8))
plt.plot(Jongro.yymm, Jongro.price, label="original")
plt.plot(forecast, label='predicted')
plt.title("Jongro-gu APT price Forecast")
plt.xlabel("Year-Month")
plt.ylabel("APT price per 1.8m*1.8m")
plt.legend()
plt.show()


# # 잔차 분석 : 잘 안 됨...
# 어떠한 패턴이나 특성이 없는지 확인!
resi = np.array(Jongro[Jongro.yymm>=start_index].price) - np.array(forecast)
# plt.figure(figsize=(15, 8))
# plt.plot(Jongro[Jongro.yymm>=start_index], resi)
# plt.xlabel("Year-Month")
# plt.ylabel("Residual")
# plt.legend()
# plt.show()


# # ACF 그래프 및 ADF 검정을 통해 정상성 확인
# # ACF
# fig = plt.figure(figsize=(15, 4))
# fig = plot_acf(resi)
# plt.show()


# # 성능 확인
from sklearn import metrics

def score_check(y_true, y_pred):
    r2 = round(metrics.r2_score(y_true, y_pred) * 100, 3)
    #     mae = round(metrices.mean_absolute_error(y_true, y_pred),3)
    corr = round(np.corrcoef(y_true, y_pred)[0, 1], 3)
    mape = round(
        metrics.mean_absolute_percentage_error(y_true, y_pred) * 100, 3)
    rmse = round(metrics.mean_squared_error(y_true, y_pred, squared=False), 3)
    
    df = pd.DataFrame({
        'R2':r2,
        'Corr':corr,
        'RMSE':rmse,
        'MAPE':mape
    },
                    index=[0])
    return df

score_check(np.array(Jongro[Jongro.yymm>=start_index].price), np.array(forecast))






