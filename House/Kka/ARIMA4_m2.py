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
data = pd.read_csv('시간별 데이터 합.csv', parse_dates=['ymd'], encoding='cp949')
m2 = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
m2['yymm'] = yymm
m2['m2(%)'] = data['m2(%)']
# Jongro.set_index('yymm', inplace=True)
# print(m2.head(3), m2.tail(3), m2.shape)
'''         yymm    m2                  yymm     m2
    0 2011-01-31  6.49        129 2021-10-31  12.39
    1 2011-02-28  4.95        130 2021-11-30  12.92
    2 2011-03-31  4.33        131 2021-12-31  13.21    (132, 2)  '''


print(m2.info())
timeSeries = m2.loc[:, ['yymm', 'm2(%)']]
timeSeries.index = timeSeries.yymm
ts = timeSeries.drop("yymm", axis=1)
'''             m2(%)
    yymm             
    2011-01-31   6.49
    2011-02-28   4.95
    ...           ...
    2021-11-30  12.92
    2021-12-31  13.21    [132 rows x 1 columns]  '''


# # 2011-01부터 2021-12 까지 m2(%) 그래프
# plt.figure(figsize=(15, 8))
# plt.plot(ts)
# plt.title("m2(%) 2011-01 ~ 2021-12")
# plt.xlabel("Year-Month")
# plt.ylabel("m2(%)")
# plt.show()



# # seasonal_decompose()
from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(ts['m2(%)'], model='additive')
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
# ax1.set_title("m2(%) ACF")
# fig = plot_pacf(ts, ax = ax2)
# ax2.set_title("m2(%) PACF")
# plt.show()


# # 정상성 확인 : ADF(Augmented Dickey-Fuller test)
# # 귀무가설 : 자료가 정상성을 만족하지 않는다. / 대립가설 : 자료가 정상성을 만족한다.
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts)
print('ADF Statistic : %f'% result[0])
print('p-value : %f'% result[1])
print('Critical Values : ')
for key, value in result[4].items():
    print('\t%s: %.3f'%(key, value))
''' ADF Statistic : 0.220353
    p-value : 0.973380
    Critical Values : 
        1%: -3.487
        5%: -2.886
        10%: -2.580            '''


# # 1차 차분
ts_diff = ts - ts.shift()
# plt.figure(figsize=(15, 8))
# plt.plot(ts_diff)
# plt.title("m2(%) 1st Differencing")
# plt.xlabel('Year-Month')
# plt.ylabel('m2(%)')
# plt.show()
# # 1차 차분 데이터로 다시 정상성 검사
result = adfuller(ts_diff[1:])
print('\n1차 차분 후')
print('ADF Statistic : %f'% result[0])
print('p-value : %f'% result[1])
print('Critical Values : ')
for key, value in result[4].items():
    print('\t%s: %.3f'%(key, value))
''' ADF Statistic : -2.810977
    p-value : 0.056723
    Critical Values : 
        1%: -3.487
        5%: -2.886
        10%: -2.580            '''


# # 1차 차분 데이터로 ACF, PACF 그려서 ARIMA 모형의 p, q 결정
# fig = plt.figure(figsize=(18, 10))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# fig = plot_acf(ts_diff[1:], ax = ax1)
# ax1.set_title("ACF_m2(%) diff1")
# fig = plot_pacf(ts_diff[1:], ax = ax2)
# ax2.set_title("PACF_m2(%) diff1")
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
# # 실제값이랑 비교하기
# print(m2.tail(12))
# print(forecast)
'''        - 실제값 -     - forecast -
          yymm   m2(%)        m2(%)
    2021-01-31  10.05      9.809443
    2021-02-28  10.72     10.122215
    2021-03-31  11.02     10.906094
    2021-04-30  11.38     11.103326
    2021-05-31  10.96     11.479991
    2021-06-30  10.94     10.843344
    2021-07-31  11.38     10.934445
    2021-08-31  12.49     11.502211
    2021-09-30  12.79     12.798305
    2021-10-31  12.39     12.873326
    2021-11-30  12.92     12.278899
    2021-12-31  13.21     13.067209
'''

# 시각화
# plt.figure(figsize=(15, 8))
# plt.plot(m2.yymm, m2['m2(%)'], label="original")
# plt.plot(forecast, label='predicted')
# plt.title("m2(%) Forecast")
# plt.xlabel("Year-Month")
# plt.ylabel("m2(%)")
# plt.legend()
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

# score_check(np.array(Jongro[Jongro.yymm>=start_index].price), np.array(forecast))
print(score_check(np.array(m2[m2.yymm>=start_index]['m2(%)']), np.array(forecast)))
# ==>        R2   Corr   RMSE   MAPE
# ==> 0  77.244  0.915  0.468  3.241





