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
cpi = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
cpi['yymm'] = yymm
cpi['cpi(abs)'] = data['cpi(abs)']
print(cpi.head(3), cpi.tail(3), cpi.shape)
'''         yymm    cpi(abs)                  yymm     cpi(abs)
    0 2011-01-31       88.29        129 2021-10-31       103.35
    1 2011-02-28       88.89        130 2021-11-30       103.87
    2 2011-03-31       89.24        131 2021-12-31       104.04    (132, 2)  '''


print(cpi.info())
timeSeries = cpi.loc[:, ['yymm', 'cpi(abs)']]
timeSeries.index = timeSeries.yymm
ts = timeSeries.drop("yymm", axis=1)
print(ts)
'''             cpi(abs)
    yymm             
    2011-01-31    88.29
    2011-02-28    88.89
    ...             ...
    2021-11-30   103.87
    2021-12-31   104.04    [132 rows x 1 columns]  '''


# # 2011-01부터 2021-12 까지 cpi(%) 그래프
# plt.figure(figsize=(15, 8))
# plt.plot(ts)
# plt.title("cpi(abs) 2011-01 ~ 2021-12")
# plt.xlabel("Year-Month")
# plt.ylabel("cpi(abs)")
# plt.show()


# # seasonal_decompose()
from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(ts['cpi(abs)'], model='additive')
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
# ax1.set_title("cpi(abs) ACF")
# fig = plot_pacf(ts, ax = ax2)
# ax2.set_title("cpi(abs) PACF")
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
''' ADF Statistic : 0.804667
    p-value : 0.991711
    Critical Values : 
           1%: -3.486
           5%: -2.886
          10%: -2.580            '''


# # 1차 차분
ts_diff = ts - ts.shift()
# plt.figure(figsize=(15, 8))
# plt.plot(ts_diff)
# plt.title("cpi(abs) 1st Differencing")
# plt.xlabel('Year-Month')
# plt.ylabel('cpi(abs)')
# plt.show()
# # 1차 차분 데이터로 다시 정상성 검사
result = adfuller(ts_diff[1:])
# print('\n1차 차분 후')
# print('ADF Statistic : %f'% result[0])
# print('p-value : %f'% result[1])
# print('Critical Values : ')
# for key, value in result[4].items():
#     print('\t%s: %.3f'%(key, value))
''' 1차 차분 후
    ADF Statistic : -2.715635
    p-value : 0.071372
    Critical Values : 
           1%: -3.486
           5%: -2.886
          10%: -2.580            '''


# # 1차 차분 데이터로 ACF, PACF 그려서 ARIMA 모형의 p, q 결정
fig = plt.figure(figsize=(18, 10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
fig = plot_acf(ts_diff[1:], ax = ax1)
ax1.set_title("ACF_cpi(abs) diff1")
fig = plot_pacf(ts_diff[1:], ax = ax2)
ax2.set_title("PACF_cpi(abs) diff1")
plt.show()
# # ==> ACF는 1 이후로 0에 수렴, PACF도 1 이후로 0에 수렴.

"""
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
# print(cpi.tail(12))
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
# plt.plot(cpi.yymm, cpi['cpi(%)'], label="original")
# plt.plot(forecast, label='predicted')
# plt.title("cpi(%) Forecast")
# plt.xlabel("Year-Month")
# plt.ylabel("cpi(%)")
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

# score_check(np.array(cpi[cpi.yymm>=start_index].cpi(%)), np.array(forecast))
print(score_check(np.array(cpi[cpi.yymm>=start_index]['cpi(%)']), np.array(forecast)))
# ==>        R2   Corr   RMSE   MAPE
# ==> 0  77.244  0.915  0.468  3.241
"""




