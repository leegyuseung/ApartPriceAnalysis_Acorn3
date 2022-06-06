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
data = pd.read_csv('../datas/행정구별_인구수.csv', parse_dates=['시점'], encoding='cp949')
popul_JR = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
popul_JR['yymm'] = yymm
popul_JR['popul'] = data['종로구']
popul_JR['popul'][131] = popul_JR['popul'][130]
# print(popul_JR.head(3), popul_JR.tail(3), popul_JR.shape)
'''         yymm        popul               yymm         popul
    0 2011-01-31     170577.0     129 2021-10-31      145346.0
    1 2011-02-28     170617.0     130 2021-11-30      145073.0
    2 2011-03-31     170099.0     131 2021-12-31      145073.0    (132, 2)  '''


print(popul_JR.info())
timeSeries = popul_JR.loc[:, ['yymm', 'popul']]
timeSeries.index = timeSeries.yymm
ts = timeSeries.drop("yymm", axis=1)
# print(ts)
'''                   popul
     yymm             
    2011-01-31     170577.0
    2011-02-28     170617.0
    ...            ...
    2021-11-30     145073.0
    2021-12-31     145073.0    [132 rows x 1 columns]  '''


# # 2011-01부터 2021-12 까지 cpi(%) 그래프
# plt.figure(figsize=(15, 8))
# plt.plot(ts)
# plt.title("Population-Jongro 2011-01 ~ 2021-12")
# plt.xlabel("Year-Month")
# plt.ylabel("popul")
# plt.show()


# # seasonal_decompose()
from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(ts['popul'], model='additive')
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
# ax1.set_title("Marriage-Jongro ACF")
# fig = plot_pacf(ts, ax = ax2)
# ax2.set_title("Marriage-Jongro PACF")
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
''' ADF Statistic : -1.113042
    p-value : 0.823664
    Critical Values : 
           1%: -3.482
           5%: -2.884
          10%: -2.579            '''


# # 1차 차분
ts_diff = ts - ts.shift()
# plt.figure(figsize=(15, 8))
# plt.plot(ts_diff)
# plt.title("Marriage-Jongro 1st Differencing")
# plt.xlabel('Year-Month')
# plt.ylabel('popul')
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
    ADF Statistic : -4.768620
    p-value : 0.000062
    Critical Values : 
           1%: -3.482
           5%: -2.884
          10%: -2.579            '''


# # 1차 차분 데이터로 ACF, PACF 그려서 ARIMA 모형의 p, q 결정
# fig = plt.figure(figsize=(18, 10))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# fig = plot_acf(ts_diff[1:], ax = ax1)
# ax1.set_title("ACF_Population-Jongro diff1")
# fig = plot_pacf(ts_diff[1:], ax = ax2)
# ax2.set_title("PACF_Population-Jongro diff1")
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
# print(popul_JR.tail(12))
# print(forecast)
'''            - 실제값 -           - forecast -
              yymm     popul            popul
        2021-01-31  149125.0    149377.976102
        2021-02-28  148884.0    149115.544306
        2021-03-31  147296.0    148875.201459
        2021-04-30  147113.0    147238.024549
        2021-05-31  146377.0    147106.318950
        2021-06-30  146029.0    146350.129766
        2021-07-31  145692.0    146016.295052
        2021-08-31  145551.0    145679.696645
        2021-09-30  145512.0    145545.852306
        2021-10-31  145346.0    145510.576170
        2021-11-30  145073.0    145339.939594
        2021-12-31  145073.0    145063.033188
'''

# # 시각화
# plt.figure(figsize=(15, 8))
# plt.plot(popul_JR.yymm, popul_JR['popul'], label="original")
# plt.plot(forecast, label='predicted')
# plt.title("Population-Jongro Forecast")
# plt.xlabel("Year-Month")
# plt.ylabel("popul")
# plt.legend()
# plt.show()



# # 성능 확인
from sklearn import metrics
# def score_check(y_true, y_pred):
#     r2 = round(metrics.r2_score(y_true, y_pred) * 100, 3)
#     #     mae = round(metrices.mean_absolute_error(y_true, y_pred),3)
#     corr = round(np.corrcoef(y_true, y_pred)[0, 1], 3)
#     mape = round(
#         metrics.mean_absolute_percentage_error(y_true, y_pred) * 100, 3)
#     rmse = round(metrics.mean_squared_error(y_true, y_pred, squared=False), 3)
#
#     df = pd.DataFrame({
#         'R2':r2,
#         'Corr':corr,
#         'RMSE':rmse,
#         'MAPE':mape
#     },
#                     index=[0])
#     return df
#
# # score_check(np.array(taxJ[taxJ.yymm>=start_index].tax_jongso), np.array(forecast))
# print(score_check(np.array(popul_JR[popul_JR.yymm>=start_index]['popul']), np.array(forecast)))
# ==>         R2   Corr   RMSE    MAPE
# ==>  0  83.934  0.964  538.785  0.237





