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
m2['m2(abs)'] = data['m2(abs)']
# m2.set_index('yymm', inplace=True)
print(m2.head(3), m2.tail(3), m2.shape)
'''         yymm     m2(abs)                  yymm       m2(abs)
    0 2011-01-31  1676448.8        129 2021-10-31      3543363.8
    1 2011-02-28  1674390.5        130 2021-11-30      3594723.2
    2 2011-03-31  1677475.9        131 2021-12-31      3620057.5    (132, 2)  '''


print(m2.info())
timeSeries = m2.loc[:, ['yymm', 'm2(abs)']]
timeSeries.index = timeSeries.yymm
ts = timeSeries.drop("yymm", axis=1)
print(ts)
'''                 m2(abs)
    yymm             
    2011-01-31   1676448.8
    2011-02-28   1674390.5
    ...          ...
    2021-11-30   3594723.2
    2021-12-31   3620057.5    [132 rows x 1 columns]  '''


# # 2011-01부터 2021-12 까지 m2(abs) 그래프
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
        1%:  -3.487
        5%:  -2.886
        10%: -2.580            '''


# # auto_arima로 최적의 모형 탐색하기
import pmdarima as pm
train = m2['m2(abs)'][:120]

model = pm.auto_arima(y = train        # 데이터
                      , d = 1            # 차분 차수, ndiffs 결과!
                      , start_p = 0 
                      , max_p = 5   
                      , start_q = 0 
                      , max_q = 5   
                      , m = 1       
                      , seasonal = False # 계절성 ARIMA가 아니라면 필수!
                      , stepwise = True
                      , trace=True
                      )

model_fit = model.fit(train)
''' >> 결과
    Performing stepwise search to minimize aic
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=2484.382, Time=0.01 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=2496.954, Time=0.07 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=2483.783, Time=0.02 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=2630.603, Time=0.00 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=2484.869, Time=0.08 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=2484.567, Time=0.03 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=2488.500, Time=0.16 sec
     ARIMA(0,1,1)(0,0,0)[0]             : AIC=2632.160, Time=0.02 sec
    
    Best model:  ARIMA(0,1,1)(0,0,0)[0] intercept
    Total fit time: 0.390 seconds
'''

# # 잔차 검정
print(model.summary())
'''                                SARIMAX Results                                                              
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  120
    Model:               SARIMAX(0, 1, 1)   Log Likelihood               -1238.892
    Date:                Tue, 07 Jun 2022   AIC                           2483.783
    Time:                        18:33:18   BIC                           2492.121
    Sample:                             0   HQIC                          2487.169
                                    - 120                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept   1.274e+04    719.339     17.716      0.000    1.13e+04    1.42e+04
    ma.L1         -0.0081      0.023     -0.354      0.723      -0.053       0.037
    sigma2      6.302e+07      0.032   1.99e+09      0.000     6.3e+07     6.3e+07
    ===================================================================================
    Ljung-Box (L1) (Q):                  15.92   Jarque-Bera (JB):                 9.53
    Prob(Q):                              0.00   Prob(JB):                         0.01
    Heteroskedasticity (H):               2.29   Skew:                             0.69
    Prob(H) (two-sided):                  0.01   Kurtosis:                         3.09
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 2.02e+24. Standard errors may be unstable.
'''


# # 1차 차분 데이터로 ACF, PACF 그려서 ARIMA 모형의 p, q 결정
# fig = plt.figure(figsize=(18, 10))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
# fig = plot_acf(ts_diff[1:], ax = ax1)
# ax1.set_title("ACF_m2(abs) diff1")
# fig = plot_pacf(ts_diff[1:], ax = ax2)
# ax2.set_title("PACF_m2(abs) diff1")
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
# plt.title("m2(abs) Forecast")
# plt.xlabel("Year-Month")
# plt.ylabel("m2(abs)")
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
print(score_check(np.array(m2[m2.yymm>=start_index]['m2(abs)']), np.array(forecast)))
# #           R2   Corr       RMSE   MAPE
# # => 0  91.089  0.997  36146.723  1.016





