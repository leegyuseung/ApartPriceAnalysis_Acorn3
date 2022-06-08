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
data = pd.read_csv('../datas/서울시 구별 혼인건수.csv', parse_dates=['날짜'], encoding='utf-8-sig')
marry_JR = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
marry_JR['yymm'] = yymm
marry_JR['marry'] = data['종로구']
# print(marry_JR.head(3), marry_JR.tail(3), marry_JR.shape)
'''         yymm    marry               yymm     marry
    0 2011-01-31     82.0     129 2021-10-31      44.0
    1 2011-02-28     93.0     130 2021-11-30      29.0
    2 2011-03-31     88.0     131 2021-12-31      40.0    (132, 2)  '''


print(marry_JR.info())
timeSeries = marry_JR.loc[:, ['yymm', 'marry']]
timeSeries.index = timeSeries.yymm
ts = timeSeries.drop("yymm", axis=1)
# print(ts)
'''               marry
     yymm             
    2011-01-31     82.0
    2011-02-28     93.0
    ...            ...
    2021-11-30     29.0
    2021-12-31     40.0    [132 rows x 1 columns]  '''


# # 2011-01부터 2021-12 까지 cpi(%) 그래프
# plt.figure(figsize=(15, 8))
# plt.plot(ts)
# plt.title("Marriage-Jongro 2011-01 ~ 2021-12")
# plt.xlabel("Year-Month")
# plt.ylabel("marry")
# plt.show()


# # seasonal_decompose()
from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(ts['marry'], model='additive')
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
''' ADF Statistic : -0.784390
    p-value : 0.823664
    Critical Values : 
           1%: -3.486
           5%: -2.886
          10%: -2.580            '''


# # 1차 차분
ts_diff = ts - ts.shift()
# plt.figure(figsize=(15, 8))
# plt.plot(ts_diff)
# plt.title("Marriage-Jongro 1st Differencing")
# plt.xlabel('Year-Month')
# plt.ylabel('marry')
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
    ADF Statistic : -8.222822
    p-value : 0.0
    Critical Values : 
           1%: -3.486
           5%: -2.886
          10%: -2.580            '''


# # auto_arima로 최적의 모형 탐색하기
import pmdarima as pm
train = marry_JR['marry'][:120]

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
''' Performing stepwise search to minimize aic
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=1014.427, Time=0.01 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=987.749, Time=0.05 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.08 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=1012.452, Time=0.01 sec
     ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=984.854, Time=0.05 sec
     ARIMA(3,1,0)(0,0,0)[0] intercept   : AIC=979.489, Time=0.07 sec
     ARIMA(4,1,0)(0,0,0)[0] intercept   : AIC=962.948, Time=0.09 sec
     ARIMA(5,1,0)(0,0,0)[0] intercept   : AIC=956.292, Time=0.13 sec
     ARIMA(5,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.28 sec
     ARIMA(4,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.24 sec
     ARIMA(5,1,0)(0,0,0)[0]             : AIC=955.727, Time=0.03 sec
     ARIMA(4,1,0)(0,0,0)[0]             : AIC=961.696, Time=0.03 sec
     ARIMA(5,1,1)(0,0,0)[0]             : AIC=951.573, Time=0.12 sec
     ARIMA(4,1,1)(0,0,0)[0]             : AIC=949.587, Time=0.07 sec
     ARIMA(3,1,1)(0,0,0)[0]             : AIC=952.651, Time=0.04 sec
     ARIMA(4,1,2)(0,0,0)[0]             : AIC=951.577, Time=0.09 sec
     ARIMA(3,1,0)(0,0,0)[0]             : AIC=977.759, Time=0.02 sec
     ARIMA(3,1,2)(0,0,0)[0]             : AIC=951.509, Time=0.08 sec
     ARIMA(5,1,2)(0,0,0)[0]             : AIC=953.535, Time=0.20 sec
    
    Best model:  ARIMA(4,1,1)(0,0,0)[0]          
    Total fit time: 1.712 seconds
'''
model_fit = model.fit(train)


# # 잔차 검정
print(model.summary())
'''                                SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  120
    Model:               SARIMAX(4, 1, 1)   Log Likelihood                -468.794
    Date:                Tue, 07 Jun 2022   AIC                            949.587
    Time:                        18:40:01   BIC                            966.262
    Sample:                             0   HQIC                           956.358
                                    - 120                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1         -0.1354      0.164     -0.827      0.408      -0.456       0.185
    ar.L2         -0.1189      0.116     -1.025      0.306      -0.346       0.108
    ar.L3         -0.2760      0.137     -2.014      0.044      -0.545      -0.007
    ar.L4         -0.2331      0.126     -1.852      0.064      -0.480       0.014
    ma.L1         -0.7442      0.121     -6.126      0.000      -0.982      -0.506
    sigma2       152.2348     23.075      6.597      0.000     107.009     197.460
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.18   Jarque-Bera (JB):                11.69
    Prob(Q):                              0.67   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.61   Skew:                             0.74
    Prob(H) (two-sided):                  0.12   Kurtosis:                         3.44
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''


# # ARIMA 모델 만들기
from statsmodels.tsa.arima.model import ARIMA
# fit model
model = ARIMA(ts, order=(4, 1, 1))
model_fit = model.fit()
# predict
start_index = yymm[120] # 2021-01-31 00:00:00
end_index = yymm[131]   # 2021-12-31 00:00:00
forecast = model_fit.predict(start=start_index, end=end_index, typ='levels')
# # 실제값이랑 비교하기
print(marry_JR.tail(12))
print(forecast)
'''        - 실제값 -       - forecast -
          yymm    marry        marry
    2021-01-31     40.0    46.167663
    2021-02-28     45.0    46.095590
    2021-03-31     41.0    44.361179
    2021-04-30     40.0    44.887139
    2021-05-31     45.0    45.940871
    2021-06-30     38.0    44.934832
    2021-07-31     38.0    44.707719
    2021-08-31     40.0    42.549521
    2021-09-30     38.0    42.281913
    2021-10-31     44.0    42.848020
    2021-11-30     29.0    41.923161
    2021-12-31     40.0    39.959259
'''

# # 시각화
# plt.figure(figsize=(15, 8))
# plt.plot(marry_JR.yymm, marry_JR['marry'], label="original")
# plt.plot(forecast, label='predicted')
# plt.title("Marriage-Jongro Forecast")
# plt.xlabel("Year-Month")
# plt.ylabel("marry")
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
# print(score_check(np.array(marry_JR[marry_JR.yymm>=start_index]['marry']), np.array(forecast)))
# ==>         R2   Corr   RMSE    MAPE
# ==>  0 -81.084  0.423  5.489  11.766





