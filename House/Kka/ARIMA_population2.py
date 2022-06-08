import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from IPython.core.pylabtools import figsize
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
import pmdarima as pm

# 데이터 불러오기
data = pd.read_csv('../datas/행정구별_인구수.csv', parse_dates=['시점'], encoding='cp949')
popul_JR = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2023-01", freq="M")
popul_JR['yymm'] = yymm[:132]
popul_JR['popul'] = data['종로구']
popul_JR['popul'][131] = popul_JR['popul'][130]
# print(yymm[132:144])
print(popul_JR.head(3), popul_JR.tail(3), popul_JR.shape)
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


# # 차분이 꼭 필요한지 알아보기
from pmdarima.arima import ndiffs
# kpss_diffs = ndiffs(popul_JR['popul'], alpha=0.05, test='kpss', max_d=6)
# adf_diffs = ndiffs(popul_JR['popul'], alpha=0.05, test='adf', max_d=6)
# n_diffs = max(adf_diffs, kpss_diffs)
#
# print(f"추정된 차수 d = {n_diffs}")
# # --> 추정된 차수 d = 1


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


# # auto_arima로 최적의 모형 탐색하기
train = popul_JR.popul[:120]

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
     ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=1650.326, Time=0.12 sec
     ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=1650.295, Time=0.01 sec
     ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=1644.963, Time=0.07 sec
     ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=1644.783, Time=0.08 sec
     ARIMA(0,1,0)(0,0,0)[0]             : AIC=1699.024, Time=0.00 sec
     ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=1646.681, Time=0.12 sec
     ARIMA(0,1,2)(0,0,0)[0] intercept   : AIC=1646.692, Time=0.11 sec
     ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=1648.765, Time=0.08 sec
     ARIMA(0,1,1)(0,0,0)[0]             : AIC=1697.730, Time=0.01 sec
    
    Best model:  ARIMA(0,1,1)(0,0,0)[0] intercept
    Total fit time: 0.606 seconds
'''

# # 잔차 검정
# print(model.summary())
'''                                SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                  120
    Model:               SARIMAX(0, 1, 1)   Log Likelihood                -819.391
    Date:                Tue, 07 Jun 2022   AIC                           1644.783
    Time:                        17:04:06   BIC                           1653.120
    Sample:                             0   HQIC                          1648.168
                                    - 120                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept   -183.8261     23.232     -7.912      0.000    -229.361    -138.291
    ma.L1          0.0780      0.015      5.284      0.000       0.049       0.107
    sigma2      5.162e+04   2820.379     18.302      0.000    4.61e+04    5.71e+04
    ===================================================================================
    Ljung-Box (L1) (Q):                  34.72   Jarque-Bera (JB):               484.51
    Prob(Q):                              0.00   Prob(JB):                         0.00
    Heteroskedasticity (H):               0.20   Skew:                             1.13
    Prob(H) (two-sided):                  0.00   Kurtosis:                        12.62
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''

# # 그래프
# model.plot_diagnostics(figsize=(16, 8))
# plt.show()


# # 미래값 예측
fore = model_fit.forecast(steps=12)
print(fore)




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
''' (p, d, q) = (0, 1, 1) 로 바꿨을 때 성능 '''
#        R2   Corr     RMSE   MAPE
# 0  83.911  0.964  539.172  0.237




