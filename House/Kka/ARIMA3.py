import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import statsmodels
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA


# 데이터 불러오기
data = pd.read_csv('구별,월별 평당가격.csv', parse_dates=['ymd'])
Jongro = pd.DataFrame()
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
Jongro['yymm'] = yymm
Jongro['price'] = data['종로구']
Jongro.set_index('yymm', inplace=True)
print(Jongro.head(3), Jongro.tail(3))
print(Jongro.shape)


# 시계열 데이터 정상성 확인
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling().mean()
    rolstd = pd.Series(timeseries).rolling().std()
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
''' <Results of Dickey-Fuller Test>
    Test Statistic                   3.745713
    p-value                          1.000000
    #Lags Used                      13.000000
    Number of Observations Used    118.000000
    Critical Value (1%)             -3.487022
    Critical Value (5%)             -2.886363
    Critical Value (10%)            -2.580009
'''


# 관측값, 추세, 계절, 불규칙요인 시각화
plt.rcParams['figure.figsize'] = [12, 12]
result = seasonal_decompose(Jongro, model='additive')
result.plot()
plt.show()


# 시계열 정상성 여부 분석
from statsmodels.tsa.stattools import kpss

def kpss_test(data, **kw):    
    statistic, p_value, n_lags, critical_values = kpss(data, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The Jongro APT price is {"not " if p_value < 0.05 else ""} stationary')
    
kpss_test(Jongro)
''' KPSS Statistic: 1.695976233947563
    p-value: 0.01
    num lags: 6
    Critial Values:
       10%    : 0.347
        5%    : 0.463
        2.5%  : 0.574
        1%    : 0.739
    Result: The Jongro APT price is not  stationary
'''


# ACF, PACF 시각화
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams['figure.figsize'] = [9, 6]
plot_acf(Jongro)
plot_pacf(Jongro)
plt.show()


print()
# 차분
Jongro['first_diff'] = Jongro['price'] - Jongro['price'].shift(1)
test_stationarity(Jongro.first_diff.dropna(inplace=False))
plt.title('1st Differencing')


print()
# ARIMA 모델 생성
model = ARIMA(Jongro['price'], order=(1, 2, 0))
model_fit = model.fit(trend= 'c')
print(model_fit.summary())
''' 
    (p, d, q) = (1, 1, 0)
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                  price   No. Observations:                  132
    Model:                 ARIMA(1, 1, 0)   Log Likelihood                -857.840
    Date:                Fri, 03 Jun 2022   AIC                           1719.681
    Time:                        02:09:28   BIC                           1725.431
    Sample:                    01-31-2011   HQIC                          1722.017
                             - 12-31-2021                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1         -0.3822      0.047     -8.185      0.000      -0.474      -0.291
    sigma2       2.87e+04   2260.306     12.698      0.000    2.43e+04    3.31e+04
    ===================================================================================
    Ljung-Box (L1) (Q):                   1.27   Jarque-Bera (JB):                53.99
    Prob(Q):                              0.26   Prob(JB):                         0.00
    Heteroskedasticity (H):               4.45   Skew:                            -0.00
    Prob(H) (two-sided):                  0.00   Kurtosis:                         6.15
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    
    
    (p, d, q) = (1, 1, 0)
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                  price   No. Observations:                  132
    Model:                 ARIMA(0, 1, 1)   Log Likelihood                -853.458
    Date:                Fri, 03 Jun 2022   AIC                           1710.916
    Time:                        09:59:50   BIC                           1716.666
    Sample:                    01-31-2011   HQIC                          1713.253
                             - 12-31-2021                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ma.L1         -0.5284      0.044    -12.096      0.000      -0.614      -0.443
    sigma2      2.662e+04   2322.759     11.460      0.000    2.21e+04    3.12e+04
    ===================================================================================
    Ljung-Box (L1) (Q):                   0.04   Jarque-Bera (JB):                29.11
    Prob(Q):                              0.85   Prob(JB):                         0.00
    Heteroskedasticity (H):               4.94   Skew:                             0.12
    Prob(H) (two-sided):                  0.00   Kurtosis:                         5.30
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
'''




