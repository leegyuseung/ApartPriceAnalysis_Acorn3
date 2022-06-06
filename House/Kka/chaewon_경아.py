import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import scipy as sp
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rc('font', family='malgun gothic')
pd.set_option('display.max_columns', None)
import tensorflow as tf
from tensorflow import keras
from keras import layers


# # 데이터 불러오기
data = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/jain/House/datas/%EC%8B%9C%EA%B0%84%EB%B3%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%95%A9.csv', encoding='cp949')
# # 시계열 데이터형 변형
data['ymd']= pd.to_datetime(data['ymd'],format='%Y%m')
data['ymd']=data['ymd'].dt.strftime('%Y%m')
# print(data.head(5))
# # 집값 데이터 추가( 강남구)
apt = pd.read_csv('https://raw.githubusercontent.com/xerathul/FinalProject3/master/House/datas/%EA%B5%AC%EB%B3%84%2C%EC%9B%94%EB%B3%84%20%ED%8F%89%EB%8B%B9%EA%B0%80%EA%B2%A9_jain.csv', encoding='utf-8')
data['apt']=apt['강남구']
# print(data.head(5))
# # 행정구별 인구수 데이터 추가
popul = pd.read_csv('../datas/행정구별_인구수.csv', encoding='cp949')
# popul = popul.drop([popul.index[0]])
# popul = popul.drop(0, axis=0)
data['popul']=popul['강남구']
# print(data['popul'])
# # 혼인건수 추가
marry = pd.read_excel('../datas/서울시 구별 혼인건수.xlsx')
data['marry']= marry['강남구']
# print(data)
# # apt거래량 추가
trading = pd.read_csv('../datas/거래량.csv', encoding='cp949')
data['trading']= trading['강남구']
# # Index 설정
data.set_index('ymd', inplace=True)
# print(data.head(3))
# print(data.info())
# print(data.columns)
''' ['interest rating', 'm2(%)', 'm2(abs)', 'cpi(%)', 'cpi(abs)',
       'expected inflation', 'supply', '종소세율', '취득세율', 'apt', 'popul', 'marry',
       'trading'] '''



# # 표준화
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
pd.set_option('display.max_columns', None)
scaler1 = StandardScaler()
GN1 = scaler1.fit_transform(data)
# print('StandardScaler : ', GN1[:10])
# # 정규화
scaler2 = MinMaxScaler(feature_range=(0,1))
GN2 = scaler2.fit_transform(GN1)
np.set_printoptions(precision=6, suppress=True)
# print('MinMaxScaler : ', GN2[:10])


# # 상관계수 분석
index=[]
for i in list(range(2011, 2022, 1)):
     for j in list(range(1, 13, 1)):
         # print(str(i)+str(j).zfill(2))
         index.append(str(i)+str(j).zfill(2))
df = pd.DataFrame(GN2, index = index, 
                  columns=['interest rating', 'm2(%)','m2_abs','cpi(%)','cpi_abs','expected_inflation','supply','tax_jongso','tax_chuideuk','price','trading','marry','popul'])
df = df.dropna()
print(df)
# print(df.corr())
# 0.9628(m2(abs)), 0.9117(cpi(abs)), 0.8912(tax_jongso), -0.8956(trading), -0.7959(marry)



# # 다중 선형 회귀
data2 = df.iloc[:, [9, 2, 4, 7, 10, 11]]
# print(data2.columns)    # 'price', 'm2_abs', 'cpi_abs', 'tax_jongso', 'trading', 'marry'
import statsmodels.formula.api as smf
result = smf.ols('price ~ m2_abs + m2_abs + tax_jongso + trading + marry', data = data2).fit()
print(result.summary())
'''                             OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.963
    Model:                            OLS   Adj. R-squared:                  0.962
    Method:                 Least Squares   F-statistic:                     826.2
    Date:                Sat, 04 Jun 2022   Prob (F-statistic):           2.42e-89
    Time:                        01:56:07   Log-Likelihood:                 196.50
    No. Observations:                 131   AIC:                            -383.0
    Df Residuals:                     126   BIC:                            -368.6
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      0.2227      0.037      6.080      0.000       0.150       0.295
    m2_abs         0.6156      0.042     14.549      0.000       0.532       0.699
    tax_jongso     0.1138      0.024      4.662      0.000       0.066       0.162
    trading       -0.2390      0.034     -6.960      0.000      -0.307      -0.171
    marry         -0.0773      0.042     -1.845      0.067      -0.160       0.006
    ==============================================================================
    Omnibus:                        1.049   Durbin-Watson:                   0.910
    Prob(Omnibus):                  0.592   Jarque-Bera (JB):                1.113
    Skew:                           0.136   Prob(JB):                        0.573
    Kurtosis:                       2.639   Cond. No.                         16.6
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
'''
# print('p-value : ', result.pvalues[1])  # p-value :  9.495295659144786e-29


# # 추정치 구하기
pred = result.predict()
print(pred, len(pred))  # 131




# # 독립변수, 종속변수 추출
# x = df.iloc[:, [2, 4, 7, 10, 11]]
# y = df.iloc[:, 9]
# print(x.head(2), x.info())
# print(y.head(2))
# feature_np = x.to_numpy()
# label_np = y.to_numpy()
# # print(feature_np, feature_np.shape)
# # print(label_np)
########################################################



"""
from sklearn.linear_model import LinearRegression

X=pd.DataFrame()
y=pd.DataFrame()

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
lr_skl=LinearRegression(fit_intercept=False) # default가 fit_intercept=True
lr_skl.fit(X_train, y_train)
y_pred_skl=lr_skl.predict(X_test)

lr_stat=sm.OLS(y_train, X_train).fit()
y_pred_stat=lr_stat.predict(X_test)

test_mse_stat=mean_squared_error(y_test, y_pred_stat)
test_rmse_stat=np.sqrt(mean_squared_error(y_test, y_pred_stat))
test_mae_stat=mean_absolute_error(y_test, y_pred_stat)
#test_mape_stat=mean_absolute_percentage_error(y_test, y_pred_stat)
test_r2_stat=r2_score(y_test, y_pred_stat)

print('Testing MSE:{:,3f}'.format(test_mse_stat))
print('Testing RMSE:{:,3f}'.format(test_rmse_stat))
print('Testing MAE:{:,3f}'.format(test_mae_stat))
#print('Testing MAPE:{:,3f}'.format(test_mape_stat))
#print('Testing R2:{:,3f}'.format(test_r2_stat)

###########################################################################
"""



