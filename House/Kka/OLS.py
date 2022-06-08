import pandas as pd
import statsmodels.formula.api as smf
from pandas.core.frame import DataFrame


# # 종로구 평당 아파트 가격
df1 = pd.read_csv('구별,월별 평당가격.csv', parse_dates=['ymd'])
apt_price = df1['종로구']
# # m2(abs)
df2 = pd.read_csv('시간별 데이터 합.csv', parse_dates=['ymd'], encoding='cp949')
m2_abs = df2['m2(abs)']
# # cpi(abs)
df3 = pd.read_csv('시간별 데이터 합.csv', parse_dates=['ymd'], encoding='cp949')
cpi_abs = df3['cpi(abs)']
# # 종소세율
df4 = pd.read_csv('시간별 데이터 합.csv', parse_dates=['ymd'], encoding='cp949')
tax_jong = df4['종소세율']
# # marriage
df5 = pd.read_csv('../datas/서울시 구별 혼인건수.csv', parse_dates=['날짜'], encoding='utf-8-sig')
marry = df5['종로구']
# # population
df6 = pd.read_csv('../datas/행정구별_인구수.csv', encoding='cp949')
popul = df6['종로구']
popul[131] = popul[130]


# # 종로구 전체 데이터
yymm = []
yymm = pd.date_range("2011-01", "2022-01", freq="M")
Jongro = pd.DataFrame()
# Jongro['yymm'] = yymm
Jongro['price'] = apt_price
Jongro['m2_abs'] = m2_abs
Jongro['cpi_abs'] = cpi_abs
Jongro['tax_jongso'] = tax_jong
Jongro['marry'] = marry
Jongro['popul'] = popul
# Jongro.set_index('yymm', inplace=True)
print(Jongro.head(4))
print(Jongro.tail(4))


# # OLS 선형회귀
model = smf.ols(formula="price ~ m2_abs + cpi_abs + tax_jongso + marry + popul", data=Jongro).fit()
print(model.summary())

'''
# ymd 없을 때
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.927
    Model:                            OLS   Adj. R-squared:                  0.925
    Method:                 Least Squares   F-statistic:                     322.1
    Date:                Wed, 08 Jun 2022   Prob (F-statistic):           5.79e-70
    Time:                        02:19:52   Log-Likelihood:                -852.05
    No. Observations:                 132   AIC:                             1716.
    Df Residuals:                     126   BIC:                             1733.
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept  -5048.6925   2547.525     -1.982      0.050   -1.01e+04      -7.216
    m2_abs         0.0015      0.000      8.656      0.000       0.001       0.002
    cpi_abs      -23.3936     21.567     -1.085      0.280     -66.073      19.286
    tax_jongso  1.172e+05   6.14e+04      1.909      0.059   -4288.769    2.39e+05
    marry         -0.7076      1.226     -0.577      0.565      -3.134       1.719
    popul          0.0375      0.007      5.313      0.000       0.024       0.052
    ==============================================================================
    Omnibus:                       13.724   Durbin-Watson:                   1.347
    Prob(Omnibus):                  0.001   Jarque-Bera (JB):               35.858
    Skew:                          -0.268   Prob(JB):                     1.63e-08
    Kurtosis:                       5.497   Cond. No.                     1.10e+10
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.1e+10. This might indicate that there are
    strong multicollinearity or other numerical problems.
'''


# # 
df = pd.DataFrame()
df['m2_abs'] = [3633169.1173565, 3647923.56079389, 3662678.00423128, 3677432.44766867, 3692186.89110606, 3706941.33454346,
            3721695.77798085, 3736450.22141824, 3751204.66485563, 3765959.10829302, 3780713.55173041, 3795467.9951678]
df['cpi_abs'] = [104.05130087, 104.02293198, 104.19138319, 104.36221861, 104.51521816, 104.58789579,
            104.69300972, 104.8077501, 104.94854305, 105.06352354, 105.17947214, 105.28809655]
df['tax_jongso'] = [0.00313509, 0.00314526, 0.00315543, 0.0031656, 0.00317577, 0.00318593,
                0.0031961 , 0.00320627, 0.00321644, 0.00322661, 0.00323677, 0.00324694]
df['marry'] = [38.58635927, 39.85559518, 39.45437184, 37.34606242, 36.8513144, 36.29541801, 36.31815503,
                36.260246, 35.92403957, 35.48512105, 34.98279921, 34.5606616]
df['popul'] = [144943.67740534, 144750.21875869, 144556.76011204, 144363.30146538, 144169.84281873, 143976.38417207,
            143782.92552542, 143589.46687877, 143396.00823211, 143202.54958546, 143009.0909388, 142815.63229215]


predict_math = model.predict(DataFrame({'m2_abs':df['m2_abs'], 'cpi_abs': df['cpi_abs'], 'tax_jongso': df['tax_jongso'], 'marry': df['marry'], 'popul': df['popul']}))
pd.options.display.float_format = '{:.5f}'.format
print("예상 아파트 집값: ", predict_math)
'''
예상 아파트 집값:  0    3703.88020
              1    3719.52260
              2    3731.74266
              3    3745.11486
              4    3757.76257
              5    3772.33139
              6    3785.73315
              7    3798.96678
              8    3811.78786
              9    3825.28547
              10   3838.80412
              11   3852.43855
dtype: float64
'''
