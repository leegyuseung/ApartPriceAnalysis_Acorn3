import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
plt.rc('font', family='malgun gothic')

data = pd.read_csv('../datas/시간별 데이터 합.csv', encoding='cp949')



#시계열 데이터형 변형
data['ymd']= pd.to_datetime(data['ymd'],format='%Y%m')
data['ymd']=data['ymd'].dt.strftime('%Y%m')
print(data.head(2))
print(data.info())

# 집값 데이터 추가( 강남구)
apt = pd.read_csv('구별,월별 평당가격.csv')
print(apt.head(3))
data['apt']=apt['강남구']
print(data.info())
print(data.head(2))


#변수별 시각화
for i in range(1,11):
    plt.plot(data['ymd'],data.iloc[:,i])
    plt.xlabel('ymd')
    plt.ylabel(data.columns[i])
    plt.show()

#df
mydata=data.iloc[:,1:11]
mydata.index=data['ymd']
print(mydata.head())
print(mydata.info())

# feature/ label

features = data.iloc[:,1:10]
# label = data.iloc[:,11]
mydata.plot(subplots=True)
plt.show()
#정상성 호
#differencing / 차분
mydata_diff = mydata.diff().dropna()

