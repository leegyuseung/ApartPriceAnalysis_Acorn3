'''
Created on 2022. 6. 2.
@author: Jain
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

plt.rc('font', family='malgun gothic')
pd.set_option('display.max_columns', None)
data = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/jain/House/datas/%EC%8B%9C%EA%B0%84%EB%B3%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%95%A9.csv', encoding='cp949')

#시계열 데이터형 변형
print(data.info())
data['ymd']= pd.to_datetime(data['ymd'],format='%Y%m')
print(data.head())
print(data.info())
data['ymd']=data['ymd'].dt.strftime('%Y%m')
print(data.head(5))
print(data.info())

# 집값 데이터 추가( 강남구)
apt = pd.read_csv('..\datas\구별,월별 평당가격_jain.csv', encoding='utf-8')
data['price']=apt['강남구']
# print(data.head(5))

# 행정구별 인구수 데이터 추가
popul = pd.read_csv('../datas/행정구별_인구수.csv', encoding='cp949')
# popul = popul.drop([popul.index[0]])
# popul = popul.drop(0, axis=0)
data['popul']=popul['강남구']
# print(data['popul'])

# 혼인건수 추가
marry = pd.read_excel('../datas/서울시 구별 혼인건수.xlsx')
data['marry']= marry['강남구']
# print(data)

# apt거래량 추가
trading = pd.read_csv('../datas/거래량.csv', encoding='cp949')
data['trading']= trading['강남구']
# print(data.head(5))

# 데이터 shape 파악
data.shape
# 데이터 통계량 파악
data.describe()
data.info()
# 데이터 null값 파악
# data.isnull().sum()
# Null값 있는 행 제거
data1 = data.drop(data.index[131])
print(data1)

# 데이터 분포 시각화
sns.pairplot(data1)
plt.title("Data의 Pair Plot")
plt.show()

# 집값 변화 시각화
def plot_data(data, x, y, xlabel='Date', ylabel='Value'):
    plt.figure(figsize=(16, 5))
    plt.plot(x, y)
    plt.gca().set(xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_data(data1, x=data['ymd'], y=data['price'])

# 표준 상관계수(피어슨 상관계수)
data1.corr() # m2(abs) : 0.962813, cpi(abs)    : 0.911707, 종소세율 : 0.891248, popul : -0.895664    

# train test split
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data1, test_size = 0.2, random_state = 42)
print(train_set.shape, test_set.shape) # (105, 10) (27, 10)

# feature, label 분리
data_feature_train = train_set.loc[:, ['m2(abs)','cpi(abs)','종소세율','popul']]
data_label_train = train_set.loc[:, "price"]
print(data_feature_train, data_label_train)

data_feature_test = test_set.loc[:, ['m2(abs)','cpi(abs)','종소세율','popul']]
data_label_test = test_set.loc[:, "price"]
print(data_feature_test, data_label_test)

data.plot(subplots=True, figsize = (12, 20))

# 수치형 특성별 히스토그램
data.hist(bins=50, figsize=(20,15))
plt.show()

# train test 데이터를 정규화
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
pd.set_option('display.max_columns', None)
scaler = MinMaxScaler()
data_feature_train_scale = scaler.fit_transform(data_feature_train)
data_label_train_scale = scaler.fit_transform(data_label_train.values.reshape(-1, 1))
print('MinMaxScaler : ', data_feature_train_scale[:10])
print('MinMaxScaler : ', data_label_train_scale[:10])

# 1. 선형회귀
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(data_feature_train_scale, data_label_train)
lin_pred = lin_reg.predict(data_feature_train_scale)
print('예측값 : ', lin_pred[:5])   # 예측값은 test data를 넣어야 하는거 아닌가? 왜 train 정규화 값을 넣어야 제대로 된 값이 나오는가?
print('실제값 : ', data_label_test[:5])
# RMSE 값
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(data_label_train, lin_pred)   # mse : 평균 제곱 오차. 예측값과 실제값의 차이를 모두 더해 평균을 낸다. 회귀에서 자주 사용되는 손실 함수. 정확도 X
lin_rmse = np.sqrt(lin_mse)
print('RMSE : ', lin_rmse)

# MAE 값
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(data_label_train, lin_pred)   # mae : 평균 절대 오차. 모든 절대 오차의 평균. 일반적인 회귀 지표
print('MAE : ', lin_mae)

# 시각화

# 2. Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_feature_train_scale, data_label_train)
tree_pred = tree_reg.predict(data_feature_train_scale)
print('예측값 : ', tree_pred[:5])
print('실제값 : ', data_label_test[:5])

# RMSE 값
tree_mse = mean_squared_error(tree_pred, data_label_train)
tree_rmse = np.sqrt(tree_mse)
print('RMSE : ', tree_rmse)  # 예측값과 실제값은 차이가 심한데 RMSE는 0. 모델 예측값과 실제값의 차이를 다를 때 사용하는 측도.

# 시각화

# 3. Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(data_feature_train_scale, data_label_train)
forest_pred = forest_reg.predict(data_feature_train_scale)
print('예측값 : ', forest_pred[:5])
print('실제값 : ', data_label_test[:5])

# RMSE 값
forest_mse = mean_squared_error(forest_pred, data_label_train)
forest_rmse = np.sqrt(forest_mse)
print('RMSE : ', forest_rmse)

# 4. XG Boost
import xgboost as xgb

xgb_model = xgb.XGBRegressor(booster = 'gblinear', max_depth = 6, n_estimators=100).fit(data_feature_train_scale, data_label_train)
xgb_pred = xgb_model.predict(data_feature_train_scale)
print('예측값 : ', xgb_pred[:5])
print('실제값 : ', data_label_test[:5])
