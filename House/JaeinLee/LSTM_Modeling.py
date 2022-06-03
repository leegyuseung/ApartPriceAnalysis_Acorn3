'''
Created on 2022. 6. 1.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

plt.rc('font', family='malgun gothic')
pd.set_option('display.max_columns', None)

data = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/jain/House/datas/%EC%8B%9C%EA%B0%84%EB%B3%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%95%A9.csv', encoding='cp949')

#시계열 데이터형 변형
data['ymd']= pd.to_datetime(data['ymd'],format='%Y%m')
data['ymd']=data['ymd'].dt.strftime('%Y%m')
# print(data.head(5))

# 집값 데이터 추가( 강남구)
apt = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/jain/House/datas/%EA%B5%AC%EB%B3%84%2C%EC%9B%94%EB%B3%84%20%ED%8F%89%EB%8B%B9%EA%B0%80%EA%B2%A9.csv', encoding='utf-8')
data['apt']=apt['강남구']
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

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
pd.set_option('display.max_columns', None)

# 표준화
scaler1 = StandardScaler()
GN1 = scaler1.fit_transform(data)
print('StandardScaler : ', GN1[:10])

# 정규화
scaler2 = MinMaxScaler(feature_range=(0,1))
GN2 = scaler2.fit_transform(GN1)
np.set_printoptions(precision=6, suppress=True)
print('MinMaxScaler : ', GN2[:10])

# 상관계수 분석
index=[]

for i in list(range(2011, 2022, 1)):
    for j in list(range(1, 13, 1)):
        # print(str(i)+str(j).zfill(2))
        index.append(str(i)+str(j).zfill(2))

df = pd.DataFrame(GN2, index = index, 
                  columns=['ymd','interest rating', 'm2(%)','m2(abs)','cpi(%)','cpi(abs)','expected_inflation','supply','tax_jongso','tax_chuideuk','price','trading','marry','popul'])
df = df.dropna()
print(df)
print(df.corr()) # 0.925312(ymd), 0.962813(m2(abs)), 0.911707(cpi(abs)), 0.891248(tax_jongso), -0.895664(trading)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

# 독립변수, 종속변수 추출
x = df.iloc[:, [0, 3, 5, 8, 11]]
y = df.iloc[:, 10]
print(x)
print(y)
feature_np = x.to_numpy()
label_np = y.to_numpy()
print(feature_np, feature_np.shape)
print(label_np)

window_size = 100
# print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # (92, 13) (40, 13) (92,) (40,)

# 입력 파라미터 feature, lable => numpy type

def make_sequence_dataset(feature, label, window_size):
    feature_list = []   # 생성될 feature list
    label_list = []     # 생성될 label list

    for i in range(len(feature)-window_size):
        
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])
    return np.array(feature_list), np.array(label_list)
X, Y = make_sequence_dataset(feature_np, label_np, window_size)

print(X.shape, Y.shape)

# training test split
split = -10

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 모델링

model = keras.Sequential()
model.add(layers.LSTM(units = 128, activation = 'tanh', input_shape = x_train[0].shape))
# model.add(keras.layers.Dense(units = 64, activation = 'relu'))
# model.add(keras.layers.Dense(units = 32, activation = 'relu'))
# model.add(keras.layers.Dense(units = 16, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))

model.summary()

# 모델 컴파일
from keras.callbacks import EarlyStopping
import tensorflow as tf


epochs = 50


model.compile(optimizer=keras.optimizers.Adam(0.0001),
              loss =tf.keras.losses.MeanSquaredError(),
              metrics = ['mse'])

early_stop = EarlyStopping(monitor='val_loss', patience = 5)

# 모델 학습
model.fit(x_train, y_train, epochs = epochs, validation_data = (x_test, y_test), batch_size = 10, callbacks=[early_stop])

# 시각화
pred = model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.plot(pred, label = 'predict')
plt.plot(y_test, label = 'actual')
plt.legend(loc='best')

plt.show()