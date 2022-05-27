'''
Created on 2022. 5. 26.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
pd.set_option('display.max_columns', None)

data = pd.read_csv('../datas/시간별 데이터 합.csv', encoding='cp949')

#시계열 데이터형 변형
data['ymd']= pd.to_datetime(data['ymd'],format='%Y%m')
data['ymd']=data['ymd'].dt.strftime('%Y%m')


# 집값 데이터 추가( 강남구)
apt = pd.read_csv('../datas/구별,월별 평당가격.csv', encoding='utf-8')
data['apt']=apt['강남구']

# 행정구별 인구수 데이터 추가
popul = pd.read_csv('../datas/행정구별_인구수.csv', encoding='cp949')
popul = popul.drop(0, axis=0)
data['popul']=popul['강남구']
print(data['popul'])


# 혼인건수 추가
marry = pd.read_excel('../datas/서울시 구별 혼인건수.xlsx')
data['marry']= marry['강남구']

# apt거래량 추가
trading = pd.read_excel('../datas/거래량.xlsx')
data['trading']= trading['강남구']

# print(data)