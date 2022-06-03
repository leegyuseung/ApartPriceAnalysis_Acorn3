import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ☆ ☆ 시간별 데이터 불러오기
# 변수 칼럼명 : 'ymd'  |  'interest rating', 'm2(%)', 'm2(abs)', 'cpi(%)', 'cpi(abs)', 'expected inflation', 'supply'
time_df = pd.read_csv("../datas/시간별 데이터 합.csv", encoding='cp949')
# print(time_df.head(5), time_df.tail(5))
print(time_df.shape)    # (132, 8)

# ☆ ☆ 아파트 실거래가 데이터 불러오기    # NaN 값 없음!!
price_df = pd.read_csv("../datas/아파트2.csv",  encoding ='cp949')
ydpP_df = price_df[price_df['구'] == '영등포구']
print(ydpP_df.head(5), ydpP_df.shape)   # (36338, 8)
# print(ydpP_df.columns)
# '거래금액', '건축년도', '년', '구', '법정동', '아파트', '월', '일', '전용면적', '지번', '지역코드', '층', 'ymd'
cols = ['거래금액', '구', '전용면적', 'ymd']
ydpP_df = ydpP_df[cols]
print()
print(ydpP_df.head(5))
print()
print(ydpP_df.info())
# 법정동 별로 데이터 묶기
# gugu = ydpP_df.groupby(ydpP_df['법정동'])
# print(gugu.size())


# ☆ ☆ 시간별 데이터와 아파트 실거래가 데이터 합치기
TTdf = pd.merge(time_df, ydpP_df)
print(TTdf.head(5),TTdf.tail(5), TTdf.shape) # (36724, 14)
print(TTdf.columns)
# 시계열 데이터 형변환
TTdf['ymd']= pd.to_datetime(TTdf['ymd'],format='%Y%m')
TTdf['ymd']=TTdf['ymd'].dt.strftime('%Y%m') # (편집됨)

print(TTdf.info())
print(TTdf['ymd'][:10])


