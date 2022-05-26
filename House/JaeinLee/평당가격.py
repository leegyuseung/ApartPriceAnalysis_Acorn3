'''
Created on 2022. 5. 26.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# 데이터를 불러오기
apart = pd.read_csv('../datas/아파트2.csv', encoding='cp949')
seoul = [['강남구', '11680'], ['강동구', '11740'], ['강북구', '11305'], ['강서구', '11500'], ['관악구', '11620'], ['광진구', '11215'], ['구로구', '11530'], ['금천구', '11545'], ['노원구', '11350'], ['도봉구', '11320'], ['동대문구', '11230'], ['동작구', '11590'], ['마포구', '11440'], ['서대문구', '11410'], ['서초구', '11650'], ['성동구', '11200'], ['성북구', '11290'], ['송파구', '11710'], ['양천구', '11470'], ['영등포구', '11560'], ['용산구', '11170'], ['은평구', '11380'], ['종로구', '11110'], ['중구', '11140'], ['중랑구', '11260']]

apt1 = apart[apart['구'] == '강남구']
apt1 = apart.loc[:,['거래금액','ymd','전용면적']]
apt1['면적가격'] = (apt1['거래금액']/apt1['전용면적']) * 3.3
apt1['평균값'] = apt1.groupby(['ymd'])['면적가격'].transform('mean')
apt2 = apt1.drop_duplicates(['ymd'], ignore_index=True)
apart3 = apt2.copy()

for city, _ in seoul: 
    apart1 = apart[apart['구'] == city]
    # print(apart1.head(1))
    apart1 = apart1.loc[:,['거래금액','ymd','전용면적']]
    apart1['면적가격'] = (apart1['거래금액']/apart1['전용면적']) * 3.3
    apart1['평균값'] = apart1.groupby(['ymd'])['면적가격'].transform('mean')
    print(apart1.head(1))
    apart2 = apart1.drop_duplicates(['ymd'], ignore_index=True)
    # print(apart2['평균값'])
    apart3[city] = apart2['평균값']

apart3 = apart3.drop(['거래금액','전용면적','평균값','면적가격'], axis="columns")

print(apart3.head(3))
apart3.to_csv('구별,월별 평당가격2.csv', index=False, encoding ='utf-8-sig')