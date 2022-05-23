# 데이터를 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# 강남구합계, 시계열데이터 정보 불러오기
apart = pd.read_csv('../datas/아파트2.csv', encoding='cp949', low_memory = False)
print(apart.head(2))
apart = apart[apart['구']=='강남구']
apart = apart.loc[:,['거래금액','ymd']]
print(apart)