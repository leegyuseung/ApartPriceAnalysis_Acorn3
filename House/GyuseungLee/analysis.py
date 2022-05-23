# # 데이터를 불러오기
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rc('font', family='malgun gothic')
#
# # 강남구합계, 시계열데이터 정보 불러오기
# apart = pd.read_csv('../datas/강남구합계.csv', encoding='utf-8')
# timedata = pd.read_csv('../datas/시간별 데이터 합.csv', encoding='utf-8')
# print(apart.head(2), timedata.head(2))
# x = apart['ymd']
# y = apart['price']
# # 강남구 가격변동 그래프
# plt.figure(figsize=(10,10))
# plt.plot(apart['ymd'], apart['price'])
# plt.title('강남구')
# plt.xlabel('date')
# plt.ylabel('price')
# plt.ylim(10000000,40000000)
# plt.xlim('2011-01','2011-12')
# plt.show()
