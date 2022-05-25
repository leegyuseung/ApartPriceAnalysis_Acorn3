'''
Created on 2022. 5. 25. 월별 데이터
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='malgun gothic')
pd.set_option('display.max_columns', None)

data1 = pd.read_csv('../datas/부동산 아파트 거래량 2011.01-2021.12.csv')
