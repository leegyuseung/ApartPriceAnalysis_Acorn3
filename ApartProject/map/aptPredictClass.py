import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from statsmodels.regression.linear_model import RegressionModel
from statsmodels import formula
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
plt.rc('font', family='malgun gothic')


class AptPred:
    def __init__(self):
        pass

    
    def predictModel(self, gu, ym, df):
        pd.set_option('display.max_columns', None)
        # 시간별 데이터.csv 합 불러오기
        data = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/jain/House/datas/%EC%8B%9C%EA%B0%84%EB%B3%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%95%A9.csv', encoding='cp949')
        
        #시계열 데이터형 변형
        data['ymd']= pd.to_datetime(data['ymd'],format='%Y%m')
        date= pd.to_datetime(data['ymd'],format='%Y%m')
        data['ymd']=data['ymd'].dt.strftime('%Y%m')
        
        # 여기를 우리 db price로 바꿔야댐
        
        data['price']=df['price']
        # print(data.head())
        # 행정구별_인구수.csv 데이터 추가
        popul = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/master/House/datas/%ED%96%89%EC%A0%95%EA%B5%AC%EB%B3%84_%EC%9D%B8%EA%B5%AC%EC%88%98.csv', encoding='cp949')
        data['popul']=popul[gu]
        
        # 혼인건수 추가
        marry = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/master/House/datas/%EC%84%9C%EC%9A%B8%EC%8B%9C%20%EA%B5%AC%EB%B3%84%20%ED%98%BC%EC%9D%B8%EA%B1%B4%EC%88%98.csv')
        data['marry']= marry[gu]
        
        # apt거래량 추가
        trading = pd.read_csv('https://raw.githubusercontent.com/Loyce0805/test333/master/House/datas/%EA%B1%B0%EB%9E%98%EB%9F%89.csv', encoding='cp949')
        data['trading']= trading[gu]
        
        # ymd column을 index로 지정
        data = data.set_index('ymd')
        
        # 종소세율을 tax_jongso로 컬럼명 변경
        data.rename(columns = {'종소세율' : 'tax_jongso'}, inplace = True)
        
        # Null값 있는 행 제거
        data1 = data.dropna()
        
        # train test split
        train_data, test_data = train_test_split(data1, test_size = 0.2, shuffle = True)
        
        # feature, label 분리
        train_data = train_data.loc[:, ['m2(abs)','cpi(abs)', 'tax_jongso','popul', 'marry','interest rating', 'price']]
        test_data = test_data.loc[:, ['m2(abs)','cpi(abs)', 'tax_jongso','popul', 'marry', 'interest rating', 'price']]
        train_data.columns = ['m2_abs', 'cpi_abs', 'tax_jongso', 'popul', 'marry','interest_rating', 'price']
        test_data.columns = ['m2_abs', 'cpi_abs', 'tax_jongso', 'popul', 'marry','interest_rating', 'price']
        
        # 1-1 ols(최소자승법)을 이용한 선형회귀
        result = sm.OLS.from_formula('price ~ m2_abs + tax_jongso + cpi_abs + marry + popul', data = train_data).fit()  # interesting rate를 feature로 넣으니 cpi의 pvalue값이 0.05를 넘음. 그래서 뺐음.
        ols_pred = result.predict(test_data)
        # print('price : ', ols_pred)
        # print('예측값 : ', ols_pred[:5])
        # print('실제값 : ', test_data['price'][:5])
        # print(gu)
        
        #if 문으로 구별 ARIMA 데이터 가져오기
        # decode = (self.gu).decode('utf-8', 'ignore')
        from urllib.parse import quote
        decode = quote(gu)
        arimaLink = 'https://raw.githubusercontent.com/Loyce0805/forecast-Dataset/main/datas/data/'+decode+'.csv'
        # print(arimaLink)
        arima = pd.read_csv(arimaLink)

        predict_math = result.predict(arima)
        pd.options.display.float_format = '{:.5f}'.format
        # print('예상 집값 : ', predict_math)
        
        #2022-2027 prediction
        index = []
        for i in list(range(2022, 2027, 1)):
            for j in list(range(1, 13, 1)):
                index.append(str(i)+'-'+str(j).zfill(2))
        
        predict_math = pd.DataFrame(predict_math)
        predict_math['ymd'] = index
        predict_math = predict_math.set_index(['ymd'])
        predict_math.columns = ['Predict price']
        
        r2 = r2_score(test_data['price'], ols_pred)

        # print(r2)
        resultPredict = predict_math.loc[:ym]
        # print(resultPredict)
        # print(predict_math)
        return resultPredict
    

