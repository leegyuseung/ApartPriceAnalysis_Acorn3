import pandas as pd
import requests
from xml.etree import ElementTree as ET
import xmltodict
#  행정구별 코드
df = pd.read_csv('../datas/법정동코드 조회자료.csv')
print(df.head())
dong=df['법정동코드'].to_list()
print(dong[:3], type(dong))
for i in dong:
    
    APIKEY = "zOVsayBFwURddA1RPDX0L2MzMSH9k7Ivls+AztXTeudlmP4d7Z/zL50JEdI4Eu82mhRBur/IYHJgb5GJR8jpzQ=="
    URL = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade"
    CODE = i
    START = "2012-01"
    END = "2021-12"
    
    data = []
    for ymd in pd.date_range(START, END, freq="1M").strftime("%Y%m").to_list():
        response = requests.get(
            URL,
            params={
                'serviceKey' : APIKEY,
                'LAWD_CD' : CODE,
                'DEAL_YMD' : ymd
            }
        )
        df = pd.DataFrame.from_dict(xmltodict.parse(ET.tostring(ET.fromstring(response.text)[1][0], encoding='unicode')).get('items').get('item'))
        df['ymd'] = ymd
        data.append(df)
    data = pd.concat(data)
print(data.info())
    

new_df = pd.DataFrame(df, columns=['ymd','법정동','아파트','전용면적','거래금액'])
print(new_df.head(3))
new_df.to_csv('apt1212.csv',encoding ='utf-8-sig')