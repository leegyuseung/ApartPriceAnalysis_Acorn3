import json            
import requests        
import xmltodict       
import pandas as pd    

APIKEY = "janG8FZzVlFCaTco2h3fKdlm9%2B3nn1gOaLMRkjcM9CczaBaE%2B1rat9qQQTuTNKcP4KAyPYLM3Bn80l%2FRQkT6%2Bw%3D%3D"

def get_data(lawd_cd, deal_ymd):
    base_url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade?serviceKey=" +APIKEY
    base_url += f'&LAWD_CD={lawd_cd}'
    base_url += f'&DEAL_YMD={deal_ymd}'
    res = requests.get(base_url)
    data = json.loads(json.dumps(xmltodict.parse(res.text)))
    # print(data)
    df = pd.DataFrame(data['response']['body']['items']['item'])
    # print(df.info)
    return df

# get_data(11680, 202101)

seoul = [['강남구', '11680'], ['강동구', '11740'], ['강북구', '11305'], ['강서구', '11500'], ['관악구', '11620'], ['광진구', '11215'], ['구로구', '11530'], ['금천구', '11545'], ['노원구', '11350'], ['도봉구', '11320'], ['동대문구', '11230'], ['동작구', '11590'], ['마포구', '11440'], ['서대문구', '11410'], ['서초구', '11650'], ['성동구', '11200'], ['성북구', '11290'], ['송파구', '11710'], ['양천구', '11470'], ['영등포구', '11560'], ['용산구', '11170'], ['은평구', '11380'], ['종로구', '11110'], ['중구', '11140'], ['중랑구', '11260']]
print(len(seoul))
date = pd.date_range('20211201', '20211231', freq='MS').strftime('%Y%m')

apt = pd.DataFrame()
for name, code in seoul:
    sgg = pd.DataFrame()
    for ym in date:
        temp = get_data(code, ym)
        sgg = pd.concat([sgg, temp])
    sgg['시군구명'] = name
    apt = pd.concat([apt, sgg])
apt['시도명'] = '서울특별시'
print(apt.info())
apt.to_csv('서울특별시 아파트 실거래가 202212.csv', index=False, encoding ='utf-8-sig')
# apt.to_csv(path_or_buf, sep, na_rep, float_format, columns, header, index, index_label, mode, encoding, compression, quoting, quotechar, line_terminator, chunksize, date_format, doublequote, escapechar, decimal, errors, storage_options)
