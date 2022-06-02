import pandas as pd

data = pd.read_csv('../datas/서울데이터완.csv')

guList = []
dongList = []
BunjiList = []

for i in range(len(data)):
    guList.append(data['구'][i])
    dongList.append(data['법정동'][i])
    BunjiList.append(data['지번'][i])

addList = []

for j in range(len(data)):
    addList.append('서울시 ' + guList[j] + ' ' + dongList[j] + ' ' + BunjiList[j])

addr = pd.DataFrame(addList)

data['주소'] = addr

data.to_csv("../datas/서울데이터완2.csv", mode='w', encoding='utf-8-sig')