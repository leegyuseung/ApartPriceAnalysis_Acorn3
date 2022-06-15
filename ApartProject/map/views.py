from django.shortcuts import render
from map.models import Addrdata, Addrapt
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.http.response import JsonResponse
from django.core.paginator import Paginator
import requests
import json
import numpy as np
from map.aptPredictClass import AptPred


def Main(request):
    return render(request,'index.html')

# AJAX 받는 함수
@csrf_exempt
def apart(request):
    # 데이터 검색 받기
    search = request.GET['search']
    
    # DB에 검색
    datas = Addrapt.objects.filter(apt__contains=search).values()
    df = pd.DataFrame(datas)
    
    # 아파트명, 주소 저장
    apt = [i for i in df['apt'] + df['dong']]
    juso = [i for i in df['addr']]
    
    # 아파트명, 주소 json형식 변경
    aptJusoJson = {}
    for apt, juso in zip(apt, juso):
        aptJusoJson[apt]=  juso 
    
    apt = [i for i in df['apt'] + df['dong']]
    
    # 페이징처리
    paginator = Paginator(apt, 10)
    page = int(request.GET.get('page', 1))
    apt_lists = paginator.get_page(page)
    
    tojson = {'isPrev':apt_lists.has_previous(),
               'range': [i for i in range(1,apt_lists.paginator.num_pages+1)],
              'num':apt_lists.number,'apt_list':apt_lists.object_list,
              'isNext':apt_lists.has_next(), 'num_pages':apt_lists.paginator.num_pages }
    
    if apt_lists.has_previous() ==True:
        prevNum=apt_lists.previous_page_number()
        tojson['prevNum']=prevNum
    if apt_lists.has_next() == True:
        nextNum = apt_lists.next_page_number()
        tojson['nextNum']=nextNum

    # json으로 리턴
    return JsonResponse({'juso':juso, 'apartdata':apt, 'aptJusoJson':aptJusoJson, 'apt_lists':tojson})

@csrf_exempt
def polygon(request):
  
    open('Gpolygon.json','wb').write(requests.get('https://raw.githubusercontent.com/xerathul/FinalProject3/master/ApartProject/map/static/resources/polygonData.json').content)

    with open ('Gpolygon.json',encoding='utf-8') as f:
        Gpolygon = json.load(f)
    
    return JsonResponse({'polygon':Gpolygon})

@csrf_exempt
def Dpolygon(request):
   
    open('Dpolygon.json','wb').write(requests.get('https://raw.githubusercontent.com/xerathul/FinalProject3/master/ApartProject/map/static/resources/polygonDong.json').content)

    with open ('Dpolygon.json',encoding='utf-8') as f:
        Dpolygon = json.load(f)
    
    return JsonResponse({'Dpolygon':Dpolygon})

# graph predict ajax Function
@csrf_exempt
def pred(request):
    year = request.POST.get('year')
    gu = request.POST.get('gu')
    addr = request.POST.get('addr')
    df = createPredDf(addr)
    pred = AptPred()
    
    predict = pred.predictModel(gu=gu, ym=year, df=df)['predPrice']
    predict = predict

    predict1 = []
    predictlist = predict['Predict price'].values
    for i in range(len(predictlist)):
        predict1.append(predictlist[i])
    
    ymd2 = []
    for i in range(len(predict.index)):
        ymd2.append(predict.index[i])
            
    for i in range(len(ymd2)):
            ymd2[i] = ymd2[i].replace('-','')
        
    for i in range(len(ymd2)):
            ymd2[i] = int(ymd2[i])
    
    print(ymd2)
    print(predict1)
    
    r2 = pred.predictModel(gu=gu, ym=year, df=df)['r2']
    predictL = round(predict1[-1])
    return JsonResponse({'new_val':year, 'predict':predict1, 'ymd2':ymd2, 'r2':r2, 'predictL':predictL})


def createPredDf(addr):

        datas = Addrdata.objects.filter(addr__contains=addr).values()
        
        df = pd.DataFrame(datas)
        df.set_index(df['num'], inplace=True)
        df = df.drop(['num'], axis=1)
        df = df.sort_values(by = 'ymd')
        
        print(df['apt'][:1].values[0])
        apt = df['apt'][:1].values[0]
        year = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        mon = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        ymd = []  # ['201101', '201102', '201103', '201104' ...
        
        for i in range(len(year)):
            for j in range(len(mon)):
                ymd.append(year[i] + mon[j])
        
        mean = []
        for i in ymd:
            condition = (df['ymd']== (i)) # 조건식 작성
        
            mp = list(df[condition].price) # 월 별 거래가격 리스트
            
            if len(mp) == 0:
                mean.append(np.nan)
        
            else: 
                mv = round(sum(mp)/len(mp))
                mean.append(mv)
       
        ddf = pd.DataFrame({'날짜':ymd})
        ddff = pd.DataFrame({'price':mean})
        dfdf = pd.concat([ddf, ddff], axis=1)
        
        return dfdf
      
def importData(request):
    # 클릭한 마커의 데이터 불러오기
    if request.method == 'GET':
        
        detailaddr = request.GET['aptName']
        
        datas = Addrdata.objects.filter(addr__contains=detailaddr).values()
        
        df = pd.DataFrame(datas)
        df.set_index(df['num'], inplace=True)
        df = df.drop(['num'], axis=1)
        df = df.sort_values(by = 'ymd')
        
        apt = df['apt'][:1].values[0]
        year = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        mon = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        ymd = []  # ['2011-01', '2011-02', '2011-03', '2011-04' ...
        
        for i in range(len(year)):
            for j in range(len(mon)):
                ymd.append(year[i] + mon[j])
                
        mean = []
        
        for i in ymd:
            condition = (df['ymd']== (i)) # 조건식 작성
        
            mp = list(df[condition].price) # 월 별 거래가격 리스트
            
            if len(mp) == 0:
                mean.append(0)
        
            else: 
                mv = round(sum(mp)/len(mp))
                mean.append(mv)

        for i in range(len(mean)):
            if mean[i] == 0:
                ymd[i] = '0' # 리스트 mean이 0인 인덱스에 똑같이 0 넣기
                                       
        while 0 in mean:    # mean에서 값이 0인것 빼기
            mean.remove(0)
            
        while '0' in ymd:    # ymd에서 값이 '0'인것 빼기
            ymd.remove('0')
        
        for i in range(len(ymd)):
            ymd[i] = int(ymd[i])
        
        gu = df['gu'][:1].values[0]
       
    return render(request, 'graph.html', {'addr':detailaddr, 'mean':mean, 'ymd':ymd,'apt':apt,'gu':gu})

@csrf_exempt
def dongmaker(request):
    dong = request.POST.get('clickedDong')
    datas = Addrapt.objects.filter(dong__contains=dong).values()
    df = pd.DataFrame(datas)
    
    dong = list(df['addr']) # 동으로 데이터 불러와서 아파트 주소 가져오기
    apt = list(df['apt']) # 불러온 아파트 이름 가져오기

    return JsonResponse({'dong':dong, 'apt':apt, 'df':df.to_html()})

    