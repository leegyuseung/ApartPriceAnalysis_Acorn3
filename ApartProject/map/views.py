from django.shortcuts import render
from map.models import Test, Addrdata, Addrapt
import pandas as pd
import json
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.http.response import HttpResponse, JsonResponse

# Create your views here.
def Main(request):
    return render(request,'home.html')

def cssTest(request):
    return render(request,'index.html')

@csrf_exempt
def apart(request):
    search = request.POST['search']
    datas = Addrapt.objects.filter(apt__contains=search).values()
    df = pd.DataFrame(datas)
    print(df)
    
    apt = [i for i in df['apt'] + df['dong']]
    juso = [i for i in df['addr']]
    
    
    aptJusoJson = {}
    for apt, juso in zip(apt, juso):
        aptJusoJson[apt]=  juso 
    
    print(aptJusoJson)
    apt = [i for i in df['apt'] + df['dong']]
    
    return JsonResponse({'juso':juso, 'apartdata':apt, 'aptJusoJson':aptJusoJson})

def importData(request):
    # 클릭한 마커의 데이터 불러오기
    if request.method == 'GET':
        
        detailaddr = request.GET['aptName']
        
        datas = Addrdata.objects.filter(addr__contains=detailaddr).values()
        
        df = pd.DataFrame(datas)
        df.set_index(df['num'], inplace=True)
        df = df.drop(['num'], axis=1)
        
        
        print(df)
        
        year = ['2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
        mon = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        
        ymd = []  # ['201101', '201102', '201103', '201104' ...
        
        for i in range(len(year)):
            for j in range(len(mon)):
                ymd.append(year[i] + mon[j])
        
        mean = []
        for i in ymd:
            condition = (df['ymd']== (i)) # 조건식 작성
        
            a = list(df[df['ymd']== (i)].price)
            
            if len(a) == 0:
                mean.append(0)
        
            else: 
                mm = round(sum(a)/len(a))
                mean.append(mm)
        
        print(mean)
        
        print(len(mean))
        
    return render(request, 'graph.html', {'addr':detailaddr, 'datas':df.to_html(), 'mean':mean})










'''
def modeling(request):
    importData(request) # db에서 특정 아파트 정보 불러오기
    
    
    
    return redirect('')

def graph(request):
    
    return render(request, 'graph.html')
'''