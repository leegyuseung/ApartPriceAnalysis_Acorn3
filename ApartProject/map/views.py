from django.shortcuts import render
from map.models import Addrdata, Addrapt
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.http.response import JsonResponse
from django.core.paginator import Paginator

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
    print(tojson)
    # json으로 리턴
    return JsonResponse({'juso':juso, 'apartdata':apt, 'aptJusoJson':aptJusoJson, 'apt_lists':tojson})



@csrf_exempt
def polygon(request):
    import json
    with open ("C:/Users/SAMSUNG/OneDrive/바탕 화면/프로젝트 데이터/시구분데이터/polygonData.json", "r", encoding='utf-8') as f:
        Gpolygon = json.load(f)
    
    
    return JsonResponse({'polygon':Gpolygon})

@csrf_exempt
def Dpolygon(request):
    import json
    
    with open ("C:/Users/SAMSUNG/OneDrive/바탕 화면/프로젝트 데이터/동구분데이터/polygonDong.json", "r", encoding='utf-8') as f:
        Dpolygon = json.load(f)
    
    return JsonResponse({'Dpolygon':Dpolygon})

@csrf_exempt
def pred(request):
    year = request.POST['year']
    # new_val = pd.DataFrame({'year':[year]})
    print(year)

    return JsonResponse({'new_val':year})


def importData(request):
    # 클릭한 마커의 데이터 불러오기
    if request.method == 'GET':
        
        detailaddr = request.GET['aptName']
        
        datas = Addrdata.objects.filter(addr__contains=detailaddr).values()
        
        df = pd.DataFrame(datas)
        df.set_index(df['num'], inplace=True)
        df = df.drop(['num'], axis=1)
        df = df.sort_values(by = 'ymd')
        
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

   
        
    return render(request, 'graph.html', {'addr':detailaddr, 'datas':df.to_html(), 'mean':mean, 'ymd':ymd})
