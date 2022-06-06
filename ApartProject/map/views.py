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
    paginator = Paginator(apt, 30)
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