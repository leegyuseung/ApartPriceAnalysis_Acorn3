from django.shortcuts import render, redirect
from qna.models import BoardTab
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from datetime import datetime
from users.models import Users

# Create your views here.

def listFunc(request):
    data_all = BoardTab.objects.all().order_by('-id')
    
    paginator = Paginator(data_all, 10)
    page = request.GET.get('page')
    try:
        datas = paginator.page(page)
    except PageNotAnInteger:
        datas = paginator.page(1)
    except EmptyPage:
        datas = paginator.page(paginator.num_pages)
        
    return render(request, 'list.html', {'datas':datas})


def insertFunc(request):
    try:
        if request.session['user']:
            userId = request.session['user']
            print(userId)
            data = Users.objects.get(id=userId)
            return render(request, 'insert.html',{'data': data})
        else:
            return render(request, '/users/login')
        
        
    except Exception as e:
        print('수정자료 읽기 오류',e)
        return render(request,'error.html')
    
    
    return render(request, 'insert.html',{'data': userId})

def insertOkFunc(request):
    if request.method == 'POST':
        try:
            gbun = 1   # group number 구하기
            datas = BoardTab.objects.all()
            if datas.count() != 0:
                gbun = BoardTab.objects.latest('id').id + 1
            
            userId = Users.objects.get(id = request.session['user'])
            BoardTab(
                userId = userId,
                title = request.POST.get('title'),
                cont = request.POST.get('cont'),
                bip = request.META['REMOTE_ADDR'],
                bdate = datetime.now(),
                readcnt = 0,
                gnum = gbun,
                onum = 0,
                nested = 0,
            ).save()
        except Exception as e:
            print('추가 에러 : ', e)
            return render(request, 'error.html')
    
    return redirect('/qna/list')   # 추가 후 목록 보기

def searchFunc(request):
    if request.method == 'POST':
        s_type = request.POST.get('s_type')
        s_value = request.POST.get('s_value')
        print(s_type, s_value)
        
        if s_type =='title':
            datas_search = BoardTab.objects.filter(title__contains = s_value).order_by('-id')
        elif s_type == 'name':
            datas_search = BoardTab.objects.filter(name__contains = s_value).order_by('-id')
        
        paginator = Paginator(datas_search, 10)
        page = request.GET.get('page')
        try:
            datas = paginator.page(page)
        except PageNotAnInteger:
            datas = paginator.page(1)
        except EmptyPage:
            datas = paginator.page(paginator.num_pages)
            
    return render(request, 'list.html', {'datas':datas})

def contentFunc(request):
    page = request.GET.get('page')
    data = BoardTab.objects.get(id=request.GET.get('id'))
    data.readcnt += 1
    data.save() #조회수 갱신
    return render(request, 'content.html', {'data_one': data, 'page': page})

def updateFunc(request):
    try:
        data = BoardTab.objects.get(id=request.GET.get('id'))
    except Exception as e:
        print('수정자료 읽기 오류',e)
        return render(request,'error.html')
    
    return render(request, 'update.html', {'data_one': data})

def updateOkFunc(request):
    try:
        upRec = BoardTab.objects.get(id=request.POST.get('id'))
        
        # 비밀번호 비교 후 수정 처리
        
    except Exception as e:
        print('수정자료 읽기 오류',e)
        return render(request,'error.html')
    
    return redirect('/board/list')   # 수정 후 목록 보기

def deleteFunc(request):
    pass

def deleteOkFunc(request):
    pass