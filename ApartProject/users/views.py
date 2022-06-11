from django.shortcuts import render, redirect
from users.models import Users

# Create your views here.
def signUp(request):
    return render(request,'signup.html')

def signUpOk(request):
    if request.method == "POST":
        idcheck = request.POST.get('id')
        id2 = Users.objects.filter(id=idcheck)
        
        if id2.count() != 0:
            
            errors = '존재하는 아이디입니다.'
            return render(request, 'signup.html', {'msg':errors})
        else:
            try:            
                Users(
                    id = request.POST.get('id'),
                    pw = request.POST.get('pw'),
                    email = request.POST.get('email'),
                    name = request.POST.get('name'),
                    phone = request.POST.get('phone'),
                    addr = request.POST.get('addr'),
                ).save()
            
            except Exception as e:
                print('추가 에러:',e)
                return render(request,'error.html')
        
    return redirect('/')
def login(request):
    return render(request, 'loginform.html')

def loginOk(request):
    if request.method == "POST":
        id = request.POST.get('id')
        pw = request.POST.get('pw')
        
        id2 = Users.objects.filter(id=id, pw=pw)
        
        if id2.count() != 0:
            # 비밀번호가 일치하면 session을 사용해 user.id 를 넘겨준다.
            request.session['user'] = id2[0].id

            # 로그인 성공 후 127.0.0.1:8000/ 이동   
            return redirect('/')
        else:
            msg = '아이디가 없거나 비밀번호가 틀렸습니다.'
            return render(request, 'loginform.html', {'msg':msg})

def logout(request):
    if request.session['user']:
        del(request.session['user'])
    return render(request, 'logout.html')

def myPage(request):
    if request.session['user']:
        id = request.session['user']
        id2 = Users.objects.get(id = id)

    return render(request, 'mypage.html', {'data':id2})

def su(request):
    if request.session['user']:
        id = request.session['user']
        id2 = Users.objects.get(id = id)
    return render(request, 'su.html', {'data':id2})

def suOk(request):
    if request.method == 'POST':
        id = request.session['user']
        pw = request.POST.get('pw')
        up = Users.objects.get(id = id)

        if pw == up.pw:
            up.email=request.POST.get('email')
            up.name=request.POST.get('name')
            up.phone=request.POST.get('phone')
            up.addr=request.POST.get('addr')
            up.save()
        else:
            msg = "비밀번호가 틀렸습니다"
            return render(request, 'mypage.html', {'msg':msg})
    return redirect('/users/mypage')

def tal(request):
    if request.session['user']:
        id = request.session['user']
        taldata = Users.objects.get(id=id)
        taldata.delete()
        del(request.session['user'])
        
    return render(request, 'tal.html')

def pwC(request):
    if request.session['user']:
        id = request.session['user']
        id2 = Users.objects.get(id = id)
    return render(request, 'pwC.html', {'data':id2})

def pwChange(request):
    if request.method == 'POST':
        id = request.session['user']
        id2 = Users.objects.get(id = id)
        pw = request.POST.get('pw')
        if pw == id2.pw:
            id2.pw=request.POST.get('newpw')
            id2.save()
        else:
            msg = "비밀번호가 틀렸습니다"
            return render(request, 'mypage.html', {'msg':msg})
    return redirect('/users/mypage')