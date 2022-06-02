from django.shortcuts import render
from map.models import Test, Addrdata
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
    datas = Addrdata.objects.filter(apt__contains=search).values()
    df = pd.DataFrame(datas)

    apt = [i for i in df['apt']]
    juso = [i for i in df['addr']]
    
    apt1 = apt
    aptJusoJson = {}
    for apt, juso in zip(apt1, juso):
        aptJusoJson[apt]=juso
    
    apt = [i for i in df['apt']]
    
    return JsonResponse({'juso':juso, 'apartdata':apt, 'aptJusoJson':aptJusoJson})

