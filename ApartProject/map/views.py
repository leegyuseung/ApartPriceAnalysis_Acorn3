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

