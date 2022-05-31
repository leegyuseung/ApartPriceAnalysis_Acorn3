from django.shortcuts import render
from map.models import Test
import pandas as pd
import json
import numpy as np
from django.views.decorators.csrf import csrf_exempt
from django.http.response import HttpResponse, JsonResponse

# Create your views here.
def Main(request):
    return render(request,'home.html')

@csrf_exempt
def apart(request):
    search = request.POST['search']
    datas = Test.objects.filter(apart__contains=search).values()
    df = pd.DataFrame(datas)
    
    apartdata = df['apart'][0]
    jusodata = df['juso'][0]
    
    return JsonResponse({'juso':jusodata, 'apartdata':apartdata})
    
def cssTest(request):
    return render(request,'index.html')