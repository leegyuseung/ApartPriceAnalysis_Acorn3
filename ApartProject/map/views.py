from django.shortcuts import render
from map.models import Test
import pandas as pd
import json
from django.views.decorators.csrf import csrf_exempt
from django.http.response import HttpResponse

# Create your views here.
def Main(request):

    return render(request,'home.html')

@csrf_exempt
def apart(request):
    search = request.POST['search']
    datas = Test.objects.filter(apart__contains=search).values()
    df = pd.DataFrame(datas)
    df2 = df.to_html()

    context = {'df':df2}
    print(context)
    return HttpResponse(json.dumps(context), content_type='application/json')
